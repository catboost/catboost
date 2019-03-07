#include "codecs.h"
#include "common.h"
#include "legacy.h"

#include <contrib/libs/lz4/lz4.h>
#include <contrib/libs/lz4/lz4hc.h>
#include <contrib/libs/lz4/generated/iface.h>
#include <contrib/libs/fastlz/fastlz.h>
#include <contrib/libs/snappy/snappy.h>
#include <contrib/libs/zlib/zlib.h>
#include <contrib/libs/lzmasdk/LzmaLib.h>
#include <contrib/libs/libbz2/bzlib.h>
#include <contrib/libs/brotli/include/brotli/encode.h>
#include <contrib/libs/brotli/include/brotli/decode.h>


#define ZSTD_STATIC_LINKING_ONLY
#include <contrib/libs/zstd/zstd.h>

#include <util/ysaveload.h>
#include <util/stream/null.h>
#include <util/stream/mem.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/system/align.h>
#include <util/system/unaligned_mem.h>
#include <util/generic/hash.h>
#include <util/generic/cast.h>
#include <util/generic/deque.h>
#include <util/generic/buffer.h>
#include <util/generic/region.h>
#include <util/generic/singleton.h>
#include <util/generic/algorithm.h>
#include <util/generic/mem_copy.h>

using namespace NBlockCodecs;

namespace {
    // lz4 codecs
    struct TLz4Base {
        static inline size_t DoMaxCompressedLength(size_t in) {
            return LZ4_compressBound(SafeIntegerCast<int>(in));
        }
    };

    struct TLz4FastCompress {
        inline TLz4FastCompress(int memory)
            : Memory(memory)
            , Methods(LZ4Methods(Memory))
        {
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            return Methods->LZ4CompressLimited(in.data(), (char*)buf, in.size(), LZ4_compressBound(in.size()));
        }

        inline TString CPrefix() {
            return "fast" + ToString(Memory);
        }

        const int Memory;
        const TLZ4Methods* Methods;
    };

    struct TLz4BestCompress {
        inline size_t DoCompress(const TData& in, void* buf) const {
            return LZ4_compress_HC(in.data(), (char*)buf, in.size(), LZ4_compressBound(in.size()), 0);
        }

        static inline TString CPrefix() {
            return "hc";
        }
    };

    struct TLz4FastDecompress {
        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            ssize_t res = LZ4_decompress_fast(in.data(), (char*)out, len);
            if (res < 0) {
                ythrow TDecompressError(res);
            }
        }

        static inline TStringBuf DPrefix() {
            return AsStringBuf("fast");
        }
    };

    struct TLz4SafeDecompress {
        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            ssize_t res = LZ4_decompress_safe(in.data(), (char*)out, in.size(), len);
            if (res < 0) {
                ythrow TDecompressError(res);
            }
        }

        static inline TStringBuf DPrefix() {
            return AsStringBuf("safe");
        }
    };

    template <class TC, class TD>
    struct TLz4Codec: public TAddLengthCodec<TLz4Codec<TC, TD>>, public TLz4Base, public TC, public TD {
        inline TLz4Codec()
            : MyName("lz4-" + TC::CPrefix() + "-" + TD::DPrefix())
        {
        }

        template <class T>
        inline TLz4Codec(const T& t)
            : TC(t)
            , MyName("lz4-" + TC::CPrefix() + "-" + TD::DPrefix())
        {
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const TString MyName;
    };

    //fastlz codecs
    struct TFastLZCodec: public TAddLengthCodec<TFastLZCodec> {
        inline TFastLZCodec(int level)
            : MyName("fastlz-" + ToString(level))
            , Level(level)
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            return Max<size_t>(in + in / 20, 128);
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            if (Level) {
                return fastlz_compress_level(Level, in.data(), in.size(), buf);
            }

            return fastlz_compress(in.data(), in.size(), buf);
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            const int ret = fastlz_decompress(in.data(), in.size(), out, len);

            if (ret < 0 || (size_t)ret != len) {
                ythrow TDataError() << AsStringBuf("can not decompress");
            }
        }

        const TString MyName;
        const int Level;
    };

    // snappy codec
    struct TSnappyCodec: public ICodec {
        size_t DecompressedLength(const TData& in) const override {
            size_t ret;

            if (snappy::GetUncompressedLength(in.data(), in.size(), &ret)) {
                return ret;
            }

            ythrow TDecompressError(0);
        }

        size_t MaxCompressedLength(const TData& in) const override {
            return snappy::MaxCompressedLength(in.size());
        }

        size_t Compress(const TData& in, void* out) const override {
            size_t ret;

            snappy::RawCompress(in.data(), in.size(), (char*)out, &ret);

            return ret;
        }

        size_t Decompress(const TData& in, void* out) const override {
            if (snappy::RawUncompress(in.data(), in.size(), (char*)out)) {
                return DecompressedLength(in);
            }

            ythrow TDecompressError(0);
        }

        TStringBuf Name() const noexcept override {
            return "snappy";
        }
    };

    // zlib codecs
    struct TZLibCodec: public TAddLengthCodec<TZLibCodec> {
        inline TZLibCodec(int level)
            : MyName("zlib-" + ToString(level))
            , Level(level)
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            return compressBound(in);
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            //TRASH detected
            uLong ret = Max<unsigned int>();

            int cres = compress2((Bytef*)buf, &ret, (const Bytef*)in.data(), in.size(), Level);

            if (cres != Z_OK) {
                ythrow TCompressError(cres);
            }

            return ret;
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            uLong ret = len;

            int uncres = uncompress((Bytef*)out, &ret, (const Bytef*)in.data(), in.size());
            if (uncres != Z_OK) {
                ythrow TDecompressError(uncres);
            }

            if (ret != len) {
                ythrow TDecompressError(len, ret);
            }
        }

        const TString MyName;
        const int Level;
    };

    // lzma codecs
    struct TLzmaCodec: public TAddLengthCodec<TLzmaCodec> {
        inline TLzmaCodec(int level)
            : Level(level)
            , MyName("lzma-" + ToString(Level))
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            return Max<size_t>(in + in / 20, 128) + LZMA_PROPS_SIZE;
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            unsigned char* props = (unsigned char*)buf;
            unsigned char* data = props + LZMA_PROPS_SIZE;
            size_t destLen = Max<size_t>();
            size_t outPropsSize = LZMA_PROPS_SIZE;

            const int ret = LzmaCompress(data, &destLen, (const unsigned char*)in.data(), in.size(), props, &outPropsSize, Level, 0, -1, -1, -1, -1, -1);

            if (ret != SZ_OK) {
                ythrow TCompressError(ret);
            }

            return destLen + LZMA_PROPS_SIZE;
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            if (in.size() <= LZMA_PROPS_SIZE) {
                ythrow TDataError() << AsStringBuf("broken lzma stream");
            }

            const unsigned char* props = (const unsigned char*)in.data();
            const unsigned char* data = props + LZMA_PROPS_SIZE;
            size_t destLen = len;
            SizeT srcLen = in.size() - LZMA_PROPS_SIZE;

            const int res = LzmaUncompress((unsigned char*)out, &destLen, data, &srcLen, props, LZMA_PROPS_SIZE);

            if (res != SZ_OK) {
                ythrow TDecompressError(res);
            }

            if (destLen != len) {
                ythrow TDecompressError(len, destLen);
            }
        }

        const int Level;
        const TString MyName;
    };

    // bzip2 codecs
    struct TBZipCodec: public TAddLengthCodec<TBZipCodec> {
        inline TBZipCodec(int level)
            : Level(level)
            , MyName("bzip2-" + ToString(Level))
        {
        }

        static inline size_t DoMaxCompressedLength(size_t in) noexcept {
            // very strange
            return in * 2 + 128;
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        inline size_t DoCompress(const TData& in, void* buf) const {
            unsigned int ret = DoMaxCompressedLength(in.size());
            const int res = BZ2_bzBuffToBuffCompress((char*)buf, &ret, (char*)in.data(), in.size(), Level, 0, 0);
            if (res != BZ_OK) {
                ythrow TCompressError(res);
            }

            return ret;
        }

        inline void DoDecompress(const TData& in, void* out, size_t len) const {
            unsigned int tmp = SafeIntegerCast<unsigned int>(len);
            const int res = BZ2_bzBuffToBuffDecompress((char*)out, &tmp, (char*)in.data(), in.size(), 0, 0);

            if (res != BZ_OK) {
                ythrow TDecompressError(res);
            }

            if (len != tmp) {
                ythrow TDecompressError(len, tmp);
            }
        }

        const int Level;
        const TString MyName;
    };

    struct TZStd08Codec: public TAddLengthCodec<TZStd08Codec> {
        inline TZStd08Codec(unsigned level)
            : Level(level)
            , MyName(AsStringBuf("zstd08_") + ToString(Level))
        {
        }

        static inline size_t CheckError(size_t ret, const char* what) {
            if (ZSTD_isError(ret)) {
                ythrow yexception() << what << AsStringBuf(" zstd error: ") << ZSTD_getErrorName(ret);
            }

            return ret;
        }

        static inline size_t DoMaxCompressedLength(size_t l) noexcept {
            return ZSTD_compressBound(l);
        }

        inline size_t DoCompress(const TData& in, void* out) const {
            return CheckError(ZSTD_compress(out, DoMaxCompressedLength(in.size()), in.data(), in.size(), Level), "compress");
        }

        inline void DoDecompress(const TData& in, void* out, size_t dsize) const {
            const size_t res = CheckError(ZSTD_decompress(out, dsize, in.data(), in.size()), "decompress");

            if (res != dsize) {
                ythrow TDecompressError(dsize, res);
            }
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const unsigned Level;
        const TString MyName;
    };

    struct TBrotliCodec : public TAddLengthCodec<TBrotliCodec> {
        static constexpr int BEST_QUALITY = 11;

        inline TBrotliCodec(ui32 level)
            : Quality(level)
            , MyName(AsStringBuf("brotli_") + ToString(level))
        {
        }

        static inline size_t DoMaxCompressedLength(size_t l) noexcept {
            return BrotliEncoderMaxCompressedSize(l);
        }

        inline size_t DoCompress(const TData& in, void* out) const {
            size_t resultSize = MaxCompressedLength(in);
            auto result = BrotliEncoderCompress(
                                /*quality*/ Quality,
                                /*window*/ BROTLI_DEFAULT_WINDOW,
                                /*mode*/ BrotliEncoderMode::BROTLI_MODE_GENERIC,
                                /*input_size*/ in.size(),
                                /*input_buffer*/ (const unsigned char*)(in.data()),
                                /*encoded_size*/ &resultSize,
                                /*encoded_buffer*/ static_cast<unsigned char*>(out));
            if (result != BROTLI_TRUE) {
                ythrow yexception() << "internal brotli error during compression";
            }

            return resultSize;
        }

        inline void DoDecompress(const TData& in, void* out, size_t dsize) const {
            size_t decoded = dsize;
            auto result = BrotliDecoderDecompress(in.size(), (const unsigned char*)in.data(), &decoded, static_cast<unsigned char*>(out));
            if (result != BROTLI_DECODER_RESULT_SUCCESS) {
                ythrow yexception() << "internal brotli error during decompression";
            } else if (decoded != dsize) {
                ythrow TDecompressError(dsize, decoded);
            }
        }

        TStringBuf Name() const noexcept override {
            return MyName;
        }

        const int Quality = BEST_QUALITY;
        const TString MyName;
    };

    // end of codecs

    struct TCodecFactory {
        inline TCodecFactory() {
            Add(&Null);
            Add(&Snappy);

            for (int i = 0; i < 30; ++i) {
                typedef TLz4Codec<TLz4FastCompress, TLz4FastDecompress> T1;
                typedef TLz4Codec<TLz4FastCompress, TLz4SafeDecompress> T2;

                THolder<T1> t1(new T1(i));
                THolder<T2> t2(new T2(i));

                if (t1->Methods) {
                    Codecs.push_back(t1.Release());
                }

                if (t2->Methods) {
                    Codecs.push_back(t2.Release());
                }
            }

            Codecs.push_back(new TLz4Codec<TLz4BestCompress, TLz4FastDecompress>());
            Codecs.push_back(new TLz4Codec<TLz4BestCompress, TLz4SafeDecompress>());

            for (int i = 0; i < 3; ++i) {
                Codecs.push_back(new TFastLZCodec(i));
            }

            for (int i = 0; i < 10; ++i) {
                Codecs.push_back(new TZLibCodec(i));
            }

            for (int i = 1; i < 10; ++i) {
                Codecs.push_back(new TBZipCodec(i));
            }

            for (int i = 0; i < 10; ++i) {
                Codecs.push_back(new TLzmaCodec(i));
            }

            for (auto& codec : LegacyZStd06Codec()) {
                Codecs.emplace_back(std::move(codec));
            }

            for (int i = 1; i <= ZSTD_maxCLevel(); ++i) {
                Codecs.push_back(new TZStd08Codec(i));
            }

            for (int i = 1; i <= TBrotliCodec::BEST_QUALITY; ++i) {
                Codecs.push_back(new TBrotliCodec(i));
            }

            for (size_t i = 0; i < Codecs.size(); ++i) {
                Add(Codecs[i].Get());
            }

            // aliases
            Registry["fastlz"] = Registry["fastlz-0"];
            Registry["zlib"] = Registry["zlib-6"];
            Registry["bzip2"] = Registry["bzip2-6"];
            Registry["lzma"] = Registry["lzma-5"];
            Registry["lz4-fast-safe"] = Registry["lz4-fast14-safe"];
            Registry["lz4-fast-fast"] = Registry["lz4-fast14-fast"];
            Registry["lz4"] = Registry["lz4-fast-safe"];
            Registry["lz4fast"] = Registry["lz4-fast-fast"];
            Registry["lz4hc"] = Registry["lz4-hc-safe"];

            for (int i = 1; i <= ZSTD_maxCLevel(); ++i) {
                Alias("zstd_" + ToString(i), "zstd08_" + ToString(i));
            }
        }

        inline const ICodec* Find(const TStringBuf& name) const {
            auto it = Registry.find(name);

            if (it == Registry.end()) {
                ythrow TNotFound() << "can not found " << name << " codec";
            }

            return it->second;
        }

        inline void ListCodecs(TCodecList& lst) const {
            for (const auto& it : Registry) {
                lst.push_back(it.first);
            }

            Sort(lst.begin(), lst.end());
        }

        inline void Add(ICodec* codec) {
            Registry[codec->Name()] = codec;
        }

        inline void Alias(TStringBuf from, TStringBuf to) {
            Tmp.emplace_back(from);
            Registry[Tmp.back()] = Registry[to];
        }

        TDeque<TString> Tmp;
        TNullCodec Null;
        TSnappyCodec Snappy;
        TVector<TCodecPtr> Codecs;
        typedef THashMap<TStringBuf, ICodec*> TRegistry;
        TRegistry Registry;

        // SEARCH-8344: Global decompressed size limiter (to prevent remote DoS)
        size_t MaxPossibleDecompressedLength = Max<size_t>();
    };
}

const ICodec* NBlockCodecs::Codec(const TStringBuf& name) {
    return Singleton<TCodecFactory>()->Find(name);
}

TCodecList NBlockCodecs::ListAllCodecs() {
    TCodecList ret;

    Singleton<TCodecFactory>()->ListCodecs(ret);

    return ret;
}

TString NBlockCodecs::ListAllCodecsAsString() {
    return JoinSeq(AsStringBuf(","), ListAllCodecs());
}

void NBlockCodecs::SetMaxPossibleDecompressedLength(size_t maxPossibleDecompressedLength) {
    Singleton<TCodecFactory>()->MaxPossibleDecompressedLength = maxPossibleDecompressedLength;
}

size_t NBlockCodecs::GetMaxPossibleDecompressedLength() {
    return Singleton<TCodecFactory>()->MaxPossibleDecompressedLength;
}

size_t ICodec::GetDecompressedLength(const TData& in) const {
    const size_t len = DecompressedLength(in);

    Y_ENSURE(
        len <= NBlockCodecs::GetMaxPossibleDecompressedLength(),
        "Attempt to decompress the block that is larger than maximum possible decompressed length, "
        "see SEARCH-8344 for details. "
    );
    return len;
}

void ICodec::Encode(const TData& in, TBuffer& out) const {
    const size_t maxLen = MaxCompressedLength(in);

    out.Reserve(maxLen);
    out.Resize(Compress(in, out.Data()));
}

void ICodec::Decode(const TData& in, TBuffer& out) const {
    const size_t len = GetDecompressedLength(in);

    out.Reserve(len);
    out.Resize(Decompress(in, out.Data()));
}

void ICodec::Encode(const TData& in, TString& out) const {
    const size_t maxLen = MaxCompressedLength(in);

    out.reserve(maxLen);
    out.ReserveAndResize(Compress(in, out.begin()));
}

void ICodec::Decode(const TData& in, TString& out) const {
    const size_t len = GetDecompressedLength(in);

    out.reserve(len);
    out.ReserveAndResize(Decompress(in, out.begin()));
}

ICodec::~ICodec() = default;
