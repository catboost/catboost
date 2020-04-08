#include "lz.h"

#include <util/system/yassert.h>
#include <util/system/byteorder.h>
#include <util/memory/addstorage.h>
#include <util/generic/utility.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/stream/mem.h>

#include <contrib/libs/lz4/lz4.h>
#include <contrib/libs/fastlz/fastlz.h>
#include <contrib/libs/snappy/snappy.h>
#include <contrib/libs/quicklz/quicklz.h>
#include <contrib/libs/minilzo/minilzo.h>

static inline ui8 HostToLittle(ui8 t) noexcept {
    return t;
}

static inline ui8 LittleToHost(ui8 t) noexcept {
    return t;
}

struct TCommonData {
    static const size_t overhead = sizeof(ui16) + sizeof(ui8);
};

const size_t SIGNATURE_SIZE = 4;

template <class TCompressor, class TBase>
class TCompressorBase: public TAdditionalStorage<TCompressorBase<TCompressor, TBase>>, public TCompressor, public TCommonData {
public:
    inline TCompressorBase(IOutputStream* slave, ui16 blockSize)
        : Slave_(slave)
        , BlockSize_(blockSize)
    {
        /*
         * save signature
         */
        static_assert(sizeof(TCompressor::signature) - 1 == SIGNATURE_SIZE, "expect sizeof(TCompressor::signature) - 1 == SIGNATURE_SIZE");
        Slave_->Write(TCompressor::signature, sizeof(TCompressor::signature) - 1);

        /*
         * save version
         */
        this->Save((ui32)1);

        /*
         * save block size
         */
        this->Save(BlockSize());
    }

    inline ~TCompressorBase() {
    }

    inline void Write(const char* buf, size_t len) {
        while (len) {
            const ui16 toWrite = (ui16)Min<size_t>(len, this->BlockSize());

            this->WriteBlock(buf, toWrite);

            buf += toWrite;
            len -= toWrite;
        }
    }

    inline void Flush() {
    }

    inline void Finish() {
        this->Flush();
        this->WriteBlock(nullptr, 0);
    }

    template <class T>
    static inline void Save(T t, IOutputStream* out) {
        t = HostToLittle(t);

        out->Write(&t, sizeof(t));
    }

    template <class T>
    inline void Save(T t) {
        Save(t, Slave_);
    }

private:
    inline void* Block() const noexcept {
        return this->AdditionalData();
    }

    inline ui16 BlockSize() const noexcept {
        return BlockSize_;
    }

    inline void WriteBlock(const void* ptr, ui16 len) {
        Y_ASSERT(len <= this->BlockSize());

        ui8 compressed = false;

        if (len) {
            const size_t out = this->Compress((const char*)ptr, len, (char*)Block(), this->AdditionalDataLength());
            // catch compressor buffer overrun (e.g. SEARCH-2043)
            //Y_VERIFY(out <= this->Hint(this->BlockSize()));

            if (out < len || TCompressor::SaveIncompressibleChunks()) {
                compressed = true;
                ptr = Block();
                len = (ui16)out;
            }
        }

        char tmp[overhead];
        TMemoryOutput header(tmp, sizeof(tmp));

        this->Save(len, &header);
        this->Save(compressed, &header);

        using TPart = IOutputStream::TPart;
        if (ptr) {
            const TPart parts[] = {
                TPart(tmp, sizeof(tmp)),
                TPart(ptr, len),
            };

            Slave_->Write(parts, sizeof(parts) / sizeof(*parts));
        } else {
            Slave_->Write(tmp, sizeof(tmp));
        }
    }

private:
    IOutputStream* Slave_;
    const ui16 BlockSize_;
};

template <class T>
static inline T GLoad(IInputStream* input) {
    T t;

    if (input->Load(&t, sizeof(t)) != sizeof(t)) {
        ythrow TDecompressorError() << "stream error";
    }

    return LittleToHost(t);
}

class TDecompressSignature {
public:
    inline TDecompressSignature(IInputStream* input) {
        if (input->Load(Buffer_, SIGNATURE_SIZE) != SIGNATURE_SIZE) {
            ythrow TDecompressorError() << "can not load stream signature";
        }
    }

    template <class TDecompressor>
    inline bool Check() const {
        static_assert(sizeof(TDecompressor::signature) - 1 == SIGNATURE_SIZE, "expect sizeof(TDecompressor::signature) - 1 == SIGNATURE_SIZE");
        return memcmp(TDecompressor::signature, Buffer_, SIGNATURE_SIZE) == 0;
    }

private:
    char Buffer_[SIGNATURE_SIZE];
};

template <class TDecompressor>
static inline IInputStream* ConsumeSignature(IInputStream* input) {
    TDecompressSignature sign(input);
    if (!sign.Check<TDecompressor>()) {
        ythrow TDecompressorError() << "incorrect signature";
    }
    return input;
}

template <class TDecompressor>
class TDecompressorBaseImpl: public TDecompressor, public TCommonData {
public:
    static inline ui32 CheckVer(ui32 v) {
        if (v != 1) {
            ythrow yexception() << AsStringBuf("incorrect stream version: ") << v;
        }

        return v;
    }

    inline TDecompressorBaseImpl(IInputStream* slave)
        : Slave_(slave)
        , Input_(nullptr, 0)
        , Eof_(false)
        , Version_(CheckVer(Load<ui32>()))
        , BlockSize_(Load<ui16>())
        , OutBufSize_(TDecompressor::Hint(BlockSize_))
        , Tmp_(::operator new(2 * OutBufSize_))
        , In_((char*)Tmp_.Get())
        , Out_(In_ + OutBufSize_)
    {
        this->InitFromStream(Slave_);
    }

    inline ~TDecompressorBaseImpl() {
    }

    inline size_t Read(void* buf, size_t len) {
        size_t ret = Input_.Read(buf, len);

        if (ret) {
            return ret;
        }

        if (Eof_) {
            return 0;
        }

        this->FillNextBlock();

        ret = Input_.Read(buf, len);

        if (ret) {
            return ret;
        }

        Eof_ = true;

        return 0;
    }

    inline void FillNextBlock() {
        char tmp[overhead];

        if (Slave_->Load(tmp, sizeof(tmp)) != sizeof(tmp)) {
            ythrow TDecompressorError() << "can not read block header";
        }

        TMemoryInput header(tmp, sizeof(tmp));

        const ui16 len = GLoad<ui16>(&header);
        const ui8 compressed = GLoad<ui8>(&header);

        if (compressed > 1) {
            ythrow TDecompressorError() << "broken header";
        }

        if (Slave_->Load(In_, len) != len) {
            ythrow TDecompressorError() << "can not read data";
        }

        if (compressed) {
            const size_t ret = this->Decompress(In_, len, Out_, OutBufSize_);

            Input_.Reset(Out_, ret);
        } else {
            Input_.Reset(In_, len);
        }
    }

    template <class T>
    inline T Load() {
        return GLoad<T>(Slave_);
    }

protected:
    IInputStream* Slave_;
    TMemoryInput Input_;
    bool Eof_;
    const ui32 Version_;
    const ui16 BlockSize_;
    const size_t OutBufSize_;
    THolder<void> Tmp_;
    char* In_;
    char* Out_;
};

template <class TDecompressor, class TBase>
class TDecompressorBase: public TDecompressorBaseImpl<TDecompressor> {
public:
    inline TDecompressorBase(IInputStream* slave)
        : TDecompressorBaseImpl<TDecompressor>(ConsumeSignature<TDecompressor>(slave))
    {
    }

    inline ~TDecompressorBase() {
    }
};

#define DEF_COMPRESSOR_COMMON(rname, name)                              \
    rname::~rname() {                                                   \
        try {                                                           \
            Finish();                                                   \
        } catch (...) {                                                 \
        }                                                               \
    }                                                                   \
                                                                        \
    void rname::DoWrite(const void* buf, size_t len) {                  \
        if (!Impl_) {                                                   \
            ythrow yexception() << "can not write to finalized stream"; \
        }                                                               \
                                                                        \
        Impl_->Write((const char*)buf, len);                            \
    }                                                                   \
                                                                        \
    void rname::DoFlush() {                                             \
        if (!Impl_) {                                                   \
            ythrow yexception() << "can not flush finalized stream";    \
        }                                                               \
                                                                        \
        Impl_->Flush();                                                 \
    }                                                                   \
                                                                        \
    void rname::DoFinish() {                                            \
        THolder<TImpl> impl(Impl_.Release());                           \
                                                                        \
        if (impl) {                                                     \
            impl->Finish();                                             \
        }                                                               \
    }

#define DEF_COMPRESSOR(rname, name)                                     \
    class rname::TImpl: public TCompressorBase<name, TImpl> {           \
    public:                                                             \
        inline TImpl(IOutputStream* out, ui16 blockSize)                \
            : TCompressorBase<name, TImpl>(out, blockSize) {            \
        }                                                               \
    };                                                                  \
                                                                        \
    rname::rname(IOutputStream* slave, ui16 blockSize)                  \
        : Impl_(new (TImpl::Hint(blockSize)) TImpl(slave, blockSize)) { \
    }                                                                   \
                                                                        \
    DEF_COMPRESSOR_COMMON(rname, name)

#define DEF_DECOMPRESSOR(rname, name)                            \
    class rname::TImpl: public TDecompressorBase<name, TImpl> {  \
    public:                                                      \
        inline TImpl(IInputStream* in)                           \
            : TDecompressorBase<name, TImpl>(in) {               \
        }                                                        \
    };                                                           \
                                                                 \
    rname::rname(IInputStream* slave)                            \
        : Impl_(new TImpl(slave)) {                              \
    }                                                            \
                                                                 \
    rname::~rname() {                                            \
    }                                                            \
                                                                 \
    size_t rname::DoRead(void* buf, size_t len) {                \
        return Impl_->Read(buf, len);                            \
    }

/*
 * MiniLzo
 */
class TMiniLzo {
    class TInit {
    public:
        inline TInit() {
            if (lzo_init() != LZO_E_OK) {
                ythrow yexception() << "can not init lzo engine";
            }
        }
    };

public:
    static const char signature[];

    inline TMiniLzo() {
        Singleton<TInit>();
    }

    inline ~TMiniLzo() {
    }

    static inline size_t Hint(size_t len) noexcept {
        // see SEARCH-2043 and, e.g. examples at
        // http://stackoverflow.com/questions/4235019/how-to-get-lzo-to-work-with-a-file-stream
        return len + (len / 16) + 64 + 3;
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};

const char TMiniLzo::signature[] = "YLZO";

template <size_t N>
class TFixedArray {
public:
    inline TFixedArray() noexcept {
        memset(WorkMem_, 0, sizeof(WorkMem_));
    }

protected:
    char WorkMem_[N];
};

class TMiniLzoCompressor: public TMiniLzo, public TFixedArray<LZO1X_MEM_COMPRESS + 1> {
public:
    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        lzo_uint out = 0;
        lzo1x_1_compress((const lzo_bytep)data, len, (lzo_bytep)ptr, &out, WorkMem_);

        return out;
    }
};

class TMiniLzoDecompressor: public TMiniLzo, public TFixedArray<LZO1X_MEM_DECOMPRESS + 1> {
public:
    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t /*max*/) {
        lzo_uint ret = 0;

        lzo1x_decompress((const lzo_bytep)data, len, (lzo_bytep)ptr, &ret, WorkMem_);

        return ret;
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }
};

DEF_COMPRESSOR(TLzoCompress, TMiniLzoCompressor)
DEF_DECOMPRESSOR(TLzoDecompress, TMiniLzoDecompressor)

/*
 * FastLZ
 */
class TFastLZ {
public:
    static const char signature[];

    static inline size_t Hint(size_t len) noexcept {
        return Max<size_t>((size_t)(len * 1.06), 100);
    }

    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        return fastlz_compress(data, len, ptr);
    }

    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t max) {
        return fastlz_decompress(data, len, ptr, max);
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};

const char TFastLZ::signature[] = "YLZF";

DEF_COMPRESSOR(TLzfCompress, TFastLZ)
DEF_DECOMPRESSOR(TLzfDecompress, TFastLZ)

/*
 * LZ4
 */
class TLZ4 {
public:
    static const char signature[];

    static inline size_t Hint(size_t len) noexcept {
        return Max<size_t>((size_t)(len * 1.06), 100);
    }

    inline size_t Compress(const char* data, size_t len, char* ptr, size_t dstMaxSize) {
        return LZ4_compress_default(data, ptr, len, dstMaxSize);
    }

    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t max) {
        int res = LZ4_decompress_safe(data, ptr, len, max);
        if (res < 0)
            ythrow TDecompressorError();
        return res;
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};

const char TLZ4::signature[] = "LZ.4";

DEF_COMPRESSOR(TLz4Compress, TLZ4)
DEF_DECOMPRESSOR(TLz4Decompress, TLZ4)

/*
 * Snappy
 */
class TSnappy {
public:
    static const char signature[];

    static inline size_t Hint(size_t len) noexcept {
        return Max<size_t>(snappy::MaxCompressedLength(len), 100);
    }

    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        size_t reslen = 0;
        snappy::RawCompress(data, len, ptr, &reslen);
        return reslen;
    }

    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t) {
        size_t srclen = 0;
        if (!snappy::GetUncompressedLength(data, len, &srclen) || !snappy::RawUncompress(data, len, ptr))
            ythrow TDecompressorError();
        return srclen;
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};

const char TSnappy::signature[] = "Snap";

DEF_COMPRESSOR(TSnappyCompress, TSnappy)
DEF_DECOMPRESSOR(TSnappyDecompress, TSnappy)

/*
 * QuickLZ
 */
class TQuickLZBase {
public:
    static const char signature[];

    static inline size_t Hint(size_t len) noexcept {
        return len + 500;
    }

    inline TQuickLZBase()
        : Table_(nullptr)
    {
    }

    inline void Init(unsigned ver, unsigned lev, unsigned mod, unsigned type) {
        Table_ = LzqTable(ver, lev, mod);

        if (!Table_) {
            ythrow yexception() << "unsupported lzq stream(" << ver << ", " << lev << ", " << mod << ")";
        }

        const size_t size = Table_->Setting(3) + Table_->Setting(type);

        Mem_.Reset(::operator new(size));
        memset(Mem_.Get(), 0, size);
    }

    inline bool SaveIncompressibleChunks() const noexcept {
        // we must save incompressible chunks "as is"
        // after compressor run in streaming mode
        return Table_->Setting(3);
    }

protected:
    const TQuickLZMethods* Table_;
    THolder<void> Mem_;
};

const char TQuickLZBase::signature[] = "YLZQ";

class TQuickLZCompress: public TQuickLZBase {
public:
    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        return Table_->Compress(data, ptr, len, (char*)Mem_.Get());
    }
};

class TQuickLZDecompress: public TQuickLZBase {
public:
    inline size_t Decompress(const char* data, size_t /*len*/, char* ptr, size_t /*max*/) {
        return Table_->Decompress(data, ptr, (char*)Mem_.Get());
    }

    inline void InitFromStream(IInputStream* in) {
        const ui8 ver = ::GLoad<ui8>(in);
        const ui8 lev = ::GLoad<ui8>(in);
        const ui8 mod = ::GLoad<ui8>(in);

        Init(ver, lev, mod, 2);
    }
};

class TLzqCompress::TImpl: public TCompressorBase<TQuickLZCompress, TImpl> {
public:
    inline TImpl(IOutputStream* out, ui16 blockSize, EVersion ver, unsigned level, EMode mode)
        : TCompressorBase<TQuickLZCompress, TImpl>(out, blockSize)
    {
        memset(AdditionalData(), 0, AdditionalDataLength());

        Init(ver, level, mode, 1);

        Save((ui8)ver);
        Save((ui8)level);
        Save((ui8)mode);
    }
};

TLzqCompress::TLzqCompress(IOutputStream* slave, ui16 blockSize, EVersion ver, unsigned level, EMode mode)
    : Impl_(new (TImpl::Hint(blockSize)) TImpl(slave, blockSize, ver, level, mode))
{
}

DEF_COMPRESSOR_COMMON(TLzqCompress, TQuickLZCompress)
DEF_DECOMPRESSOR(TLzqDecompress, TQuickLZDecompress)

namespace {
    template <class T>
    struct TInputHolder {
        static inline T Set(T t) noexcept {
            return t;
        }
    };

    template <class T>
    struct TInputHolder<TAutoPtr<T>> {
        inline T* Set(TAutoPtr<T> v) noexcept {
            V_ = v;

            return V_.Get();
        }

        TAutoPtr<T> V_;
    };

    // Decompressing input streams without signature verification
    template <class TInput, class TDecompressor>
    class TLzDecompressInput: public TInputHolder<TInput>, public IInputStream {
    public:
        inline TLzDecompressInput(TInput in)
            : Impl_(this->Set(in))
        {
        }

    private:
        size_t DoRead(void* buf, size_t len) override {
            return Impl_.Read(buf, len);
        }

    private:
        TDecompressorBaseImpl<TDecompressor> Impl_;
    };
}

template <class T>
static TAutoPtr<IInputStream> TryOpenLzDecompressorX(const TDecompressSignature& s, T input) {
    if (s.Check<TLZ4>())
        return new TLzDecompressInput<T, TLZ4>(input);

    if (s.Check<TSnappy>())
        return new TLzDecompressInput<T, TSnappy>(input);

    if (s.Check<TMiniLzo>())
        return new TLzDecompressInput<T, TMiniLzoDecompressor>(input);

    if (s.Check<TFastLZ>())
        return new TLzDecompressInput<T, TFastLZ>(input);

    if (s.Check<TQuickLZDecompress>())
        return new TLzDecompressInput<T, TQuickLZDecompress>(input);

    return nullptr;
}

template <class T>
static inline TAutoPtr<IInputStream> TryOpenLzDecompressorImpl(const TStringBuf& signature, T input) {
    if (signature.size() == SIGNATURE_SIZE) {
        TMemoryInput mem(signature.data(), signature.size());
        TDecompressSignature s(&mem);

        return TryOpenLzDecompressorX(s, input);
    }

    return nullptr;
}

template <class T>
static inline TAutoPtr<IInputStream> TryOpenLzDecompressorImpl(T input) {
    TDecompressSignature s(&*input);

    return TryOpenLzDecompressorX(s, input);
}

template <class T>
static inline TAutoPtr<IInputStream> OpenLzDecompressorImpl(T input) {
    TAutoPtr<IInputStream> ret = TryOpenLzDecompressorImpl(input);

    if (!ret) {
        ythrow TDecompressorError() << "Unknown compression format";
    }

    return ret;
}

TAutoPtr<IInputStream> OpenLzDecompressor(IInputStream* input) {
    return OpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenLzDecompressor(IInputStream* input) {
    return TryOpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenLzDecompressor(const TStringBuf& signature, IInputStream* input) {
    return TryOpenLzDecompressorImpl(signature, input);
}

TAutoPtr<IInputStream> OpenOwnedLzDecompressor(TAutoPtr<IInputStream> input) {
    return OpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenOwnedLzDecompressor(TAutoPtr<IInputStream> input) {
    return TryOpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenOwnedLzDecompressor(const TStringBuf& signature, TAutoPtr<IInputStream> input) {
    return TryOpenLzDecompressorImpl(signature, input);
}
