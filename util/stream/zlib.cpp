#include "zlib.h"

#include <util/memory/addstorage.h>
#include <util/generic/scope.h>
#include <util/generic/utility.h>

#include <zlib.h>

#include <cstring>

namespace {
    static const int opts[] = {
        // Auto
        15 + 32,
        // ZLib
        15 + 0,
        // GZip
        15 + 16,
        // Raw
        -15};

    class TZLibCommon {
    public:
        inline TZLibCommon() noexcept {
            memset(Z(), 0, sizeof(*Z()));
        }

        inline ~TZLibCommon() = default;

        inline const char* GetErrMsg() const noexcept {
            return Z()->msg != nullptr ? Z()->msg : "unknown error";
        }

        inline z_stream* Z() const noexcept {
            return (z_stream*)(&Z_);
        }

    private:
        z_stream Z_;
    };

    static inline ui32 MaxPortion(size_t s) noexcept {
        return (ui32)Min<size_t>(Max<ui32>(), s);
    }

    struct TChunkedZeroCopyInput {
        inline TChunkedZeroCopyInput(IZeroCopyInput* in)
            : In(in)
            , Buf(nullptr)
            , Len(0)
        {
        }

        template <class P, class T>
        inline bool Next(P** buf, T* len) {
            if (!Len) {
                Len = In->Next(&Buf);
                if (!Len) {
                    return false;
                }
            }

            const T toread = (T)Min((size_t)Max<T>(), Len);

            *len = toread;
            *buf = (P*)Buf;

            Buf += toread;
            Len -= toread;

            return true;
        }

        IZeroCopyInput* In;
        const char* Buf;
        size_t Len;
    };
} // namespace

class TZLibDecompress::TImpl: private TZLibCommon, public TChunkedZeroCopyInput {
public:
    inline TImpl(IZeroCopyInput* in, ZLib::StreamType type, TStringBuf dict)
        : TChunkedZeroCopyInput(in)
        , Dict(dict)
    {
        if (inflateInit2(Z(), opts[type]) != Z_OK) {
            ythrow TZLibDecompressorError() << "can not init inflate engine";
        }

        if (dict.size() && type == ZLib::Raw) {
            SetDict();
        }
    }

    virtual ~TImpl() {
        inflateEnd(Z());
    }

    void SetAllowMultipleStreams(bool allowMultipleStreams) {
        AllowMultipleStreams_ = allowMultipleStreams;
    }

    inline size_t Read(void* buf, size_t size) {
        Z()->next_out = (unsigned char*)buf;
        Z()->avail_out = size;

        while (true) {
            if (Z()->avail_in == 0) {
                if (!FillInputBuffer()) {
                    return 0;
                }
            }

            switch (inflate(Z(), Z_SYNC_FLUSH)) {
                case Z_NEED_DICT: {
                    SetDict();
                    continue;
                }

                case Z_STREAM_END: {
                    if (AllowMultipleStreams_) {
                        if (inflateReset(Z()) != Z_OK) {
                            ythrow TZLibDecompressorError() << "inflate reset error(" << GetErrMsg() << ")";
                        }
                    } else {
                        return size - Z()->avail_out;
                    }

                    [[fallthrough]];
                }

                case Z_OK: {
                    const size_t processed = size - Z()->avail_out;

                    if (processed) {
                        return processed;
                    }

                    break;
                }

                default:
                    ythrow TZLibDecompressorError() << "inflate error(" << GetErrMsg() << ")";
            }
        }
    }

private:
    inline bool FillInputBuffer() {
        return Next(&Z()->next_in, &Z()->avail_in);
    }

    void SetDict() {
        if (inflateSetDictionary(Z(), (const Bytef*)Dict.data(), Dict.size()) != Z_OK) {
            ythrow TZLibCompressorError() << "can not set inflate dictionary";
        }
    }

    bool AllowMultipleStreams_ = true;
    TStringBuf Dict;
};

namespace {
    class TDecompressStream: public IZeroCopyInput, public TZLibDecompress::TImpl, public TAdditionalStorage<TDecompressStream> {
    public:
        inline TDecompressStream(IInputStream* input, ZLib::StreamType type, TStringBuf dict)
            : TZLibDecompress::TImpl(this, type, dict)
            , Stream_(input)
        {
        }

        ~TDecompressStream() override = default;

    private:
        size_t DoNext(const void** ptr, size_t len) override {
            void* buf = AdditionalData();

            *ptr = buf;
            return Stream_->Read(buf, Min(len, AdditionalDataLength()));
        }

    private:
        IInputStream* Stream_;
    };

    using TZeroCopyDecompress = TZLibDecompress::TImpl;
} // namespace

class TZLibCompress::TImpl: public TAdditionalStorage<TImpl>, private TZLibCommon {
    static inline ZLib::StreamType Type(ZLib::StreamType type) {
        if (type == ZLib::Auto) {
            return ZLib::ZLib;
        }

        if (type >= ZLib::Invalid) {
            ythrow TZLibError() << "invalid compression type: " << static_cast<unsigned long>(type);
        }

        return type;
    }

public:
    inline TImpl(const TParams& p)
        : Stream_(p.Out)
    {
        if (deflateInit2(Z(), Min<size_t>(9, p.CompressionLevel), Z_DEFLATED, opts[Type(p.Type)], 8, Z_DEFAULT_STRATEGY)) {
            ythrow TZLibCompressorError() << "can not init inflate engine";
        }

        // Create exactly the same files on all platforms by fixing OS field in the header.
        if (p.Type == ZLib::GZip) {
            GZHeader_ = MakeHolder<gz_header>();
            GZHeader_->os = 3; // UNIX
            deflateSetHeader(Z(), GZHeader_.Get());
        }

        if (p.Dict.size()) {
            if (deflateSetDictionary(Z(), (const Bytef*)p.Dict.data(), p.Dict.size())) {
                ythrow TZLibCompressorError() << "can not set deflate dictionary";
            }
        }

        Z()->next_out = TmpBuf();
        Z()->avail_out = TmpBufLen();
    }

    inline ~TImpl() {
        deflateEnd(Z());
    }

    inline void Write(const void* buf, size_t size) {
        const Bytef* b = (const Bytef*)buf;
        const Bytef* e = b + size;

        Y_DEFER {
            Z()->next_in = nullptr;
            Z()->avail_in = 0;
        };
        do {
            b = WritePart(b, e);
        } while (b < e);
    }

    inline const Bytef* WritePart(const Bytef* b, const Bytef* e) {
        Z()->next_in = const_cast<Bytef*>(b);
        Z()->avail_in = MaxPortion(e - b);

        while (Z()->avail_in) {
            const int ret = deflate(Z(), Z_NO_FLUSH);

            switch (ret) {
                case Z_OK:
                    continue;

                case Z_BUF_ERROR:
                    FlushBuffer();

                    break;

                default:
                    ythrow TZLibCompressorError() << "deflate error(" << GetErrMsg() << ")";
            }
        }

        return Z()->next_in;
    }

    inline void Flush() {
        int ret = deflate(Z(), Z_SYNC_FLUSH);

        while ((ret == Z_OK || ret == Z_BUF_ERROR) && !Z()->avail_out) {
            FlushBuffer();
            ret = deflate(Z(), Z_SYNC_FLUSH);
        }

        if (ret != Z_OK && ret != Z_BUF_ERROR) {
            ythrow TZLibCompressorError() << "deflate flush error(" << GetErrMsg() << ")";
        }

        if (Z()->avail_out < TmpBufLen()) {
            FlushBuffer();
        }
    }

    inline void FlushBuffer() {
        Stream_->Write(TmpBuf(), TmpBufLen() - Z()->avail_out);
        Z()->next_out = TmpBuf();
        Z()->avail_out = TmpBufLen();
    }

    inline void Finish() {
        int ret = deflate(Z(), Z_FINISH);

        while (ret == Z_OK || ret == Z_BUF_ERROR) {
            FlushBuffer();
            ret = deflate(Z(), Z_FINISH);
        }

        if (ret == Z_STREAM_END) {
            Stream_->Write(TmpBuf(), TmpBufLen() - Z()->avail_out);
        } else {
            ythrow TZLibCompressorError() << "deflate finish error(" << GetErrMsg() << ")";
        }
    }

private:
    inline unsigned char* TmpBuf() noexcept {
        return (unsigned char*)AdditionalData();
    }

    inline size_t TmpBufLen() const noexcept {
        return AdditionalDataLength();
    }

private:
    IOutputStream* Stream_;
    THolder<gz_header> GZHeader_;
};

TZLibDecompress::TZLibDecompress(IZeroCopyInput* input, ZLib::StreamType type, TStringBuf dict)
    : Impl_(new TZeroCopyDecompress(input, type, dict))
{
}

TZLibDecompress::TZLibDecompress(IInputStream* input, ZLib::StreamType type, size_t buflen, TStringBuf dict)
    : Impl_(new (buflen) TDecompressStream(input, type, dict))
{
}

void TZLibDecompress::SetAllowMultipleStreams(bool allowMultipleStreams) {
    Impl_->SetAllowMultipleStreams(allowMultipleStreams);
}

TZLibDecompress::~TZLibDecompress() = default;

size_t TZLibDecompress::DoRead(void* buf, size_t size) {
    return Impl_->Read(buf, MaxPortion(size));
}

void TZLibCompress::Init(const TParams& params) {
    Y_ENSURE(params.BufLen >= 16, "ZLib buffer too small");
    Impl_.Reset(new (params.BufLen) TImpl(params));
}

void TZLibCompress::TDestruct::Destroy(TImpl* impl) {
    delete impl;
}

TZLibCompress::~TZLibCompress() {
    try {
        Finish();
    } catch (...) {
        // ¯\_(ツ)_/¯
    }
}

void TZLibCompress::DoWrite(const void* buf, size_t size) {
    if (!Impl_) {
        ythrow TZLibCompressorError() << "can not write to finished zlib stream";
    }

    Impl_->Write(buf, size);
}

void TZLibCompress::DoFlush() {
    if (Impl_) {
        Impl_->Flush();
    }
}

void TZLibCompress::DoFinish() {
    THolder<TImpl> impl(Impl_.Release());

    if (impl) {
        impl->Finish();
    }
}

TBufferedZLibDecompress::~TBufferedZLibDecompress() = default;
