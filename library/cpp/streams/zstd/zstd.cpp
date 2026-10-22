#include "zstd.h"

#include <util/generic/buffer.h>
#include <util/generic/yexception.h>

#define ZSTD_STATIC_LINKING_ONLY
#include <contrib/libs/zstd/include/zstd.h>

namespace {
    inline void CheckError(const char* op, size_t code) {
        if (::ZSTD_isError(code)) {
            ythrow yexception() << op << TStringBuf(" zstd error: ") << ::ZSTD_getErrorName(code);
        }
    }

    struct DestroyZCStream {
        static void Destroy(::ZSTD_CStream* p) noexcept {
            ::ZSTD_freeCStream(p);
        }
    };

    struct DestroyZDStream {
        static void Destroy(::ZSTD_DStream* p) noexcept {
            ::ZSTD_freeDStream(p);
        }
    };
}

class TZstdCompress::TImpl {
public:
    TImpl(IOutputStream* slave, int quality)
        : Slave_(slave)
        , ZCtx_(::ZSTD_createCStream())
        , Buffer_(::ZSTD_CStreamOutSize())  // do reserve
    {
        Y_ENSURE(nullptr != ZCtx_.Get(), "Failed to allocate ZSTD_CStream");
        Y_ENSURE(0 != Buffer_.Capacity(), "ZSTD_CStreamOutSize was too small");
        CheckError("init", ZSTD_initCStream(ZCtx_.Get(), quality));
    }

    void Write(const void* buffer, size_t size) {
        ::ZSTD_inBuffer zIn{buffer, size, 0};
        auto zOut = OutBuffer();

        while (0 != zIn.size) {
            CheckError("compress", ::ZSTD_compressStream(ZCtx_.Get(), &zOut, &zIn));
            DoWrite(zOut);
            // forget about the data we already compressed
            zIn.src = static_cast<const unsigned char*>(zIn.src) + zIn.pos;
            zIn.size -= zIn.pos;
            zIn.pos = 0;
        }
    }

    void Flush() {
        auto zOut = OutBuffer();
        CheckError("flush", ::ZSTD_flushStream(ZCtx_.Get(), &zOut));
        DoWrite(zOut);
    }

    void Finish() {
        auto zOut = OutBuffer();
        size_t returnCode;
        do {
            returnCode = ::ZSTD_endStream(ZCtx_.Get(), &zOut);
            CheckError("finish", returnCode);
            DoWrite(zOut);
        } while (0 != returnCode);  // zero means there is no more bytes to flush
    }

private:
    ::ZSTD_outBuffer OutBuffer() {
        return {Buffer_.Data(), Buffer_.Capacity(), 0};
    }

    void DoWrite(::ZSTD_outBuffer& buffer) {
        Slave_->Write(buffer.dst, buffer.pos);
        buffer.pos = 0;
    }
private:
    IOutputStream* Slave_;
    THolder<::ZSTD_CStream, DestroyZCStream> ZCtx_;
    TBuffer Buffer_;
};

TZstdCompress::TZstdCompress(IOutputStream* slave, int quality)
    : Impl_(new TImpl(slave, quality)) {
}

TZstdCompress::~TZstdCompress() {
    try {
        Finish();
    } catch (...) {
    }
}

void TZstdCompress::DoWrite(const void* buffer, size_t size) {
    Y_ENSURE(Impl_, "Cannot use stream after finish.");
    Impl_->Write(buffer, size);
}

void TZstdCompress::DoFlush() {
    Y_ENSURE(Impl_, "Cannot use stream after finish.");
    Impl_->Flush();
}

void TZstdCompress::DoFinish() {
    // Finish should be idempotent
    if (Impl_) {
        auto impl = std::move(Impl_);
        impl->Finish();
    }
}

////////////////////////////////////////////////////////////////////////////////

class TZstdDecompress::TImpl {
public:
    TImpl(IInputStream* slave, size_t bufferSize)
        : Slave_(slave)
        , ZCtx_(::ZSTD_createDStream())
        , Buffer_(bufferSize)  // do reserve
        , Offset_(0)
    {
        Y_ENSURE(nullptr != ZCtx_.Get(), "Failed to allocate ZSTD_DStream");
        Y_ENSURE(0 != Buffer_.Capacity(), "Buffer size was too small");
    }

    size_t Read(void* buffer, size_t size) {
        Y_ASSERT(size > 0);

        ::ZSTD_outBuffer zOut{buffer, size, 0};
        ::ZSTD_inBuffer zIn{Buffer_.Data(), Buffer_.Size(), Offset_};

        size_t returnCode = 0;
        while (zOut.pos != zOut.size) {
            if (zIn.pos == zIn.size) {
                zIn.size = Slave_->Read(Buffer_.Data(), Buffer_.Capacity());
                Buffer_.Resize(zIn.size);
                zIn.pos = Offset_ = 0;
                if (0 == zIn.size) {
                    // end of stream, need to check that there is no uncompleted blocks
                    Y_ENSURE(0 == returnCode, "Incomplete block");
                    break;
                }
            }
            returnCode = ::ZSTD_decompressStream(ZCtx_.Get(), &zOut, &zIn);
            CheckError("decompress", returnCode);
            if (0 == returnCode) {
                // The frame is over, prepare to (maybe) start a new frame
                ZSTD_initDStream(ZCtx_.Get());
            }
        }
        Offset_ = zIn.pos;
        return zOut.pos;
    }

private:
    IInputStream* Slave_;
    THolder<::ZSTD_DStream, DestroyZDStream> ZCtx_;
    TBuffer Buffer_;
    size_t  Offset_;
};

TZstdDecompress::TZstdDecompress(IInputStream* slave, size_t bufferSize)
    : Impl_(new TImpl(slave, bufferSize)) {
}

TZstdDecompress::~TZstdDecompress() = default;

size_t TZstdDecompress::DoRead(void* buffer, size_t size) {
    return Impl_->Read(buffer, size);
}
