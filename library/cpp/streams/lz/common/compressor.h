#pragma once

#include <util/system/yassert.h>
#include <util/system/byteorder.h>
#include <util/memory/addstorage.h>
#include <util/generic/buffer.h>
#include <util/generic/utility.h>
#include <util/generic/singleton.h>
#include <util/stream/mem.h>

#include "error.h"

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
            ythrow yexception() << TStringBuf("incorrect stream version: ") << v;
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
        , Tmp_(2 * OutBufSize_)
        , In_(Tmp_.Data())
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
        if (len > Tmp_.Capacity()) {
            ythrow TDecompressorError() << "invalid len inside block header";
        }
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
    TBuffer Tmp_;
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
