#include "lzma.h"

#include <util/stream/mem.h>
#include <util/system/context.h>
#include <util/generic/cast.h>
#include <util/memory/addstorage.h>
#include <util/generic/ptr.h>
#include <util/generic/intrlist.h>
#include <util/generic/scope.h>

extern "C" {
#include <contrib/libs/lzmasdk/LzmaEnc.h>
#include <contrib/libs/lzmasdk/LzmaDec.h>
}

namespace {
    class TMemoryGc {
        class TAllocation: public TIntrusiveListItem<TAllocation>, public TAdditionalStorage<TAllocation> {
        };

    public:
        inline void* Allocate(size_t len) {
            if (len > 1024 * 1024 * 1024) {
                return nullptr;
            }

            TAllocation* ret = new (len) TAllocation;

            Allocs_.PushBack(ret);

            return ret->AdditionalData();
        }

        inline void Deallocate(void* ptr) noexcept {
            if (ptr) {
                delete TAllocation::ObjectFromData(ptr);
            }
        }

    private:
        TIntrusiveListWithAutoDelete<TAllocation, TDelete> Allocs_;
    };

    template <class T>
    class TInverseFilter {
        class TTrampoLine: public ITrampoLine {
        public:
            inline TTrampoLine(TInverseFilter* parent)
                : Parent_(parent)
            {
            }

            void DoRun() override {
                Parent_->RunFilter();
            }

        private:
            TInverseFilter* Parent_;
        };

        class TInput: public IInputStream {
        public:
            inline TInput(TInverseFilter* parent)
                : Parent_(parent)
            {
            }

            ~TInput() override {
            }

            size_t DoRead(void* ptr, size_t len) override {
                return Parent_->ReadImpl(ptr, len);
            }

        private:
            TInverseFilter* Parent_;
        };

        class TOutput: public IOutputStream {
        public:
            inline TOutput(TInverseFilter* parent)
                : Parent_(parent)
            {
            }

            ~TOutput() override {
            }

            void DoWrite(const void* ptr, size_t len) override {
                Parent_->WriteImpl(ptr, len);
            }

        private:
            TInverseFilter* Parent_;
        };

    public:
        inline TInverseFilter(IOutputStream* slave, T* filter)
            : Slave_(slave)
            , Filter_(filter)
            , TrampoLine_(this)
            , FilterCtx_(FilterClosure())
            , Finished_(false)
            , In_(nullptr, 0)
        {
        }

        virtual ~TInverseFilter() {
            if (!UncaughtException()) {
                try {
                    Finish();
                } catch (...) {
                }
            } else {
                //rely on gc
            }
        }

        inline void Write(const void* ptr, size_t len) {
            In_.Reset(ptr, len);

            Y_DEFER {
                In_.Reset(0, 0);
            };

            while (In_.Avail()) {
                SwitchTo();
            }
        }

        inline void Finish() {
            if (!Finished_) {
                Finished_ = true;
                SwitchTo();
            }
        }

    private:
        inline void RunFilter() {
            try {
                TInput in(this);
                TOutput out(this);

                (*Filter_)(&in, &out);
            } catch (...) {
                Err_ = std::current_exception();
            }

            SwitchFrom();
        }

        inline TContClosure FilterClosure() {
            return {&TrampoLine_, TArrayRef(Stack_, sizeof(Stack_))};
        }

        inline size_t ReadImpl(void* ptr, size_t len) {
            while (!Finished_) {
                const size_t ret = In_.Read(ptr, len);

                if (ret) {
                    return ret;
                }

                SwitchFrom();
            }

            return 0;
        }

        inline void WriteImpl(const void* ptr, size_t len) {
            Y_ASSERT(!Out_.Avail());

            Out_.Reset(ptr, len);

            while (Out_.Avail()) {
                SwitchFrom();
            }
        }

        inline bool FlushImpl() {
            if (Out_.Avail()) {
                TransferData(&Out_, Slave_);
                Out_.Reset(nullptr, 0);

                return true;
            }

            return false;
        }

        inline void SwitchTo() {
            do {
                CurrentCtx_.SwitchTo(&FilterCtx_);

                if (Err_) {
                    Finished_ = true;

                    std::rethrow_exception(Err_);
                }
            } while (FlushImpl());
        }

        inline void SwitchFrom() {
            FilterCtx_.SwitchTo(&CurrentCtx_);
        }

    private:
        IOutputStream* Slave_;
        T* Filter_;
        TTrampoLine TrampoLine_;
        char Stack_[16 * 1024];
        TContMachineContext FilterCtx_;
        TContMachineContext CurrentCtx_;
        bool Finished_;
        TMemoryInput In_;
        TMemoryInput Out_;
        std::exception_ptr Err_;
    };

    class TLzma {
    public:
        class TLzmaInput: public ISeqInStream {
        public:
            inline TLzmaInput(IInputStream* slave)
                : Slave_(slave)
            {
                Read = ReadFunc;
            }

        private:
            static inline SRes ReadFunc(const ISeqInStream* p, void* ptr, size_t* len) {
                *len = const_cast<TLzmaInput*>(static_cast<const TLzmaInput*>(p))->Slave_->Read(ptr, *len);

                return SZ_OK;
            }

        private:
            IInputStream* Slave_;
        };

        class TLzmaOutput: public ISeqOutStream {
        public:
            inline TLzmaOutput(IOutputStream* slave)
                : Slave_(slave)
            {
                Write = WriteFunc;
            }

        private:
            static inline size_t WriteFunc(const ISeqOutStream* p, const void* ptr, size_t len) {
                const_cast<TLzmaOutput*>(static_cast<const TLzmaOutput*>(p))->Slave_->Write(ptr, len);

                return len;
            }

        private:
            IOutputStream* Slave_;
        };

        class TAlloc: public ISzAlloc {
        public:
            inline TAlloc() {
                Alloc = AllocFunc;
                Free = FreeFunc;
            }

        private:
            static void* AllocFunc(const ISzAlloc* t, size_t len) {
                return static_cast<TAlloc*>(((ISzAlloc*)t))->Gc_.Allocate(len);
            }

            static void FreeFunc(const ISzAlloc* t, void* p) {
                static_cast<TAlloc*>(((ISzAlloc*)t))->Gc_.Deallocate(p);
            }

        private:
            TMemoryGc Gc_;
        };

        inline ISzAlloc* Alloc() noexcept {
            return &Alloc_;
        }

        static inline void Check(SRes r) {
            if (r != SZ_OK) {
                ythrow yexception() << "lzma error(" << r << ")";
            }
        }

    private:
        TAlloc Alloc_;
    };

    class TLzmaCompressBase: public TLzma {
    public:
        inline TLzmaCompressBase(size_t level)
            : H_(LzmaEnc_Create(Alloc()))
        {
            if (!H_) {
                ythrow yexception() << "can not init lzma engine";
            }

            LzmaEncProps_Init(&Props_);

            Props_.level = level;
            Props_.dictSize = 0;
            Props_.lc = -1;
            Props_.lp = -1;
            Props_.pb = -1;
            Props_.fb = -1;
            Props_.numThreads = -1;
            Props_.writeEndMark = 1;

            Check(LzmaEnc_SetProps(H_, &Props_));
            size_t bufLen = sizeof(PropsBuf_);
            Zero(PropsBuf_);
            Check(LzmaEnc_WriteProperties(H_, PropsBuf_, &bufLen));
        }

        inline ~TLzmaCompressBase() {
            LzmaEnc_Destroy(H_, Alloc(), Alloc());
        }

        inline void operator()(IInputStream* in, IOutputStream* out) {
            TLzmaInput input(in);
            TLzmaOutput output(out);

            out->Write(PropsBuf_, sizeof(PropsBuf_));

            Check(LzmaEnc_Encode(H_, &output, &input, nullptr, Alloc(), Alloc()));
        }

    private:
        CLzmaEncHandle H_;
        CLzmaEncProps Props_;
        Byte PropsBuf_[LZMA_PROPS_SIZE];
    };
}

class TLzmaCompress::TImpl: public TLzmaCompressBase, public TInverseFilter<TLzmaCompressBase> {
public:
    inline TImpl(IOutputStream* slave, size_t level)
        : TLzmaCompressBase(level)
        , TInverseFilter<TLzmaCompressBase>(slave, this)
    {
    }
};

class TLzmaDecompress::TImpl: public TLzma {
public:
    inline TImpl()
        : InBegin_(nullptr)
        , InEnd_(nullptr)
    {
        LzmaDec_Construct(&H_);
    }
    inline virtual ~TImpl() {
        LzmaDec_Free(&H_, Alloc());
    }

    inline size_t Read(void* ptr, size_t len) {
        Byte* pos = (Byte*)ptr;
        Byte* end = pos + len;

    retry:
        size_t availLen = InEnd_ - InBegin_;
        size_t bufLen = end - pos;
        ELzmaStatus status;

        Check(LzmaDec_DecodeToBuf(&H_, pos, &bufLen, (Byte*)InBegin_, &availLen, LZMA_FINISH_ANY, &status));

        InBegin_ += availLen;
        pos += bufLen;

        if (status == LZMA_STATUS_NEEDS_MORE_INPUT) {
            Y_ASSERT(InEnd_ == InBegin_);
            if (!Fill()) {
                ythrow yexception() << "incomplete lzma stream";
            }

            goto retry;
        }

        return pos - (Byte*)ptr;
    }

private:
    virtual bool Fill() = 0;

protected:
    CLzmaDec H_;
    char* InBegin_;
    char* InEnd_;
};

class TLzmaDecompress::TImplStream: public TImpl {
public:
    inline TImplStream(IInputStream* slave)
        : Slave_(slave)
    {
        Byte buf[LZMA_PROPS_SIZE];

        if (Slave_->Load(buf, sizeof(buf)) != sizeof(buf))
            ythrow yexception() << "can't read lzma header";

        Check(LzmaDec_Allocate(&H_, buf, sizeof(buf), Alloc()));
        LzmaDec_Init(&H_);
    }

private:
    bool Fill() override {
        size_t size = Slave_->Read(In_, sizeof(In_));
        InBegin_ = In_;
        InEnd_ = In_ + size;

        return size;
    }

private:
    IInputStream* Slave_;
    char In_[4096];
};

class TLzmaDecompress::TImplZeroCopy: public TLzmaDecompress::TImpl {
public:
    inline TImplZeroCopy(IZeroCopyInput* in)
        : Input_(in)
    {
        if (!Fill())
            ythrow yexception() << "can't read lzma header";

        char buf[LZMA_PROPS_SIZE];
        char* header;
        if (InEnd_ - InBegin_ >= LZMA_PROPS_SIZE) {
            header = InBegin_;
            InBegin_ += LZMA_PROPS_SIZE;
        } else {
            //bad luck, first part is less than header
            //try to copy header part by part to the local buffer
            const char* end = buf + sizeof(buf);
            char* pos = buf;
            while (1) {
                size_t left = end - pos;
                size_t avail = InEnd_ - InBegin_;
                if (left < avail) {
                    memcpy(pos, InBegin_, left);
                    InBegin_ += left;
                    break;
                } else {
                    memcpy(pos, InBegin_, avail);
                    pos += avail;
                    if (!Fill()) {
                        ythrow yexception() << "can't read lzma header";
                    }
                }
            }
            header = buf;
        }

        Check(LzmaDec_Allocate(&H_, (Byte*)header, LZMA_PROPS_SIZE, Alloc()));

        LzmaDec_Init(&H_);
    }

private:
    bool Fill() override {
        size_t size = Input_->Next(&InBegin_);

        if (size) {
            InEnd_ = InBegin_ + size;

            return true;
        }

        return false;
    }

    IZeroCopyInput* Input_;
};

TLzmaCompress::TLzmaCompress(IOutputStream* slave, size_t level)
    : Impl_(new TImpl(slave, level))
{
}

TLzmaCompress::~TLzmaCompress() {
}

void TLzmaCompress::DoWrite(const void* buf, size_t len) {
    if (!Impl_) {
        ythrow yexception() << "can not write to finished lzma stream";
    }

    Impl_->Write(buf, len);
}

void TLzmaCompress::DoFinish() {
    THolder<TImpl> impl(Impl_.Release());

    if (impl) {
        impl->Finish();
    }
}

TLzmaDecompress::TLzmaDecompress(IInputStream* slave)
    : Impl_(new TImplStream(slave))
{
}

TLzmaDecompress::TLzmaDecompress(IZeroCopyInput* input)
    : Impl_(new TImplZeroCopy(input))
{
}

TLzmaDecompress::~TLzmaDecompress() {
}

size_t TLzmaDecompress::DoRead(void* buf, size_t len) {
    return Impl_->Read(buf, len);
}
