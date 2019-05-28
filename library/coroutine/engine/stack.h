#pragma once

#include <util/memory/alloc.h>
#include <util/system/align.h>
#include <util/generic/ptr.h>
#include <util/system/valgrind.h>
#include <util/system/context.h>
#include <util/system/info.h>

class TContStackAllocator {
public:
    class TStackType {
    public:
        TStackType() noexcept {
        }

        virtual ~TStackType() {
        }

        virtual void Release() noexcept {
            delete this;
        }

        virtual void* Data() noexcept = 0;
        virtual size_t Length() const noexcept = 0;

        void RegisterStackInValgrind() noexcept {
#if defined(WITH_VALGRIND)
            StackId_ = VALGRIND_STACK_REGISTER(Data(), (char*)Data() + Length());
#endif
        }

        void UnRegisterStackInValgrind() noexcept {
#if defined(WITH_VALGRIND)
            VALGRIND_STACK_DEREGISTER(StackId_);
#endif
        }

        void InsertStackOverflowCanary() noexcept {
            MagicNumberLocation() = MAGIC_NUMBER;
        }

        void VerifyNoStackOverflow() noexcept {
            if (Y_UNLIKELY(MagicNumberLocation() != MAGIC_NUMBER)) {
                FailStackOverflow();
            }
        }

    private:
        [[noreturn]] static void FailStackOverflow();

        ui32& MagicNumberLocation() noexcept {
            return *((ui32*)Data());
        }

#if defined(WITH_VALGRIND)
        size_t StackId_;
#endif
        // should not use constants like 0x11223344 or 0xCAFEBABE,
        // because of higher probablity of clash
        static const ui32 MAGIC_NUMBER = 0x9BC556F8u;
    };

    struct TRelease {
        static void Destroy(TStackType* s) noexcept {
            s->VerifyNoStackOverflow();
            s->UnRegisterStackInValgrind();
            s->Release();
        }
    };

    using TStackPtr = THolder<TStackType, TRelease>;

    TContStackAllocator() noexcept {
    }

    virtual ~TContStackAllocator() {
    }

    virtual TStackPtr Allocate() {
        TStackPtr ret = DoAllocate();

        // cheap operation, inserting even in release mode
        ret->InsertStackOverflowCanary();

        ret->RegisterStackInValgrind();

        return ret;
    }

private:
    virtual TStackType* DoAllocate() = 0;
};


class TGenericContStackAllocatorBase: public TContStackAllocator {
    using TBlock = IAllocator::TBlock;

    class TGenericStack: public TStackType {
    public:
        TGenericStack(TGenericContStackAllocatorBase* parent, const TBlock& block) noexcept
            : Parent_(parent)
            , Block_(block)
        {}

        ~TGenericStack() override {
        }

        void Release() noexcept override {
            TGenericContStackAllocatorBase* parent = Parent_;
            const TBlock block(Block_);
            this->~TGenericStack();
            parent->Alloc_->Release(block);
        }

        void* Data() noexcept override {
            return this + 1;
        }

        size_t Length() const noexcept override {
            return Block_.Len - sizeof(*this);
        }

    private:
        TGenericContStackAllocatorBase* Parent_;
        const TBlock Block_;
    };

public:
    TGenericContStackAllocatorBase(IAllocator* alloc, size_t len) noexcept
        : Alloc_(alloc)
        , Len_(len)
    {}

    ~TGenericContStackAllocatorBase() override {
    }

    TStackType* DoAllocate() override {
        TBlock block = Alloc_->Allocate(Len_ + sizeof(TGenericStack));
        return new (block.Data) TGenericStack(this, block);
    }

private:
    IAllocator* Alloc_;
    const size_t Len_;
};


class TProtectedContStackAllocator: public TContStackAllocator {
    static const size_t PageSize_ = 4096;

    static void Protect(void* ptr, size_t len) noexcept;
    static void UnProtect(void* ptr, size_t len) noexcept;

    class TProtectedStack: public TStackType {
    public:
        TProtectedStack(TStackType* slave) noexcept
            : Substack_(slave)
        {
            Y_ASSERT(Length() % PageSize_ == 0);

            Protect((char*)AlignedData(), PageSize_);
            Protect((char*)Data() + Length(), PageSize_);
        }

        ~TProtectedStack() override {
            UnProtect((char*)AlignedData(), PageSize_);
            UnProtect((char*)Data() + Length(), PageSize_);

            Substack_->Release();
        }

        void Release() noexcept override {
            delete this;
        }

        void* AlignedData() noexcept {
            return AlignUp(Substack_->Data(), PageSize_);
        }

        void* Data() noexcept override {
            return (char*)AlignedData() + PageSize_;
        }

        size_t Length() const noexcept override {
            return Substack_->Length() - 3 * PageSize_;
        }

    private:
        TStackType* Substack_;
    };

public:
    TProtectedContStackAllocator(IAllocator* alloc, size_t len) noexcept
        : Alloc_(alloc, AlignUp(len, PageSize_) + 3 * PageSize_)
    {
        if (NSystemInfo::GetPageSize() > PageSize_) {
            Enabled_ = false;
        }
    }

    ~TProtectedContStackAllocator() override {
    }

    TStackType* DoAllocate() override {
        auto* subStack = Alloc_.DoAllocate();
        return Enabled_ ? new TProtectedStack(subStack) : subStack;
    }

private:
    TGenericContStackAllocatorBase Alloc_;
    bool Enabled_ = true;
};

#if defined(NDEBUG) && !defined(_san_enabled_)
using TGenericContStackAllocator = TGenericContStackAllocatorBase;
#else
using TGenericContStackAllocator = TProtectedContStackAllocator;
#endif

class THeapStackAllocator: public TGenericContStackAllocator {
public:
    THeapStackAllocator(size_t len)
        : TGenericContStackAllocator(TDefaultAllocator::Instance(), len)
    {
    }
};

using TDefaultStackAllocator = THeapStackAllocator;
