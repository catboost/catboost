#pragma once

#include <util/memory/alloc.h>
#include <util/system/align.h>
#include <util/generic/ptr.h>
#include <util/system/valgrind.h>
#include <util/system/context.h>

class TContStackAllocator {
public:
    class TStackType {
    public:
        inline TStackType() noexcept {
        }

        virtual ~TStackType() {
        }

        virtual void Release() noexcept {
            delete this;
        }

        virtual void* Data() noexcept = 0;
        virtual size_t Length() const noexcept = 0;

        inline void RegisterStackInValgrind() noexcept {
#if defined(WITH_VALGRIND)
            StackId_ = VALGRIND_STACK_REGISTER(Data(), (char*)Data() + Length());
#endif
        }

        inline void UnRegisterStackInValgrind() noexcept {
#if defined(WITH_VALGRIND)
            VALGRIND_STACK_DEREGISTER(StackId_);
#endif
        }

        inline void InsertStackOverflowProtector() noexcept {
            MagicNumberLocation() = MAGIC_NUMBER;
        }

        inline void VerifyNoStackOverflow() noexcept {
            if (Y_UNLIKELY(MagicNumberLocation() != MAGIC_NUMBER)) {
                FailStackOverflow();
            }
        }

    private:
        Y_NO_RETURN static void FailStackOverflow();

        inline ui32& MagicNumberLocation() noexcept {
#if STACK_GROW_DOWN == 1
            return *((ui32*)Data());
#elif STACK_GROW_DOWN == 0
            return *(((ui32*)(((char*)Data()) + Length())) - 1);
#else
#error "unknown"
#endif
        }

#if defined(WITH_VALGRIND)
        size_t StackId_;
#endif
        // should not use constants like 0x11223344 or 0xCAFEBABE,
        // because of higher probablity of clash
        static const ui32 MAGIC_NUMBER = 0x9BC556F8u;
    };

    struct TRelease {
        static inline void Destroy(TStackType* s) noexcept {
            s->VerifyNoStackOverflow();

            s->UnRegisterStackInValgrind();

            s->Release();
        }
    };

    typedef TAutoPtr<TStackType, TRelease> TStackPtr;

    inline TContStackAllocator() noexcept {
    }

    virtual ~TContStackAllocator() {
    }

    virtual TStackPtr Allocate() {
        TStackPtr ret(DoAllocate());

        // cheap operation, inserting even in release mode
        ret->InsertStackOverflowProtector();

        ret->RegisterStackInValgrind();

        return ret;
    }

private:
    virtual TStackType* DoAllocate() = 0;
};

class TGenericContStackAllocatorBase: public TContStackAllocator {
    typedef IAllocator::TBlock TBlock;

    class TGenericStack: public TStackType {
    public:
        inline TGenericStack(TGenericContStackAllocatorBase* parent, const TBlock& block) noexcept
            : Parent_(parent)
            , Block_(block)
        {
        }

        ~TGenericStack() override {
        }

        void Release() noexcept override {
            TGenericContStackAllocatorBase* parent(Parent_);
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
    inline TGenericContStackAllocatorBase(IAllocator* alloc, size_t len) noexcept
        : Alloc_(alloc)
        , Len_(len)
    {
    }

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
    static inline size_t PageSize() noexcept {
        return 4096;
    }

    static void Protect(void* ptr, size_t len) noexcept;
    static void UnProtect(void* ptr, size_t len) noexcept;

    class TProtectedStack: public TStackType {
    public:
        inline TProtectedStack(TStackType* slave) noexcept
            : Slave_(slave)
        {
            Y_ASSERT(Length() % PageSize() == 0);

            Protect((char*)AlignedData(), PageSize());
            Protect((char*)Data() + Length(), PageSize());
        }

        ~TProtectedStack() override {
            UnProtect((char*)AlignedData(), PageSize());
            UnProtect((char*)Data() + Length(), PageSize());

            Slave_->Release();
        }

        void Release() noexcept override {
            delete this;
        }

        inline void* AlignedData() noexcept {
            return AlignUp(Slave_->Data(), PageSize());
        }

        void* Data() noexcept override {
            return (char*)AlignedData() + PageSize();
        }

        size_t Length() const noexcept override {
            return Slave_->Length() - 3 * PageSize();
        }

    private:
        TStackType* Slave_;
    };

public:
    inline TProtectedContStackAllocator(IAllocator* alloc, size_t len) noexcept
        : Alloc_(alloc, AlignUp(len, PageSize()) + 3 * PageSize())
    {
    }

    ~TProtectedContStackAllocator() override {
    }

    TStackType* DoAllocate() override {
        return new TProtectedStack(Alloc_.DoAllocate());
    }

private:
    TGenericContStackAllocatorBase Alloc_;
};

#if defined(NDEBUG) && !defined(_san_enabled_)
using TGenericContStackAllocator = TGenericContStackAllocatorBase;
#else
using TGenericContStackAllocator = TProtectedContStackAllocator;
#endif

class THeapStackAllocator: public TGenericContStackAllocator {
public:
    inline THeapStackAllocator(size_t len)
        : TGenericContStackAllocator(TDefaultAllocator::Instance(), len)
    {
    }
};

using TDefaultStackAllocator = THeapStackAllocator;
