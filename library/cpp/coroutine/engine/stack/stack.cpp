#include "stack.h"

#include "stack_allocator.h"
#include "stack_guards.h"


namespace NCoro::NStack {

namespace NDetails {

    TStack::TStack(void* rawMemory, void* alignedMemory, size_t alignedSize, const char* /*name*/)
            : RawMemory_((char*)rawMemory)
            , AlignedMemory_((char*)alignedMemory)
            , Size_(alignedSize)
    {
        Y_ASSERT(AlignedMemory_ && RawMemory_ && Size_);
        Y_ASSERT(!(Size_ & PageSizeMask));
        Y_ASSERT(!((size_t)AlignedMemory_ & PageSizeMask));
    }

    TStack::TStack(TStack&& rhs) noexcept
            : RawMemory_(rhs.RawMemory_)
            , AlignedMemory_(rhs.AlignedMemory_)
            , Size_(rhs.Size_)
    {
        rhs.Reset();
    }

    TStack& TStack::operator=(TStack&& rhs) noexcept {
        std::swap(*this, rhs);
        rhs.Reset();
        return *this;
    }

    void TStack::Reset() noexcept {
        Y_ASSERT(AlignedMemory_ && RawMemory_ && Size_);

        RawMemory_ = nullptr;
        AlignedMemory_ = nullptr;
        Size_ = 0;
    }

} // namespace NDetails


    TStackHolder::TStackHolder(NStack::IAllocator& allocator, uint32_t size, const char* name) noexcept
        : Allocator_(allocator)
        , Stack_(Allocator_.AllocStack(size, name))
    {}

    TStackHolder::~TStackHolder() {
        Allocator_.FreeStack(Stack_);
    }

    TArrayRef<char> TStackHolder::Get() noexcept {
        return Allocator_.GetStackWorkspace(Stack_.GetAlignedMemory(), Stack_.GetSize());
    }

    bool TStackHolder::LowerCanaryOk() const noexcept {
        return Allocator_.CheckStackOverflow(Stack_.GetAlignedMemory());
    }

    bool TStackHolder::UpperCanaryOk() const noexcept {
        return Allocator_.CheckStackOverride(Stack_.GetAlignedMemory(), Stack_.GetSize());
    }

}
