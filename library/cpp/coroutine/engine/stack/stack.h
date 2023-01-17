#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/noncopyable.h>

#include <cstdint>


namespace NCoro::NStack {

    class IAllocator;

namespace NDetails {

    //! Do not use directly, use TStackHolder instead
    class TStack final : private TMoveOnly {
    public:
        /*! rawMemory: can be used by unaligned allocator to free stack memory after use
         *  alignedMemory: pointer to aligned memory on which stack workspace and guard are actually placed
         *  alignedSize: size of workspace memory + memory for guard
         *  guard: guard to protect this stack
         *  name: name of coroutine for which this stack is allocated
         */
        TStack(void* rawMemory, void* alignedMemory, size_t alignedSize, const char* name);
        TStack(TStack&& rhs) noexcept;
        TStack& operator=(TStack&& rhs) noexcept;

        char* GetRawMemory() const noexcept {
            return RawMemory_;
        }

        char* GetAlignedMemory() const noexcept {
            return AlignedMemory_;
        }

        //! Stack size (includes memory for guard)
        size_t GetSize() const noexcept {
            return Size_;
        }

        //! Resets parameters, should be called after stack memory is freed
        void Reset() noexcept;

    private:
        char* RawMemory_ = nullptr; // not owned
        char* AlignedMemory_ = nullptr; // not owned
        size_t Size_ = 0;
    };

} // namespace NDetails

    class TStackHolder final : private TMoveOnly {
    public:
        explicit TStackHolder(IAllocator& allocator, uint32_t size, const char* name) noexcept;
        TStackHolder(TStackHolder&&) = default;
        TStackHolder& operator=(TStackHolder&&) = default;

        ~TStackHolder();

        char* GetAlignedMemory() const noexcept {
            return Stack_.GetAlignedMemory();
        }
        size_t GetSize() const noexcept {
            return Stack_.GetSize();
        }

        TArrayRef<char> Get() noexcept;
        bool LowerCanaryOk() const noexcept;
        bool UpperCanaryOk() const noexcept;

    private:
        IAllocator& Allocator_;
        NDetails::TStack Stack_;
    };

}
