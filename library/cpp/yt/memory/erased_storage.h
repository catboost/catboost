#pragma once

#include <concepts>
#include <memory>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

constexpr size_t ErasedStorageMaxByteSize = 32;

////////////////////////////////////////////////////////////////////////////////

class TErasedStorage;

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CTriviallyErasable =
    std::default_initializable<T> &&
    std::is_trivially_destructible_v<T> &&
    std::is_trivially_copyable_v<T> &&
    (sizeof(T) <= ErasedStorageMaxByteSize) &&
    (alignof(T) <= ErasedStorageMaxByteSize) &&
    !std::is_reference_v<T> &&
    !std::same_as<T, TErasedStorage>;

////////////////////////////////////////////////////////////////////////////////

// This class does not call dtor of erased object
// thus we require trivial destructability.
class TErasedStorage
{
public:
    template <CTriviallyErasable TDecayedConcrete>
    explicit TErasedStorage(TDecayedConcrete concrete) noexcept;

    TErasedStorage(const TErasedStorage& other) = default;
    TErasedStorage& operator=(const TErasedStorage& other) = default;

    template <CTriviallyErasable TDecayedConcrete>
    TDecayedConcrete& AsConcrete() & noexcept;

    template <CTriviallyErasable TDecayedConcrete>
    const TDecayedConcrete& AsConcrete() const & noexcept;

    template <CTriviallyErasable TDecayedConcrete>
    TDecayedConcrete&& AsConcrete() && noexcept;

private:
    // NB(arkady-e1ppa): aligned_storage is deprecated.
    alignas(ErasedStorageMaxByteSize) std::byte Bytes_[ErasedStorageMaxByteSize];
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ERASED_STORAGE_INL_H_
#include "erased_storage-inl.h"
#undef ERASED_STORAGE_INL_H_
