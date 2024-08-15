#pragma once

#include <concepts>
#include <memory>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <size_t MaxByteSize>
class TErasedStorage;

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class T>
struct TIsErasedStorage
    : public std::false_type
{ };

template <size_t N>
struct TIsErasedStorage<TErasedStorage<N>>
    : public std::true_type
{ };

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class T, size_t ErasedStorageMaxByteSize>
concept CTriviallyErasable =
    std::default_initializable<T> &&
    std::is_trivially_destructible_v<T> &&
    std::is_trivially_copyable_v<T> &&
    (sizeof(T) <= ErasedStorageMaxByteSize) &&
    (alignof(T) <= ErasedStorageMaxByteSize) &&
    !std::is_reference_v<T> &&
    !NDetail::TIsErasedStorage<T>::value;

////////////////////////////////////////////////////////////////////////////////

// This class does not call dtor of erased object
// thus we require trivial destructability.
template <size_t MaxByteSize>
class TErasedStorage
{
public:
    static constexpr size_t ByteSize = MaxByteSize;

    template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
    explicit TErasedStorage(TDecayedConcrete concrete) noexcept;

    TErasedStorage(const TErasedStorage& other) = default;
    TErasedStorage& operator=(const TErasedStorage& other) = default;

    template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
    TDecayedConcrete& AsConcrete() & noexcept;

    template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
    const TDecayedConcrete& AsConcrete() const & noexcept;

    template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
    TDecayedConcrete&& AsConcrete() && noexcept;

    bool operator==(const TErasedStorage& other) const = default;

private:
    // NB(arkady-e1ppa): aligned_storage is deprecated.
    alignas(MaxByteSize) std::byte Bytes_[MaxByteSize];
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ERASED_STORAGE_INL_H_
#include "erased_storage-inl.h"
#undef ERASED_STORAGE_INL_H_
