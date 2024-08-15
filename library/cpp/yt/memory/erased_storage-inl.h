#ifndef ERASED_STORAGE_INL_H_
#error "Direct inclusion of this file is not allowed, include erased_storage.h"
// For the sake of sane code completion.
#include "erased_storage.h"
#endif

#include <algorithm>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <size_t MaxByteSize>
template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
TErasedStorage<MaxByteSize>::TErasedStorage(TDecayedConcrete concrete) noexcept
{
    // NB(arkady-e1ppa): We want to be able to compare
    // erased objects as if they are not erased.
    // Assuming erased type's operator ==
    // is equivalent to bitwise comparison.
    std::ranges::fill(Bytes_, std::byte(0));
    std::construct_at(
        &AsConcrete<TDecayedConcrete>(),
        concrete);
}

template <size_t MaxByteSize>
template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
TDecayedConcrete& TErasedStorage<MaxByteSize>::AsConcrete() & noexcept
{
    using TPtr = TDecayedConcrete*;
    return *std::launder(reinterpret_cast<TPtr>(&Bytes_));
}

template <size_t MaxByteSize>
template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
const TDecayedConcrete& TErasedStorage<MaxByteSize>::AsConcrete() const & noexcept
{
    using TPtr = const TDecayedConcrete*;
    return *std::launder(reinterpret_cast<TPtr>(&Bytes_));
}

template <size_t MaxByteSize>
template <CTriviallyErasable<MaxByteSize> TDecayedConcrete>
TDecayedConcrete&& TErasedStorage<MaxByteSize>::AsConcrete() && noexcept
{
    using TPtr = TDecayedConcrete*;
    return std::move(*std::launder(reinterpret_cast<TPtr>(&Bytes_)));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
