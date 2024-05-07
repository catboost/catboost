#ifndef ERASED_STORAGE_INL_H_
#error "Direct inclusion of this file is not allowed, include erased_storage.h"
// For the sake of sane code completion.
#include "erased_storage.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <CTriviallyErasable TDecayedConcrete>
TErasedStorage::TErasedStorage(TDecayedConcrete concrete) noexcept
{
    std::construct_at(
        &AsConcrete<TDecayedConcrete>(),
        concrete);
}

template <CTriviallyErasable TDecayedConcrete>
TDecayedConcrete& TErasedStorage::AsConcrete() & noexcept
{
    using TPtr = TDecayedConcrete*;
    return *std::launder(reinterpret_cast<TPtr>(&Bytes_));
}

template <CTriviallyErasable TDecayedConcrete>
const TDecayedConcrete& TErasedStorage::AsConcrete() const & noexcept
{
    using TPtr = const TDecayedConcrete*;
    return *std::launder(reinterpret_cast<TPtr>(&Bytes_));
}

template <CTriviallyErasable TDecayedConcrete>
TDecayedConcrete&& TErasedStorage::AsConcrete() && noexcept
{
    using TPtr = TDecayedConcrete*;
    return std::move(*std::launder(reinterpret_cast<TPtr>(&Bytes_)));
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
