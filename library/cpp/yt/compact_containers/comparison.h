#pragma once

#include <concepts>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CHasIsTransparentFlag = requires {
    typename T::is_transparent;
};

template <class T, class U, class TCompare>
concept CComparisonAllowed = std::same_as<T, U> || CHasIsTransparentFlag<TCompare>;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
