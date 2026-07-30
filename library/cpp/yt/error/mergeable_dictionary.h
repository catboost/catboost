#pragma once

#include "error_attribute.h"

#include <library/cpp/yt/memory/type_erasure.h>

#include <library/cpp/yt/mpl/tag_invoke_cpo.h>

#include <ranges>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NMergeableRangeImpl {

struct TFn
    : public NMpl::TTagInvokeCpoBase<TFn>
{ };

////////////////////////////////////////////////////////////////////////////////

using TMergeableRange = std::vector<std::pair<TErrorAttribute::TKey, TErrorAttribute::TValue>>;

} // namespace NMergeableRangeImpl

////////////////////////////////////////////////////////////////////////////////

// Can be customized to make your dictionary satisfy CMergeableDictionary.
inline constexpr NMergeableRangeImpl::TFn AsMergeableRange = {};

////////////////////////////////////////////////////////////////////////////////

template <class T>
concept CMergeableDictionary = NMpl::CTagInvocableS<
    NMpl::TTagInvokeTag<AsMergeableRange>,
    NMergeableRangeImpl::TMergeableRange(const T&)>;

////////////////////////////////////////////////////////////////////////////////

using TAnyMergeableDictionaryRef = TAnyRef<
    TOverload<AsMergeableRange, NMergeableRangeImpl::TMergeableRange(const TErasedThis&)>>;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
