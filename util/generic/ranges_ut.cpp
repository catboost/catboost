#include <version>

#if __cplusplus >= 202002L && defined(__cpp_lib_concepts)

// clang-format off
#include "ranges.h"

#include <vector>
#include <map>
#include <ranges>
// clang-format on

static_assert(CInputRangeOf<std::vector<int>, int>);
static_assert(!CInputRangeOf<std::vector<unsigned int>, int>);

auto MapValuesView() {
    static const std::map<int, char> map;
    return map | std::views::values;
}

static_assert(CInputRangeOf<decltype(MapValuesView()), char>);
static_assert(CInputRangeOf<decltype(std::views::iota(1)), int>);

#endif
