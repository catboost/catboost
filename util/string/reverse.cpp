#include "reverse.h"

#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/charset/wide_specific.h>

#include <algorithm>

void ReverseInPlace(TString& string) {
    auto* begin = string.begin();
    std::reverse(begin, begin + string.size());
}

void ReverseInPlace(TUtf16String& string) {
    auto* begin = string.begin();
    const auto len = string.size();
    auto* end = begin + string.size();

    TVector<wchar16> buffer(len);
    wchar16* rbegin = buffer.data() + len;
    for (wchar16* p = begin; p < end;) {
        const size_t symbolSize = W16SymbolSize(p, end);
        rbegin -= symbolSize;
        std::copy(p, p + symbolSize, rbegin);
        p += symbolSize;
    }
    std::copy(buffer.begin(), buffer.end(), begin);
}

void ReverseInPlace(TUtf32String& string) {
    auto* begin = string.begin();
    std::reverse(begin, begin + string.size());
}
