#include "ascii.h"

#include <util/system/yassert.h>
#include <util/system/compat.h>

int AsciiCompareIgnoreCase(const TStringBuf s1, const TStringBuf s2) noexcept {
    if (s1.size() <= s2.size()) {
        if (int cmp = ::strnicmp(s1.data(), s2.data(), s1.size())) {
            return cmp;
        }
        return (s1.size() < s2.size()) ? -1 : 0;
    }

    Y_ASSERT(s1.size() > s2.size());
    if (int cmp = ::strnicmp(s1.data(), s2.data(), s2.size())) {
        return cmp;
    }
    return 1;
}
