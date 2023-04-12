#include "strip.h"
#include "ascii.h"

#include <util/string/reverse.h>

void CollapseText(const TString& from, TString& to, size_t maxLen) {
    Collapse(from, to, maxLen);
    StripInPlace(to);
    if (to.size() >= maxLen) {
        to.remove(maxLen - 5); // " ..."
        ReverseInPlace(to);
        size_t pos = to.find_first_of(" .,;");
        if (pos != TString::npos && pos < 32) {
            to.remove(0, pos + 1);
        }
        ReverseInPlace(to);
        to.append(" ...");
    }
}
