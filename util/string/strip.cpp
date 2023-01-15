#include "strip.h"
#include "ascii.h"

bool Collapse(const TString& from, TString& to, size_t maxLen) {
    return CollapseImpl<TString, bool (*)(unsigned char)>(from, to, maxLen, IsAsciiSpace);
}

void CollapseText(const TString& from, TString& to, size_t maxLen) {
    Collapse(from, to, maxLen);
    Strip(to);
    if (to.size() >= maxLen) {
        to.remove(maxLen - 5); // " ..."
        to.reverse();
        size_t pos = to.find_first_of(" .,;");
        if (pos != TString::npos && pos < 32)
            to.remove(0, pos + 1);
        to.reverse();
        to.append(" ...");
    }
}
