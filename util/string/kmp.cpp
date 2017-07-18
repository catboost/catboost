#include "kmp.h"

#include <util/generic/yexception.h>

TKMPMatcher::TKMPMatcher(const char* patternBegin, const char* patternEnd)
    : Pattern(patternBegin, patternEnd)
{
    ComputePrefixFunction();
}

TKMPMatcher::TKMPMatcher(const TString& pattern)
    : Pattern(pattern)
{
    ComputePrefixFunction();
}

void TKMPMatcher::ComputePrefixFunction() {
    ssize_t* pf;
    ::ComputePrefixFunction(~Pattern, ~Pattern + +Pattern, &pf);
    PrefixFunction.Reset(pf);
}
