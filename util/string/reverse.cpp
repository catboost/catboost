#include "reverse.h"

#include <util/generic/string.h>

void ReverseInPlace(TString& string) {
    string.reverse();
}

void ReverseInPlace(TUtf16String& string) {
    string.reverse();
}

void ReverseInPlace(TUtf32String& string) {
    string.reverse();
}
