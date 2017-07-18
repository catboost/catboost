#include <util/string/strip.h>
#include <util/charset/wide.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    TUtf16String w((const wchar16*)data, size / 2);
    Collapse(w);

    TString s((const char*)data, size);
    Collapse(s);

    return 0; // Non-zero return values are reserved for future use.
}
