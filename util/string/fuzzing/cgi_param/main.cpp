#include <util/string/cgiparam.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    try {
        TCgiParameters(TStringBuf((const char*)data, size));
    } catch (...) {
    }

    return 0; // Non-zero return values are reserved for future use.
}
