#include <util/string/cast.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    double res;

    TryFromString<double>((const char*)data, size, res);

    return 0; // Non-zero return values are reserved for future use.
}
