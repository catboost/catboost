#include <io.h>
#include <stdio.h>
#include <windows.h>

int main() {
    auto handle = (unsigned long long)_get_osfhandle(0);
    fprintf(stderr, "_get_osfhandle(0)=%llu\n", handle);
    // It look's like classic windows undocumented behaviour
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/get-osfhandle
    // _get_osfhandle returns INVALID_HANDLE_VALUE - 1 without any sign of error if specified fd was closed.
    // Working with such handle will lead to future various errors.
    if (handle + 1 == (unsigned long long)INVALID_HANDLE_VALUE) {
        return 1;
    }
    return 0;
}
