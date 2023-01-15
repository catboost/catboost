#include "utility.h"

#ifdef _MSC_VER
    #include <Windows.h>
#endif

void SecureZero(void* pointer, size_t count) noexcept {
#ifdef _MSC_VER
    SecureZeroMemory(pointer, count);
#elif defined(memset_s)
    memset_s(pointer, count, 0, count);
#else
    volatile char* vPointer = (volatile char*)pointer;

    while (count) {
        *vPointer = 0;
        vPointer++;
        count--;
    }
#endif
}
