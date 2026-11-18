#include "metal_exception.h"

#include <algorithm>
#include <cstring>
#include <exception>

namespace {
void CopyExceptionText(char* destination, size_t capacity, const char* source) {
    if (!destination || !capacity) return;
    if (!source) source = "Unexpected failure in Metal training";
    const size_t length = std::min(capacity - 1, std::strlen(source));
    std::memcpy(destination, source, length);
    destination[length] = '\0';
}
}

__attribute__((noinline))
int CBMInvokeCppGuard(int (*callback)(void*), void* context, char* error, size_t capacity) {
    try {
        return callback(context);
    } catch (const std::exception& exception) {
        CopyExceptionText(error, capacity, exception.what());
    } catch (...) {
        CopyExceptionText(error, capacity, "Unexpected failure in Metal training");
    }
    return 1;
}
