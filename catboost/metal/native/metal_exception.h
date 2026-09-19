#pragma once

#include <stddef.h>

// Keep the typed C++ catch boundary in a C++ translation unit: the Objective-C++
// personality does not match CatBoost's vendored C++ exception types reliably.
// The callback must handle Objective-C exceptions before returning through here.
int CBMInvokeCppGuard(int (*callback)(void*), void* context, char* error, size_t capacity);
