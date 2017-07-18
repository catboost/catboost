#pragma once

#include <stddef.h>

#if defined(__clang__)
#if __has_feature(memory_sanitizer)
extern "C" void __msan_unpoison(const volatile void* ptr, size_t size);
#ifndef __SANITIZE_MEMORY__
#define __SANITIZE_MEMORY__
#endif
#endif
#endif
