#pragma once

#if defined(__clang__)
# if __has_feature(memory_sanitizer)
void __msan_unpoison(const volatile void* a, size_t size);
# else
#  define __msan_unpoison(a, size)
# endif
#else
# define __msan_unpoison(a, size)
#endif  // __clang__
