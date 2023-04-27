#include "jconfigint-linux.h"

#undef INLINE
#define INLINE __forceinline

#undef HAVE_BUILTIN_CTZL

#undef WEAK
#define WEAK

#undef THREAD_LOCAL
#define THREAD_LOCAL __declspec(thread)
