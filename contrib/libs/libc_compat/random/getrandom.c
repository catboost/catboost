#include <sys/random.h>
#include "syscall.h"

#if defined(__has_feature)
	#if __has_feature(memory_sanitizer)
		#include <sanitizer/msan_interface.h>
	#endif
#endif

ssize_t getrandom(void *buf, size_t buflen, unsigned flags)
{
#if defined(__has_feature)
	#if __has_feature(memory_sanitizer)
	__msan_unpoison(buf, buflen);
	#endif
#endif
	return syscall(SYS_getrandom, buf, buflen, flags);
}
