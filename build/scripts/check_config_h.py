import sys

data = """
#if defined(SIZEOF_LONG)
static_assert(sizeof(long) == SIZEOF_LONG, "fixme 1");
#endif

#if defined(SIZEOF_PTHREAD_T)
#include <pthread.h>

static_assert(sizeof(pthread_t) == SIZEOF_PTHREAD_T, "fixme 2");
#endif

#if defined(SIZEOF_SIZE_T)
#include <stddef.h>

static_assert(sizeof(size_t) == SIZEOF_SIZE_T, "fixme 3");
#endif

#if defined(SIZEOF_TIME_T)
#include <time.h>

static_assert(sizeof(time_t) == SIZEOF_TIME_T, "fixme 4");
#endif

#if defined(SIZEOF_UINTPTR_T)
#include <stdint.h>

static_assert(sizeof(uintptr_t) == SIZEOF_UINTPTR_T, "fixme 5");
#endif

#if defined(SIZEOF_VOID_P)
static_assert(sizeof(void*) == SIZEOF_VOID_P, "fixme 6");
#endif

#if defined(SIZEOF_FPOS_T)
#include <stdio.h>

static_assert(sizeof(fpos_t) == SIZEOF_FPOS_T, "fixme 7");
#endif

#if defined(SIZEOF_DOUBLE)
static_assert(sizeof(double) == SIZEOF_DOUBLE, "fixme 8");
#endif

#if defined(SIZEOF_LONG_DOUBLE)
static_assert(sizeof(long double) == SIZEOF_LONG_DOUBLE, "fixme 9");
#endif

#if defined(SIZEOF_FLOAT)
static_assert(sizeof(float) == SIZEOF_FLOAT, "fixme 10");
#endif

#if defined(SIZEOF_INT)
static_assert(sizeof(int) == SIZEOF_INT, "fixme 11");
#endif

#if defined(SIZEOF_LONG_LONG)
static_assert(sizeof(long long) == SIZEOF_LONG_LONG, "fixme 12");
#endif

#if defined(SIZEOF_OFF_T)
#include <stdio.h>

static_assert(sizeof(off_t) == SIZEOF_OFF_T, "fixme 13");
#endif

#if defined(SIZEOF_PID_T)
#include <unistd.h>

static_assert(sizeof(pid_t) == SIZEOF_PID_T, "fixme 14");
#endif

#if defined(SIZEOF_SHORT)
static_assert(sizeof(short) == SIZEOF_SHORT, "fixme 15");
#endif

#if defined(SIZEOF_WCHAR_T)
static_assert(sizeof(wchar_t) == SIZEOF_WCHAR_T, "fixme 16");
#endif

#if defined(SIZEOF__BOOL)
//TODO
#endif

#if defined(ALIGNOF_VOID_P)
static_assert(alignof(void*) == ALIGNOF_VOID_P, "fixme 18");
#endif

#if defined(ALIGNOF_DOUBLE)
static_assert(alignof(double) == ALIGNOF_DOUBLE, "fixme 19");
#endif
"""
if __name__ == '__main__':
    with open(sys.argv[2], 'w') as f:
        f.write('#include <' + sys.argv[1] + '>\n\n')
        f.write(data)
