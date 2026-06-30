#include "va_args.h"

// Test that it compiles
#define __DUMMY__(x)
Y_MAP_ARGS(__DUMMY__, 1, 2, 3)
#define __DUMMY_LAST__(x)
Y_MAP_ARGS_WITH_LAST(__DUMMY__, __DUMMY_LAST__, 1, 2, 3)
#undef __DUMMY_LAST__
#undef __DUMMY__

#define __MULTI_DUMMY__(x, y)
#define __MULTI_DUMMY_PROXY__(x) __MULTI_DUMMY__ x
Y_MAP_ARGS(__MULTI_DUMMY_PROXY__, (1, 2), (3, 4))
#undef __MULTI_DUMMY_PROXY__
#undef __MULTI_DUMMY__
