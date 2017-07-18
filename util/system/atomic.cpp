#include "atomic.h"

static_assert(sizeof(TAtomic) == sizeof(void*), "expect sizeof(TAtomic) == sizeof(void*)");
