#pragma once

#include <library/malloc/api/malloc.h>

namespace NMalloc {

    TAllocatorPlugin CreateLFPlugin(size_t signature);

}
