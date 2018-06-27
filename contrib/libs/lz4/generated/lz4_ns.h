#pragma once

#include "iface.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace LZ4_NAMESPACE {

#define ONLY_COMPRESS
#include "../lz4.c"

struct TLZ4Methods ytbl = {
    LZ4_compress_default,
};

}
