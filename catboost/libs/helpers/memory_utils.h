#pragma once

#include <cstddef>
#include <util/generic/strbuf.h>
#include <util/system/types.h>

namespace NCB {
    size_t ToAllocationSize(size_t size);
}

bool IsInfinity(const TStringBuf value);

ui64 ParseMemorySizeDescription(TStringBuf memSizeDescription);
