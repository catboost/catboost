#include "exception.h"
#include "memory_utils.h"

#include <util/string/cast.h>
#include <util/charset/utf8.h>
#include <util/system/info.h>

size_t NCB::ToAllocationSize(size_t size) {
    if (size == 0) {
        return 0;
    }

    const auto pageSize = NSystemInfo::GetPageSize();

    // - allocator adds header to large allocations
    // - for large allocations allocator will allocate in pages
    size += pageSize - 1;
    if (size % pageSize == 0) {
        return size;
    }

    return size + (pageSize - size % pageSize);
}

bool IsInfinity(const TStringBuf value) {
    static const TStringBuf examples[] = {
        "",
        "no", "none",
        "off",
        "inf", "infinity",
        "unlim", "unlimited"
    };
    for (const auto example : examples) {
        if (example == value) {
            return true;
        }
    }

    return false;
}

ui64 ParseMemorySizeDescription(const TStringBuf description) {
    char* suffixBegin = nullptr;
    const double number = StrToD(description.begin(), description.end(), &suffixBegin);
    if (suffixBegin > description.begin() && number >= 0) {
        // `number` is valid
        const auto suffix = to_lower(TString(suffixBegin, description.end()));
        if (suffix == "tb") {
            return static_cast<ui64>(number * (1ll << 40));
        } else if (suffix == "gb") {
            return static_cast<ui64>(number * (1ll << 30));
        } else if (suffix == "mb") {
            return static_cast<ui64>(number * (1ll << 20));
        } else if (suffix == "kb") {
            return static_cast<ui64>(number * (1ll << 10));
        } else if (suffix == "b" || suffix.empty()) {
            return static_cast<ui64>(number);
        }
    } else {
        if (IsInfinity(ToLowerUTF8(description))) {
            return Max<ui64>();
        }
    }
    CB_ENSURE(false, "incomprehensible memory size description: " << description);
}
