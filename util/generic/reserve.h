#pragma once

namespace NDetail {
    struct TReserveTag {
        size_t Capacity;
    };
}

inline ::NDetail::TReserveTag Reserve(size_t capacity) {
    return ::NDetail::TReserveTag{capacity};
}
