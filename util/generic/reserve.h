#pragma once

namespace NDetail {
    struct TReserveTag {
        size_t Capacity;
    };
}

constexpr ::NDetail::TReserveTag Reserve(size_t capacity) {
    return ::NDetail::TReserveTag{capacity};
}
