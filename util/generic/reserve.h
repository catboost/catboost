#pragma once

namespace NDetail {
    struct TReserveTag {
        size_t Capacity;
    };
} // namespace NDetail

constexpr ::NDetail::TReserveTag Reserve(size_t capacity) {
    return ::NDetail::TReserveTag{capacity};
}
