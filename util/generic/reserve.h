#pragma once

namespace NDetail {
    struct [[nodiscard]] TReserveTag {
        size_t Capacity;
    };
} // namespace NDetail

constexpr ::NDetail::TReserveTag Reserve(size_t capacity) {
    return ::NDetail::TReserveTag{capacity};
}
