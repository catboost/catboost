#pragma once

#include <exception>
#include <cstddef>

namespace NCB {
    template <typename T>
    void ValidateFlatBuffer(const void* data, size_t size);

    template <typename T>
    bool IsValidFlatBuffer(const void* data, size_t size) {
        try {
            ValidateFlatBuffer<T>(data, size);
        } catch (const std::exception&) {
            return false;
        }

        return true;
    }
}
