#pragma once

#include <exception>

#include <cstddef>

namespace NCB {
    // Facilities to verify integrity and consistency of FlatBuffers. For each FlatBuffer user must
    // provide definition of template (better in .cpp), otherwise it will result in linkage error.
    //

    template <typename T>
    void ValidateFlatBuffer(const void* data, size_t size);

    template <typename T>
    inline bool IsValidFlatBuffer(const void* data, size_t size) {
        try {
            ValidateFlatBuffer<T>(data, size);
        } catch (const std::exception&) {
            return false;
        }

        return true;
    }
}
