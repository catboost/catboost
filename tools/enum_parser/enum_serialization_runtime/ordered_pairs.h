#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/strbuf.h>

namespace NEnumSerializationRuntime {
    enum class ESortOrder: int {
        Unordered = 0,
        Ascending = 1,
        StrictlyAscending = 2,
        DirectMapping = 3,
    };

    template <typename TEnumRepresentationType>
    struct TEnumStringPair {
        const TEnumRepresentationType Key;
        const TStringBuf Name;
    };

    template <typename TEnumRepresentationType>
    constexpr ESortOrder GetKeyFieldSortOrder(const TArrayRef<const TEnumStringPair<TEnumRepresentationType>> initializer) {
        if (initializer.empty()) {
            return ESortOrder::DirectMapping;
        }
        bool direct = true;
        bool strict = true;
        bool sorted = true;
        TEnumRepresentationType expected = initializer.data()[0].Key;
        for (size_t i = 1; i < initializer.size(); ++i) {
            const auto& prev = initializer.data()[i - 1].Key;
            const auto& next = initializer.data()[i - 0].Key;
            if (++expected != next) {
                direct = false;
            }
            if (prev >= next) {
                strict = false;
            }
            if (prev > next) {
                sorted = false;
                break;
            }
        }
        return direct   ? ESortOrder::DirectMapping
               : strict ? ESortOrder::StrictlyAscending
               : sorted ? ESortOrder::Ascending
                        : ESortOrder::Unordered;
    }

    template <typename TEnumRepresentationType>
    constexpr ESortOrder GetNameFieldSortOrder(const TArrayRef<const TEnumStringPair<TEnumRepresentationType>> initializer) {
        if (initializer.empty()) {
            return ESortOrder::DirectMapping;
        }
        bool strict = true;
        bool sorted = true;
        for (size_t i = 1; i < initializer.size(); ++i) {
            const std::string_view prev = initializer.data()[i - 1].Name;
            const std::string_view next = initializer.data()[i - 0].Name;
            if (prev >= next) {
                strict = false;
            }
            if (prev > next) {
                sorted = false;
                break;
            }
        }
        return strict   ? ESortOrder::StrictlyAscending
               : sorted ? ESortOrder::Ascending
                        : ESortOrder::Unordered;
    }
}
