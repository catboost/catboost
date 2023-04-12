#pragma once

#include "ordered_pairs.h"

#include <util/generic/strbuf.h>
#include <util/stream/fwd.h>

namespace NEnumSerializationRuntime {
    // compile-time conversion method selection

    constexpr size_t LINEAR_SEARCH_KEYS_SIZE_THRESHOLD = 6;
    constexpr size_t LINEAR_SEARCH_VALUES_SIZE_THRESHOLD = 2;

    template <class TNameBufs, typename EEnum>
    inline void DispatchOutFn(IOutputStream& os, EEnum n) {
        constexpr auto order = TNameBufs::NamesOrder;
        if constexpr (order >= ::NEnumSerializationRuntime::ESortOrder::DirectMapping) {
            return TNameBufs::OutDirect(&os, n, TNameBufs::EnumInitializationData);
        } else if constexpr (std::size(TNameBufs::EnumInitializationData.NamesInitializer) <= LINEAR_SEARCH_KEYS_SIZE_THRESHOLD) {
            return TNameBufs::OutFullScan(&os, n, TNameBufs::EnumInitializationData);
        } else if constexpr (order >= ::NEnumSerializationRuntime::ESortOrder::Ascending) {
            return TNameBufs::OutSorted(&os, n, TNameBufs::EnumInitializationData);
        } else {
            const TNameBufs& names = TNameBufs::Instance();
            return names.Out(&os, n);
        }
    }

    template <class TNameBufs, typename EEnum>
    inline TStringBuf DispatchToStringBufFn(EEnum n) {
        constexpr auto order = TNameBufs::NamesOrder;
        if constexpr (order >= ::NEnumSerializationRuntime::ESortOrder::DirectMapping) {
            return TNameBufs::ToStringBufDirect(n, TNameBufs::EnumInitializationData);
        } else if constexpr (std::size(TNameBufs::EnumInitializationData.NamesInitializer) <= LINEAR_SEARCH_KEYS_SIZE_THRESHOLD) {
            return TNameBufs::ToStringBufFullScan(n, TNameBufs::EnumInitializationData);
        } else if constexpr (order >= ::NEnumSerializationRuntime::ESortOrder::Ascending) {
            return TNameBufs::ToStringBufSorted(n, TNameBufs::EnumInitializationData);
        } else {
            const TNameBufs& names = TNameBufs::Instance();
            return names.ToStringBuf(n);
        }
    }

    template <class TNameBufs, typename EEnum>
    inline EEnum DispatchFromStringImplFn(const char* data, size_t len) {
        const TStringBuf name{data, len};
        constexpr auto order = TNameBufs::ValuesOrder;
        static_assert(order >= ::NEnumSerializationRuntime::ESortOrder::Ascending, "enum_parser produced unsorted output"); // comment this line to use run-time sort for temporary workaround
        if constexpr (std::size(TNameBufs::EnumInitializationData.ValuesInitializer) <= LINEAR_SEARCH_VALUES_SIZE_THRESHOLD) {
            return TNameBufs::FromStringFullScan(name, TNameBufs::EnumInitializationData);
        } else if constexpr (order >= ::NEnumSerializationRuntime::ESortOrder::Ascending) {
            return TNameBufs::FromStringSorted(name, TNameBufs::EnumInitializationData);
        } else {
            const TNameBufs& names = TNameBufs::Instance();
            return names.FromString(name);
        }
    }

    template <class TNameBufs, typename EEnum>
    inline bool DispatchTryFromStringImplFn(const char* data, size_t len, EEnum& result) {
        const TStringBuf name{data, len};
        constexpr auto order = TNameBufs::ValuesOrder;
        if constexpr (std::size(TNameBufs::EnumInitializationData.ValuesInitializer) <= LINEAR_SEARCH_VALUES_SIZE_THRESHOLD) {
            return TNameBufs::TryFromStringFullScan(name, result, TNameBufs::EnumInitializationData);
        } else if constexpr (order >= ::NEnumSerializationRuntime::ESortOrder::Ascending) {
            return TNameBufs::TryFromStringSorted(name, result, TNameBufs::EnumInitializationData);
        } else {
            const TNameBufs& names = TNameBufs::Instance();
            return names.FromString(name, result);
        }
    }
}
