#pragma once

#include "maybe_owning_array_holder.h"

#include <catboost/private/libs/data_types/text.h>

#include <library/cpp/digest/crc32c/crc32c.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>

#include <type_traits>


namespace NCB {

    template <class T>
    ui32 UpdateCheckSum(ui32 init, const T& value);

    // override UpdateCheckSumImpl for non-trivial types

    template <class T>
    inline ui32 UpdateCheckSumImpl(ui32 init, TConstArrayRef<T> arrayRef) {
        if constexpr(std::is_trivial<T>::value) {
            return Crc32cExtend(init, arrayRef.data(), sizeof(T)*arrayRef.size());
        } else {
            ui32 checkSum = init;
            for (const auto& element : arrayRef) {
                checkSum = UpdateCheckSum(checkSum, element);
            }
            return checkSum;
        }
    }

    inline ui32 UpdateCheckSumImpl(ui32 init, const TStringBuf str) {
        return Crc32cExtend(init, str.begin(), str.size());
    }

    inline ui32 UpdateCheckSumImpl(ui32 init, const NCB::TText& text) {
        ui32 checkSum = init;
        for (const auto& tokenCount : text) {
            checkSum = UpdateCheckSum(checkSum, (ui32)tokenCount.Token());
            checkSum = UpdateCheckSum(checkSum, tokenCount.Count());
        }
        return checkSum;
    }

    template <class TKey, class TValue>
    ui32 UpdateCheckSumImpl(ui32 init, const TMap<TKey, TValue>& map) {
        ui32 checkSum = init;
        for (const auto& [key, value] : map) {
            checkSum = UpdateCheckSum(checkSum, key);
            checkSum = UpdateCheckSum(checkSum, value);
        }
        return checkSum;
    }

    template <class T>
    inline ui32 UpdateCheckSumImpl(ui32 init, const TVector<T>& vector) {
        return UpdateCheckSumImpl(init, TConstArrayRef<T>(vector));
    }

    template <class T>
    inline ui32 UpdateCheckSumImpl(ui32 init, const TMaybe<T>& value) {
        const ui8 defined = value.Defined();
        if (defined) {
            return UpdateCheckSum(UpdateCheckSum(init, defined), *value);
        } else {
            return UpdateCheckSum(init, defined);
        }
    }

    template <class T>
    inline ui32 UpdateCheckSumImpl(ui32 init, const TMaybeOwningArrayHolder<T>& value) {
        return UpdateCheckSumImpl(init, *value);
    }

    template <class T>
    inline ui32 UpdateCheckSum(ui32 init, const T& value) {
        if constexpr(std::is_trivial<T>::value) {
            return Crc32cExtend(init, &value, sizeof(value));
        } else {
            return UpdateCheckSumImpl(init, value);
        }
    }

    template <typename THead, typename... TTail>
    inline ui32 UpdateCheckSum(ui32 init, const THead& head, const TTail&... tail) {
        return UpdateCheckSum(UpdateCheckSum(init, head), tail...);
    }

}
