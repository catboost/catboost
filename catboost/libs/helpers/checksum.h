#pragma once

#include <library/digest/crc32c/crc32c.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/map.h>
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

    template <class TKey, class TValue>
    ui32 UpdateCheckSumImpl(ui32 init, const THashMap<TKey, TValue>& hashMap) {
        ui32 checkSum = init;
        for (const auto& [key, value] : hashMap) {
            ui32 pairCheckSum = UpdateCheckSum(0, key);
            pairCheckSum = UpdateCheckSum(pairCheckSum, value);
            checkSum = checkSum ^ pairCheckSum;
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
