#pragma once

#include "library/cpp/json/writer/json_value.h"

#include <limits>
#include <util/generic/array_ref.h>
#include <util/generic/deque.h>
#include <util/generic/hash.h>
#include <util/generic/list.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>


namespace NJson {
    template<typename T>
    struct TConverter {
    };

    namespace {
        template<typename T>
        struct TDefaultEncoder {
            static inline TJsonValue Encode(T value) {
                return TJsonValue(value);
            }
        };

        template<typename T, typename E>
        struct TDefaultArrayEncoder {
            static TJsonValue Encode(const T& value) {
                TJsonValue result(NJson::JSON_ARRAY);
                auto& encodedArray = result.GetArraySafe();
                for (const auto& element : value) {
                    encodedArray.push_back(TConverter<E>::Encode(element));
                }
                return result;
            }
        };

        template<typename T, typename E>
        struct TDefaultArrayDecoder {
            static T Decode(const TJsonValue& value) {
                T result;
                for (const auto& element : value.GetArraySafe()) {
                    result.push_back(TConverter<E>::Decode(element));
                }
                return result;
            }
        };

        template<typename T, typename E>
        struct TDefaultArrayConverter: public TDefaultArrayEncoder<T, E>, public TDefaultArrayDecoder<T, E> {
        };

        template<typename T, typename E>
        struct TDefaultMapEncoder {
            static TJsonValue Encode(const T& value) {
                TJsonValue result(NJson::JSON_MAP);
                auto& encodedMap = result.GetMapSafe();
                for (const auto& [key, element] : value) {
                    encodedMap[key] = TConverter<E>::Encode(element);
                }
                return result;
            }
        };

        template<typename T, typename E>
        struct TDefaultMapDecoder {
            static T Decode(const TJsonValue& value) {
                T result;
                for (const auto& [key, element] : value.GetMapSafe()) {
                    result[key] = TConverter<E>::Decode(element);
                }
                return result;
            }
        };

        template<typename T, typename E>
        struct TDefaultMapConverter: public TDefaultMapEncoder<T, E>, public TDefaultMapDecoder<T, E> {
        };
    }

    template<>
    struct TConverter<TJsonValue> {
        static TJsonValue Encode(const TJsonValue& value) {
            return value;
        }

        static TJsonValue Decode(const TJsonValue& value) {
            return value;
        }
    };

    template<>
    struct TConverter<bool>: public TDefaultEncoder<bool> {
        static inline bool Decode(const TJsonValue& value) {
            return value.GetBooleanSafe();
        }
    };

    template<typename T>
    requires std::is_integral_v<T> && (!std::is_same_v<T, bool>)
    struct TConverter<T>: public TDefaultEncoder<T> {
        static T Decode(const TJsonValue& value) {
            if constexpr (std::is_signed_v<T>) {
                const auto decodedValue = value.GetIntegerSafe();
                if (decodedValue < std::numeric_limits<T>::min() || std::numeric_limits<T>::max() < decodedValue) {
                    ythrow yexception() << "Out of range (got " << decodedValue << ")";
                }
                return static_cast<T>(decodedValue);
            } else {
                const auto decodedValue = value.GetUIntegerSafe();
                if (std::numeric_limits<T>::max() < decodedValue) {
                    ythrow yexception() << "Out of range (got " << decodedValue << ")";
                }
                return static_cast<T>(decodedValue);
            }
        }
    };

    template<typename T>
    requires std::is_floating_point_v<T>
    struct TConverter<T>: public TDefaultEncoder<T> {
        static inline T Decode(const TJsonValue& value) {
            return static_cast<T>(value.GetDoubleSafe());
        }
    };

    template<>
    struct TConverter<TStringBuf>: public TDefaultEncoder<TStringBuf> {
    };

    template<>
    struct TConverter<TString>: public TDefaultEncoder<TString> {
        static inline TString Decode(const TJsonValue& value) {
            return value.GetStringSafe();
        }
    };

    template<typename T>
    struct TConverter<TMaybe<T>> {
        static TJsonValue Encode(const TMaybe<T>& value) {
            if (value.Defined()) {
                return TConverter<T>::Encode(*value);
            } else {
                return TJsonValue(NJson::JSON_NULL);
            }
        }

        static TMaybe<T> Decode(const TJsonValue& value) {
            if (value.IsDefined()) {
                return TConverter<T>::Decode(value);
            } else {
                return Nothing();
            }
        }
    };

    template<typename T>
    struct TConverter<TArrayRef<T>>: public TDefaultArrayEncoder<TArrayRef<T>, T> {
    };

    template<typename T>
    struct TConverter<TVector<T>>: public TDefaultArrayConverter<TVector<T>, T> {
    };

    template<typename T>
    struct TConverter<TList<T>>: public TDefaultArrayConverter<TList<T>, T> {
    };

    template<typename T>
    struct TConverter<TDeque<T>>: public TDefaultArrayConverter<TDeque<T>, T> {
    };

    template<typename T>
    struct TConverter<THashMap<TStringBuf, T>>: public TDefaultMapEncoder<THashMap<TStringBuf, T>, T> {
    };

    template<typename T>
    struct TConverter<THashMap<TString, T>>: public TDefaultMapConverter<THashMap<TString, T>, T> {
    };

    template<typename T>
    struct TConverter<TMap<TStringBuf, T>>: public TDefaultMapEncoder<TMap<TStringBuf, T>, T> {
    };

    template<typename T>
    struct TConverter<TMap<TString, T>>: public TDefaultMapConverter<TMap<TString, T>, T> {
    };
}
