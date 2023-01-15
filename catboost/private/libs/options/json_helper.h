#pragma once

#include <utility>

#include "option.h"
#include "unimplemented_aware_option.h"
#include "loss_description.h"

#include <library/cpp/json/json_value.h>

#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/string/cast.h>
#include <util/system/compiler.h>

namespace NCatboostOptions {
    template <class T, bool IsEnum = std::is_enum<T>::value>
    class TJsonFieldHelper {
    public:
        static void Read(const NJson::TJsonValue& src, T* dst) {
            dst->Load(src);
        }

        static void Write(const T& value, NJson::TJsonValue* dst) {
            value.Save(dst);
        }
    };

    template <class T>
    class TJsonFieldHelper<T, true> {
    public:
        static void Read(const NJson::TJsonValue& src, T* dst) {
            (*dst) = FromString<T>(src.GetStringSafe());
        }

        static void Write(const T& value, NJson::TJsonValue* dst) {
            (*dst) = ToString(value);
        }
    };

#define DECLARE_FIELD_HELPER(cppType, type)                               \
    template <>                                                           \
    class TJsonFieldHelper<cppType, false> {                              \
    public:                                                               \
        static void Read(const NJson::TJsonValue& src, cppType* dst) {    \
            (*dst) = src.Get##type##Safe();                               \
        }                                                                 \
                                                                          \
        static void Write(const cppType& value, NJson::TJsonValue* dst) { \
            (*dst) = value;                                               \
        }                                                                 \
    };

    DECLARE_FIELD_HELPER(double, Double);

    DECLARE_FIELD_HELPER(float, Double);

    DECLARE_FIELD_HELPER(ui32, UInteger);

    DECLARE_FIELD_HELPER(int, Integer);

    DECLARE_FIELD_HELPER(ui64, UInteger);

    DECLARE_FIELD_HELPER(i64, Integer);

    DECLARE_FIELD_HELPER(TString, String);

    DECLARE_FIELD_HELPER(TStringBuf, String);

    DECLARE_FIELD_HELPER(bool, Boolean);

#undef DECLARE_FIELD_HELPER

    template <class T>
    class TJsonFieldHelper<TVector<T>, false> {
    public:
        static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TVector<T>* dst) {
            dst->clear();
            if (src.IsArray()) {
                const NJson::TJsonValue::TArray& data = src.GetArraySafe();
                dst->resize(data.size());
                for (ui32 i = 0; i < dst->size(); ++i) {
                    TJsonFieldHelper<T>::Read(data.at(i), &(*dst)[i]);
                }
            } else {
                T tmp;
                TJsonFieldHelper<T>::Read(src, &tmp);
                dst->push_back(std::move(tmp));
            }
        }

        static Y_NO_INLINE void Write(const TVector<T>& src, NJson::TJsonValue* dst) {
            (*dst) = NJson::TJsonValue(NJson::EJsonValueType::JSON_ARRAY);
            for (const auto& entry : src) {
                NJson::TJsonValue value;
                TJsonFieldHelper<T>::Write(entry, &value);
                dst->AppendValue(std::move(value));
            }
        }
    };

    template <class TKey, class T>
    class TJsonFieldHelper<TMap<TKey, T>, false> {
    public:
        static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TMap<TKey, T>* dst) {
            dst->clear();
            if (src.IsMap()) {
                const auto& data = src.GetMapSafe();
                for (const auto& entry : data) {
                    TJsonFieldHelper<T>::Read(entry.second, &((*dst)[FromString<TKey>(entry.first)]));
                }
            } else {
                ythrow TCatBoostException() << "Error: wrong json type";
            }
        }

        static Y_NO_INLINE void Write(const TMap<TKey, T>& src, NJson::TJsonValue* dst) {
            (*dst) = NJson::TJsonValue(NJson::EJsonValueType::JSON_MAP);
            for (const auto& entry : src) {
                NJson::TJsonValue value;
                TJsonFieldHelper<T>::Write(entry.second, &value);
                (*dst)[ToString<TKey>(entry.first)] = std::move(value);
            }
        }
    };

    template <>
    class TJsonFieldHelper<NJson::TJsonValue, false> {
    public:
        static void Read(const NJson::TJsonValue& src, NJson::TJsonValue* dst) {
            (*dst) = src;
        }

        static void Write(const NJson::TJsonValue& src, NJson::TJsonValue* dst) {
            (*dst) = src;
        }
    };

    template <class T>
    class TJsonFieldHelper<TOption<T>, false> {
    public:
        static Y_NO_INLINE bool Read(const NJson::TJsonValue& src, TOption<T>* dst) {
            if (dst->IsDisabled()) {
                return false;
            }

            const auto& key = dst->OptionName;
            if (src.Has(key)) {
                const auto& srcValue = src[key];
                try {
                    TJsonFieldHelper<T>::Read(srcValue, &dst->Value);
                    dst->IsSetFlag = true;
                } catch (NJson::TJsonException) {
                    ythrow TCatBoostException() << "Can't parse parameter \"" << key << "\" with value: " << srcValue;
                }
                return true;
            } else {
                return false;
            }
        }

        static Y_NO_INLINE void Write(const TOption<T>& src, NJson::TJsonValue* dst) {
            if (src.IsDisabled()) {
                return;
            }
            CB_ENSURE(dst, "Error: can't write to nullptr");
            TJsonFieldHelper<T>::Write(src.Get(), &(*dst)[src.GetName()]);
        }
    };

    template <class T>
    class TJsonFieldHelper<TMaybe<T>, false> {
    public:
        static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TMaybe<T>* dst) {
            if (src.IsNull()) {
                *dst = Nothing();
            } else {
                T value;
                TJsonFieldHelper<T>::Read(src, &value);
                *dst = value;
            }
        }

        static Y_NO_INLINE void Write(const TMaybe<T>& src, NJson::TJsonValue* dst) {
            CB_ENSURE(dst, "Error: can't write to nullptr");
            if (!src) {
                *dst = NJson::TJsonValue(NJson::EJsonValueType::JSON_NULL);
            } else {
                TJsonFieldHelper<T>::Write(src.GetRef(), dst);
            }
        }
    };

    template <>
    class TJsonFieldHelper<TLossParams, false> {
    public:
        static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TLossParams* dst) {
            CB_ENSURE(dst, "Error: can't write to nullptr");
            TVector<std::pair<TString, TString>> keyValuePairs;
            // The input JSON might be either a map of parameters or a list of key-value pairs (which are lists of
            // length 2).
            if (src.IsArray()) {
                // [["key1, "value1"], ["key2", "value2"]]
                const NJson::TJsonValue::TArray& data = src.GetArraySafe();
                TVector<TString> keyValuePair;
                for (ui32 i = 0; i < data.size(); ++i) {
                    TJsonFieldHelper<TVector<TString>, false>::Read(data.at(i), &keyValuePair);
                    CB_ENSURE(keyValuePair.size() == 2, "Error: payload must contain lists of length 2 (key-value pairs)");
                    keyValuePairs.emplace_back(keyValuePair[0], keyValuePair[1]);
                }
            } else if (src.IsMap()) {
                // {"key1": "value1", "key2": "value2"}
                CB_ENSURE(src.IsMap(), "Error: TLossParams serialized JSON is not a map nor a list.");
                const auto& data = src.GetMapSafe();
                for (const auto& entry : data) {
                    CB_ENSURE(entry.second.IsString(), "Error: TLossParams map values must be strings.");
                    keyValuePairs.emplace_back(entry.first, entry.second.GetStringSafe());
                }
            } else {
                ythrow TCatBoostException() << "Error: wrong json type";
            }
            *dst = TLossParams::FromVector(keyValuePairs);
        }

        static Y_NO_INLINE void Write(const TLossParams& src, NJson::TJsonValue* dst) {
            // Writing TLossParams as a vector of key-value pairs to preserve the params order.
            CB_ENSURE(dst, "Error: can't write to nullptr");
            TVector<TVector<TString>> keyValuePairs;
            keyValuePairs.reserve(src.GetUserSpecifiedKeyOrder().size());
            for (const auto& key : src.GetUserSpecifiedKeyOrder()) {
                const TString& value = src.GetParamsMap().at(key);
                keyValuePairs.emplace_back(TVector<TString>{key, value});
            }
            TJsonFieldHelper<TVector<TVector<TString>>, false>::Write(keyValuePairs, dst);
        }
    };

    class TUnimplementedAwareOptionsLoader {
    public:
        explicit TUnimplementedAwareOptionsLoader(const NJson::TJsonValue& src)
            : Source(src)
        {
        }

        template <typename T, class TSupportedTasks>
        Y_NO_INLINE void LoadMany(TUnimplementedAwareOption<T, TSupportedTasks>* option) {
            if (option->IsDisabled()) {
                return;
            }
            const bool keyWasFound = Source.Has(option->GetName());
            const bool isUnimplemented = option->IsUnimplementedForCurrentTask();

            if (keyWasFound && isUnimplemented) {
                switch (option->GetLoadUnimplementedPolicy()) {
                    case ELoadUnimplementedPolicy::SkipWithWarning: {
                        UnimplementedKeys.insert(option->GetName());
                        return;
                    }
                    case ELoadUnimplementedPolicy::Exception: {
                        ythrow TCatBoostException() << "Error: option " << option->GetName()
                                                    << " is unimplemented for task " << option->GetCurrentTaskType();
                    }
                    case ELoadUnimplementedPolicy::ExceptionOnChange: {
                        UnimplementedKeys.insert(option->GetName());
                        T oldValue = option->GetUnchecked();
                        LoadMany(static_cast<TOption<T>*>(option));
                        if (oldValue != option->GetUnchecked()) {
                            ythrow TCatBoostException() << "Error: change of option " << option->GetName()
                                                        << " is unimplemented for task type " << option->GetCurrentTaskType() << " and was not default in previous run";
                        }
                        return;
                    }
                    default: {
                        ythrow TCatBoostException() << "Unknown policy " << option->GetLoadUnimplementedPolicy();
                    }
                }
            }
            LoadMany(static_cast<TOption<T>*>(option));
        }

        template <typename T>
        void LoadMany(TOption<T>* option) {
            const bool keyWasFound = TJsonFieldHelper<TOption<T>>::Read(Source, option);
            if (keyWasFound) {
                ValidKeys.insert(option->GetName());
            }
        }

        template <typename T, typename... R>
        void LoadMany(T* t, R*... r) {
            LoadMany(t);
            LoadMany(r...);
        }

        void CheckForUnseenKeys() {
            for (const auto& keyVal : Source.GetMap()) {
                CB_ENSURE(ValidKeys.contains(keyVal.first) || UnimplementedKeys.contains(keyVal.first), "Invalid parameter: " << keyVal.first << Endl << Source);
            }
        }

    private:
        const NJson::TJsonValue& Source;
        TSet<TString> ValidKeys;
        TSet<TString> UnimplementedKeys;
    };

    template <typename... Fields>
    inline void CheckedLoad(const NJson::TJsonValue& source, Fields*... fields) {
        TUnimplementedAwareOptionsLoader loader(source);
        loader.LoadMany(fields...);
        loader.CheckForUnseenKeys();
    };

    class TUnimplementedAwareOptionsSaver {
    public:
        explicit TUnimplementedAwareOptionsSaver(NJson::TJsonValue* dst)
            : Result(dst)
        {
        }

        template <typename T, class TSupportedTasks>
        inline void SaveMany(const TUnimplementedAwareOption<T, TSupportedTasks>& option) {
            if (option.IsDisabled()) {
                return;
            }
            if (option.IsUnimplementedForCurrentTask()) {
                return;
            }
            SaveMany(static_cast<const TOption<T>&>(option));
        }

        template <typename T>
        void SaveMany(const TOption<T>& option) {
            TJsonFieldHelper<TOption<T>>::Write(option, Result);
        }

        template <typename T, class... TRest>
        void SaveMany(const T& option, const TRest&... rest) {
            SaveMany(option);
            SaveMany(rest...);
        }

    private:
        NJson::TJsonValue* Result;
    };

    template <typename... Fields>
    inline void SaveFields(NJson::TJsonValue* dst, const Fields&... fields) {
        TUnimplementedAwareOptionsSaver saver(dst);
        saver.SaveMany(fields...);
    };
}

template <typename T>
void FromJson(const NJson::TJsonValue& value, T* result) {
    switch (value.GetType()) {
        case NJson::EJsonValueType::JSON_INTEGER:
            *result = T(value.GetInteger());
            break;
        case NJson::EJsonValueType::JSON_DOUBLE:
            *result = T(value.GetDouble());
            break;
        case NJson::EJsonValueType::JSON_UINTEGER:
            *result = T(value.GetUInteger());
            break;
        case NJson::EJsonValueType::JSON_STRING:
            *result = FromString<T>(value.GetString());
            break;
        default:
            CB_ENSURE("Incorrect format");
    }
}

template <>
void FromJson(const NJson::TJsonValue& value, TString* result);

template <typename T>
T FromJson(const NJson::TJsonValue& value) {
    T result;
    FromJson(value, &result);
    return result;
}

NJson::TJsonValue ReadTJsonValue(TStringBuf paramsJson);

/*
 * Use this function instead of simple ToString(jsonValue) because it saves floating point values with proper precision
 */
TString WriteTJsonValue(const NJson::TJsonValue& jsonValue);
