#pragma once

#include <utility>

#include "option.h"
#include "unimplemented_aware_option.h"
#include "loss_description.h"

#include <catboost/libs/column_description/feature_tag.h>
#include <catboost/libs/helpers/json_helpers.h>


#include <library/cpp/json/json_value.h>
#include <library/cpp/json/json_reader.h>

#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/map.h>
#include <util/generic/maybe.h>
#include <util/string/cast.h>
#include <util/system/compiler.h>




template <class T>
class TJsonFieldHelper<NCatboostOptions::TOption<T>, false> {
public:
    static Y_NO_INLINE bool Read(const NJson::TJsonValue& src, NCatboostOptions::TOption<T>* dst) {
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

    static Y_NO_INLINE void Write(const NCatboostOptions::TOption<T>& src, NJson::TJsonValue* dst) {
        if (src.IsDisabled()) {
            return;
        }
        CB_ENSURE(dst, "Error: can't write to nullptr");
        TJsonFieldHelper<T>::Write(src.Get(), &(*dst)[src.GetName()]);
    }
};



template <>
class TJsonFieldHelper<TLossParams, false> {
public:
    constexpr static TStringBuf ParamsKeyOrderRecord = "__params_key_order";
public:
    static Y_NO_INLINE void Read(const NJson::TJsonValue& src, TLossParams* dst) {
        CB_ENSURE(dst, "Error: can't write to nullptr");
        TVector<std::pair<TString, TString>> keyValuePairs;
        // The input JSON might be either a map of parameters or a list of key-value pairs (which are lists of
        // length 2).
        if (src.IsArray()) {
            // support format from catboost 0.23.2 version
            // (don't want to break deserialization again)
            // [["key1, "value1"], ["key2", "value2"]]
            const NJson::TJsonValue::TArray& data = src.GetArraySafe();
            TVector<TString> keyValuePair;
            for (ui32 i = 0; i < data.size(); ++i) {
                TJsonFieldHelper<TVector<TString>, false>::Read(data.at(i), &keyValuePair);
                CB_ENSURE(keyValuePair.size() == 2, "Error: payload must contain lists of length 2 (key-value pairs)");
                keyValuePairs.emplace_back(keyValuePair[0], keyValuePair[1]);
            }
        } else if (src.IsMap()) {
            // {"key1": "value1", "key2": "value2", "__params_key_order": ["key2", "key1"]}
            const auto& data = src.GetMapSafe();
            if (!data.contains(ParamsKeyOrderRecord)) {
                for (const auto& entry : data) {
                    CB_ENSURE(entry.second.IsString(), "Error: TLossParams map values must be strings.");
                    keyValuePairs.emplace_back(entry.first, entry.second.GetStringSafe());
                }
            } else {
                TVector<TString> keyOrder;
                NJson::TJsonValue keyOrderJson;
                NJson::ReadJsonTree(data.at(ParamsKeyOrderRecord).GetString(), &keyOrderJson);
                TJsonFieldHelper<TVector<TString>, false>::Read(
                    keyOrderJson,
                    &keyOrder
                );
                CB_ENSURE(
                    keyOrder.size() + 1 == data.size(),
                    "Error: key order list size don't match dictionary size"
                );
                keyValuePairs.reserve(keyOrder.size());
                for (const auto& key : keyOrder) {
                    const auto& value = data.at(key);
                    CB_ENSURE(value.IsString(), "Error: TLossParams map values must be strings.");
                    keyValuePairs.emplace_back(key, value.GetString());
                }
            }
        } else {
            ythrow TCatBoostException() << "Error: TLossParams serialized JSON is not a map nor a list.";
        }
        *dst = TLossParams::FromVector(keyValuePairs);
    }

    static Y_NO_INLINE void Write(const TLossParams& src, NJson::TJsonValue* dst) {
        // Writing TLossParams as a vector of key-value pairs to preserve the params order.
        CB_ENSURE(dst, "Error: can't write to nullptr");
        TJsonFieldHelper<TMap<TString, TString>, false>::Write(src.GetParamsMap(), dst);
        if (!src.GetUserSpecifiedKeyOrder().empty()) {
            NJson::TJsonValue keyOrderList;
            TJsonFieldHelper<TVector<TString>, false>::Write(src.GetUserSpecifiedKeyOrder(), &keyOrderList);
            dst->InsertValue(
                ParamsKeyOrderRecord,
                keyOrderList.GetStringRobust()
            );
        }
    }
};

template<>
class TJsonFieldHelper<NCB::TTagDescription, false> {
public:
    static Y_NO_INLINE void Read(const NJson::TJsonValue& src, NCB::TTagDescription* dst) {
        if (src.IsMap()) {
            const auto& data = src.GetMapSafe();
            TJsonFieldHelper<TVector<ui32>>::Read(data.at("features"), &dst->Features);
            if (data.find("cost") == data.end()) {
                dst->Cost = 1.0;
            } else {
                TJsonFieldHelper<float>::Read(data.at("cost"), &dst->Cost);
            }
        } else {
            ythrow TCatBoostException() << "Error: wrong json type";
        }
    }

    static Y_NO_INLINE void Write(const NCB::TTagDescription& src, NJson::TJsonValue* dst) {
        (*dst) = NJson::TJsonValue(NJson::EJsonValueType::JSON_MAP);
        TJsonFieldHelper<TVector<ui32>>::Write(src.Features, &(*dst)["features"]);
    }
};


namespace NCatboostOptions {

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
