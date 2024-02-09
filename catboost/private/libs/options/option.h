#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <util/generic/string.h>

#include <utility>

namespace NCatboostOptions {
    template <class TValue>
    class TOption {
    public:
        TOption(TString key,
                const TValue& defaultValue)
            : Value(defaultValue)
            , DefaultValue(defaultValue)
            , OptionName(std::move(key))
        {
        }

        TOption(const TOption& other) = default;

        virtual ~TOption() {
        }

        template <class T>
        void SetDefault(T&& value) {
            DefaultValue =  std::forward<T>(value);
            if (!IsSetFlag) {
                Value = DefaultValue;
            }
        }

        const TValue& GetDefaultValue() const {
            return DefaultValue;
        }

        template <class T>
        void Set(T&& value) {
            Value = std::forward<T>(value);
            IsSetFlag = true;
        }

        void Reset() {
            Value = DefaultValue;
            IsSetFlag = false;
        }

        virtual const TValue& Get() const {
            return Value;
        }

        virtual TValue& Get() {
            CB_ENSURE(!IsDisabled(), "Error: option " << OptionName << " is disabled");
            return Value;
        }

        inline TValue* operator->() noexcept {
            return &Value;
        }

        //disabled options would not be serialized/deserialized
        bool IsDisabled() const {
            return IsDisabledFlag;
        }

        bool IsDefault() const {
            return Value == DefaultValue;
        }

        void SetDisabledFlag(bool flag) {
            IsDisabledFlag = flag;
        }

        inline const TValue* operator->() const noexcept {
            return &Value;
        }

        bool IsSet() const {
            return IsSetFlag;
        }

        bool NotSet() const {
            return !IsSet();
        }

        const TString& GetName() const {
            return OptionName;
        }

        operator const TValue&() const {
            return Get();
        }

        operator TValue&() {
            return Get();
        }

        bool operator==(const TOption& rhs) const {
            return Value == rhs.Value && OptionName == rhs.OptionName;
        }

        bool operator!=(const TOption& rhs) const {
            return !(rhs == *this);
        }

        template <typename TComparableType>
        bool operator==(const TComparableType& otherValue) const {
            return Value == otherValue;
        }

        template <typename TComparableType>
        bool operator!=(const TComparableType& otherValue) const {
            return Value != otherValue;
        }

        inline TOption& operator=(const TValue& value) {
            Set(value);
            return *this;
        }

        inline TOption& operator=(const TOption& other) = default;

    private:
        template <class>
        friend struct ::THash;

        template <class, bool>
        friend class ::TJsonFieldHelper;

    private:
        TValue Value;
        TValue DefaultValue;
        TString OptionName;
        bool IsSetFlag = false;
        bool IsDisabledFlag = false;
    };
}

template <class T>
struct THash<NCatboostOptions::TOption<T>> {
    size_t operator()(const NCatboostOptions::TOption<T>& option) const noexcept {
        return THash<T>()(option.Value);
    }
};

template <class T>
inline IOutputStream& operator<<(IOutputStream& o, const NCatboostOptions::TOption<T>& option) {
    o << option.Get();
    return o;
}
