#include "json_value.h"
#include "json.h"

#include <util/generic/ymath.h>
#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/singleton.h>
#include <util/stream/str.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/string/type.h>
#include <util/string/vector.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>
#include <util/generic/yexception.h>

static bool
AreJsonMapsEqual(const NJson::TJsonValue& lhs, const NJson::TJsonValue& rhs) {
    using namespace NJson;

    Y_ABORT_UNLESS(lhs.GetType() == JSON_MAP, "lhs has not a JSON_MAP type.");

    if (rhs.GetType() != JSON_MAP)
        return false;

    typedef TJsonValue::TMapType TMapType;
    const TMapType& lhsMap = lhs.GetMap();
    const TMapType& rhsMap = rhs.GetMap();

    if (lhsMap.size() != rhsMap.size())
        return false;

    for (const auto& lhsIt : lhsMap) {
        TMapType::const_iterator rhsIt = rhsMap.find(lhsIt.first);
        if (rhsIt == rhsMap.end())
            return false;

        if (lhsIt.second != rhsIt->second)
            return false;
    }

    return true;
}

static bool
AreJsonArraysEqual(const NJson::TJsonValue& lhs, const NJson::TJsonValue& rhs) {
    using namespace NJson;

    Y_ABORT_UNLESS(lhs.GetType() == JSON_ARRAY, "lhs has not a JSON_ARRAY type.");

    if (rhs.GetType() != JSON_ARRAY)
        return false;

    typedef TJsonValue::TArray TArray;
    const TArray& lhsArray = lhs.GetArray();
    const TArray& rhsArray = rhs.GetArray();

    if (lhsArray.size() != rhsArray.size())
        return false;

    for (TArray::const_iterator lhsIt = lhsArray.begin(), rhsIt = rhsArray.begin();
         lhsIt != lhsArray.end(); ++lhsIt, ++rhsIt) {
        if (*lhsIt != *rhsIt)
            return false;
    }

    return true;
}

namespace NJson {
    const TJsonValue TJsonValue::UNDEFINED{};

    TJsonValue::TJsonValue(const EJsonValueType type) {
        SetType(type);
    }

    TJsonValue::TJsonValue(TJsonValue&& vval) noexcept
        : Type(JSON_UNDEFINED)
    {
        vval.SwapWithUndefined(*this);
        Zero(vval.Value);
    }

    TJsonValue::TJsonValue(const TJsonValue& val)
        : Type(val.Type)
    {
        switch (Type) {
            case JSON_STRING:
                new (&Value.String) TString(val.GetString());
                break;
            case JSON_MAP:
                Value.Map = new TMapType(val.GetMap());
                break;
            case JSON_ARRAY:
                Value.Array = new TArray(val.GetArray());
                break;
            case JSON_UNDEFINED:
            case JSON_NULL:
            case JSON_BOOLEAN:
            case JSON_INTEGER:
            case JSON_UINTEGER:
            case JSON_DOUBLE:
                std::memcpy(&Value, &val.Value, sizeof(Value));
                break;
        }
    }

    TJsonValue& TJsonValue::operator=(const TJsonValue& val) {
        if (this == &val)
            return *this;
        TJsonValue tmp(val);
        tmp.Swap(*this);
        return *this;
    }

    TJsonValue& TJsonValue::operator=(TJsonValue&& val) noexcept {
        if (this == &val)
            return *this;
        TJsonValue tmp(std::move(val));
        tmp.Swap(*this);
        return *this;
    }

    TJsonValue::TJsonValue(const bool value) noexcept {
        SetType(JSON_BOOLEAN);
        Value.Boolean = value;
    }

    TJsonValue::TJsonValue(const long long value) noexcept {
        SetType(JSON_INTEGER);
        Value.Integer = value;
    }

    TJsonValue::TJsonValue(const unsigned long long value) noexcept {
        SetType(JSON_UINTEGER);
        Value.UInteger = value;
    }

    TJsonValue::TJsonValue(const int value) noexcept {
        SetType(JSON_INTEGER);
        Value.Integer = value;
    }

    TJsonValue::TJsonValue(const unsigned int value) noexcept {
        SetType(JSON_UINTEGER);
        Value.UInteger = value;
    }

    TJsonValue::TJsonValue(const long value) noexcept {
        SetType(JSON_INTEGER);
        Value.Integer = value;
    }

    TJsonValue::TJsonValue(const unsigned long value) noexcept {
        SetType(JSON_UINTEGER);
        Value.UInteger = value;
    }

    TJsonValue::TJsonValue(const double value) noexcept {
        SetType(JSON_DOUBLE);
        Value.Double = value;
    }

    TJsonValue::TJsonValue(TString value) {
        SetType(JSON_STRING);
        Value.String = std::move(value);
    }

    TJsonValue::TJsonValue(const TStringBuf value) {
        SetType(JSON_STRING);
        Value.String = value;
    }

    TJsonValue::TJsonValue(const char* value) {
        SetType(JSON_STRING);
        Value.String = value;
    }

    EJsonValueType TJsonValue::GetType() const noexcept {
        return Type;
    }

    TJsonValue& TJsonValue::SetType(const EJsonValueType type) {
        if (Type == type)
            return *this;

        Clear();
        Type = type;

        switch (Type) {
            case JSON_STRING:
                new (&Value.String) TString();
                break;
            case JSON_MAP:
                Value.Map = new TMapType();
                break;
            case JSON_ARRAY:
                Value.Array = new TArray();
                break;
            case JSON_UNDEFINED:
            case JSON_NULL:
            case JSON_BOOLEAN:
            case JSON_INTEGER:
            case JSON_UINTEGER:
            case JSON_DOUBLE:
                break;
        }

        return *this;
    }

    TJsonValue& TJsonValue::SetValue(const TJsonValue& value) {
        return *this = value;
    }

    TJsonValue& TJsonValue::SetValue(TJsonValue&& value) {
        *this = std::move(value);
        return *this;
    }

    TJsonValue& TJsonValue::InsertValue(const TString& key, const TJsonValue& value) {
        SetType(JSON_MAP);
        return (*Value.Map)[key] = value;
    }

    TJsonValue& TJsonValue::InsertValue(const TStringBuf key, const TJsonValue& value) {
        SetType(JSON_MAP);
        return (*Value.Map)[key] = value;
    }

    TJsonValue& TJsonValue::InsertValue(const char* key, const TJsonValue& value) {
        SetType(JSON_MAP);
        return (*Value.Map)[key] = value;
    }

    TJsonValue& TJsonValue::InsertValue(const TString& key, TJsonValue&& value) {
        SetType(JSON_MAP);
        return (*Value.Map)[key] = std::move(value);
    }

    TJsonValue& TJsonValue::InsertValue(const TStringBuf key, TJsonValue&& value) {
        SetType(JSON_MAP);
        return (*Value.Map)[key] = std::move(value);
    }

    TJsonValue& TJsonValue::InsertValue(const char* key, TJsonValue&& value) {
        SetType(JSON_MAP);
        return (*Value.Map)[key] = std::move(value);
    }

    TJsonValue& TJsonValue::Back() {
        BackChecks();
        return Value.Array->back();
    }

    const TJsonValue& TJsonValue::Back() const {
        BackChecks();
        return Value.Array->back();
    }

    TJsonValue& TJsonValue::AppendValue(const TJsonValue& value) {
        SetType(JSON_ARRAY);
        Value.Array->push_back(value);
        return Value.Array->back();
    }

    TJsonValue& TJsonValue::AppendValue(TJsonValue&& value) {
        SetType(JSON_ARRAY);
        Value.Array->push_back(std::move(value));
        return Value.Array->back();
    }

    void TJsonValue::EraseValue(const TStringBuf key) {
        if (IsMap()) {
            TMapType::iterator it = Value.Map->find(key);
            if (it != Value.Map->end())
                Value.Map->erase(it);
        }
    }

    void TJsonValue::EraseValue(const size_t index) {
        if (IsArray()) {
            if (index >= Value.Array->size()) {
                return;
            }
            TArray::iterator it = Value.Array->begin() + index;
            Value.Array->erase(it);
        }
    }

    void TJsonValue::Clear() noexcept {
        switch (Type) {
            case JSON_STRING:
                Value.String.~TString();
                break;
            case JSON_MAP:
                delete Value.Map;
                break;
            case JSON_ARRAY:
                delete Value.Array;
                break;
            case JSON_UNDEFINED:
            case JSON_NULL:
            case JSON_BOOLEAN:
            case JSON_INTEGER:
            case JSON_UINTEGER:
            case JSON_DOUBLE:
                break;
        }
        Zero(Value);
        Type = JSON_UNDEFINED;
    }

    TJsonValue& TJsonValue::operator[](const size_t idx) {
        SetType(JSON_ARRAY);
        if (Value.Array->size() <= idx)
            Value.Array->resize(idx + 1);
        return (*Value.Array)[idx];
    }

    TJsonValue& TJsonValue::operator[](const TStringBuf& key) {
        SetType(JSON_MAP);
        return (*Value.Map)[key];
    }

    namespace {
        struct TDefaultsHolder {
            const TString String{};
            const TJsonValue::TMapType Map{};
            const TJsonValue::TArray Array{};
            const TJsonValue Value{};
        };
    }

    const TJsonValue& TJsonValue::operator[](const size_t idx) const noexcept {
        const TJsonValue* ret = nullptr;
        if (GetValuePointer(idx, &ret))
            return *ret;

        return Singleton<TDefaultsHolder>()->Value;
    }

    const TJsonValue& TJsonValue::operator[](const TStringBuf& key) const noexcept {
        const TJsonValue* ret = nullptr;
        if (GetValuePointer(key, &ret))
            return *ret;

        return Singleton<TDefaultsHolder>()->Value;
    }

    bool TJsonValue::GetBoolean() const {
        return Type != JSON_BOOLEAN ? false : Value.Boolean;
    }

    long long TJsonValue::GetInteger() const {
        if (!IsInteger())
            return 0;

        switch (Type) {
            case JSON_INTEGER:
                return Value.Integer;

            case JSON_UINTEGER:
                return Value.UInteger;

            case JSON_DOUBLE:
                return Value.Double;

            default:
                Y_ASSERT(false && "Unexpected type.");
                return 0;
        }
    }

    unsigned long long TJsonValue::GetUInteger() const {
        if (!IsUInteger())
            return 0;

        switch (Type) {
            case JSON_UINTEGER:
                return Value.UInteger;

            case JSON_INTEGER:
                return Value.Integer;

            case JSON_DOUBLE:
                return Value.Double;

            default:
                Y_ASSERT(false && "Unexpected type.");
                return 0;
        }
    }

    double TJsonValue::GetDouble() const {
        if (!IsDouble())
            return 0.0;

        switch (Type) {
            case JSON_DOUBLE:
                return Value.Double;

            case JSON_INTEGER:
                return Value.Integer;

            case JSON_UINTEGER:
                return Value.UInteger;

            default:
                Y_ASSERT(false && "Unexpected type.");
                return 0.0;
        }
    }

    const TString& TJsonValue::GetString() const {
        return Type != JSON_STRING ? Singleton<TDefaultsHolder>()->String : Value.String;
    }

    const TJsonValue::TMapType& TJsonValue::GetMap() const {
        return Type != JSON_MAP ? Singleton<TDefaultsHolder>()->Map : *Value.Map;
    }

    const TJsonValue::TArray& TJsonValue::GetArray() const {
        return (Type != JSON_ARRAY) ? Singleton<TDefaultsHolder>()->Array : *Value.Array;
    }

    bool TJsonValue::GetBooleanSafe() const {
        if (Type != JSON_BOOLEAN)
            ythrow TJsonException() << "Not a boolean";

        return Value.Boolean;
    }

    long long TJsonValue::GetIntegerSafe() const {
        if (!IsInteger())
            ythrow TJsonException() << "Not an integer";

        return GetInteger();
    }

    unsigned long long TJsonValue::GetUIntegerSafe() const {
        if (!IsUInteger())
            ythrow TJsonException() << "Not an unsigned integer";

        return GetUInteger();
    }

    double TJsonValue::GetDoubleSafe() const {
        if (!IsDouble())
            ythrow TJsonException() << "Not a double";

        return GetDouble();
    }

    const TString& TJsonValue::GetStringSafe() const {
        if (Type != JSON_STRING)
            ythrow TJsonException() << "Not a string";

        return Value.String;
    }

    bool TJsonValue::GetBooleanSafe(const bool defaultValue) const {
        if (Type == JSON_UNDEFINED)
            return defaultValue;

        return GetBooleanSafe();
    }

    long long TJsonValue::GetIntegerSafe(const long long defaultValue) const {
        if (Type == JSON_UNDEFINED)
            return defaultValue;

        return GetIntegerSafe();
    }

    unsigned long long TJsonValue::GetUIntegerSafe(const unsigned long long defaultValue) const {
        if (Type == JSON_UNDEFINED)
            return defaultValue;

        return GetUIntegerSafe();
    }

    double TJsonValue::GetDoubleSafe(const double defaultValue) const {
        if (Type == JSON_UNDEFINED)
            return defaultValue;

        return GetDoubleSafe();
    }

    TString TJsonValue::GetStringSafe(const TString& defaultValue) const {
        if (Type == JSON_UNDEFINED)
            return defaultValue;

        return GetStringSafe();
    }

    const TJsonValue::TMapType& TJsonValue::GetMapSafe() const {
        if (Type != JSON_MAP)
            ythrow TJsonException() << "Not a map";

        return *Value.Map;
    }

    TJsonValue::TMapType& TJsonValue::GetMapSafe() {
        return const_cast<TJsonValue::TMapType&>(const_cast<const TJsonValue*>(this)->GetMapSafe());
    }

    const TJsonValue::TArray& TJsonValue::GetArraySafe() const {
        if (Type != JSON_ARRAY)
            ythrow TJsonException() << "Not an array";

        return *Value.Array;
    }

    TJsonValue::TArray& TJsonValue::GetArraySafe() {
        return const_cast<TJsonValue::TArray&>(const_cast<const TJsonValue*>(this)->GetArraySafe());
    }

    bool TJsonValue::GetBooleanRobust() const noexcept {
        switch (Type) {
            case JSON_ARRAY:
                return !Value.Array->empty();
            case JSON_MAP:
                return !Value.Map->empty();
            case JSON_INTEGER:
            case JSON_UINTEGER:
            case JSON_DOUBLE:
                return GetIntegerRobust();
            case JSON_STRING:
                return GetIntegerRobust() || IsTrue(Value.String);
            case JSON_NULL:
            case JSON_UNDEFINED:
            default:
                return false;
            case JSON_BOOLEAN:
                return Value.Boolean;
        }
    }

    long long TJsonValue::GetIntegerRobust() const noexcept {
        switch (Type) {
            case JSON_ARRAY:
                return Value.Array->size();
            case JSON_MAP:
                return Value.Map->size();
            case JSON_BOOLEAN:
                return Value.Boolean;
            case JSON_DOUBLE:
                return GetDoubleRobust();
            case JSON_STRING:
                try {
                    i64 res = 0;
                    if (Value.String && TryFromString(Value.String, res)) {
                        return res;
                    }
                } catch (const yexception&) {
                }
                return 0;
            case JSON_NULL:
            case JSON_UNDEFINED:
            default:
                return 0;
            case JSON_INTEGER:
            case JSON_UINTEGER:
                return Value.Integer;
        }
    }

    unsigned long long TJsonValue::GetUIntegerRobust() const noexcept {
        switch (Type) {
            case JSON_ARRAY:
                return Value.Array->size();
            case JSON_MAP:
                return Value.Map->size();
            case JSON_BOOLEAN:
                return Value.Boolean;
            case JSON_DOUBLE:
                return GetDoubleRobust();
            case JSON_STRING:
                try {
                    ui64 res = 0;
                    if (Value.String && TryFromString(Value.String, res)) {
                        return res;
                    }
                } catch (const yexception&) {
                }
                return 0;
            case JSON_NULL:
            case JSON_UNDEFINED:
            default:
                return 0;
            case JSON_INTEGER:
            case JSON_UINTEGER:
                return Value.UInteger;
        }
    }

    double TJsonValue::GetDoubleRobust() const noexcept {
        switch (Type) {
            case JSON_ARRAY:
                return Value.Array->size();
            case JSON_MAP:
                return Value.Map->size();
            case JSON_BOOLEAN:
                return Value.Boolean;
            case JSON_INTEGER:
                return Value.Integer;
            case JSON_UINTEGER:
                return Value.UInteger;
            case JSON_STRING:
                try {
                    double res = 0;
                    if (Value.String && TryFromString(Value.String, res)) {
                        return res;
                    }
                } catch (const yexception&) {
                }
                return 0;
            case JSON_NULL:
            case JSON_UNDEFINED:
            default:
                return 0;
            case JSON_DOUBLE:
                return Value.Double;
        }
    }

    TString TJsonValue::GetStringRobust() const {
        switch (Type) {
            case JSON_ARRAY:
            case JSON_MAP:
            case JSON_BOOLEAN:
            case JSON_DOUBLE:
            case JSON_INTEGER:
            case JSON_UINTEGER:
            case JSON_NULL:
            case JSON_UNDEFINED:
            default: {
                NJsonWriter::TBuf sout;
                sout.WriteJsonValue(this);
                return sout.Str();
            }
            case JSON_STRING:
                return Value.String;
        }
    }

    bool TJsonValue::GetBoolean(bool* value) const noexcept {
        if (Type != JSON_BOOLEAN)
            return false;

        *value = Value.Boolean;
        return true;
    }

    bool TJsonValue::GetInteger(long long* value) const noexcept {
        if (!IsInteger())
            return false;

        *value = GetInteger();
        return true;
    }

    bool TJsonValue::GetUInteger(unsigned long long* value) const noexcept {
        if (!IsUInteger())
            return false;

        *value = GetUInteger();
        return true;
    }

    bool TJsonValue::GetDouble(double* value) const noexcept {
        if (!IsDouble())
            return false;

        *value = GetDouble();
        return true;
    }

    bool TJsonValue::GetString(TString* value) const {
        if (Type != JSON_STRING)
            return false;

        *value = Value.String;
        return true;
    }

    bool TJsonValue::GetMap(TJsonValue::TMapType* value) const {
        if (Type != JSON_MAP)
            return false;

        *value = *Value.Map;
        return true;
    }

    bool TJsonValue::GetArray(TJsonValue::TArray* value) const {
        if (Type != JSON_ARRAY)
            return false;

        *value = *Value.Array;
        return true;
    }

    bool TJsonValue::GetMapPointer(const TJsonValue::TMapType** value) const noexcept {
        if (Type != JSON_MAP)
            return false;

        *value = Value.Map;
        return true;
    }

    bool TJsonValue::GetArrayPointer(const TJsonValue::TArray** value) const noexcept {
        if (Type != JSON_ARRAY)
            return false;

        *value = Value.Array;
        return true;
    }

    bool TJsonValue::GetValue(const size_t index, TJsonValue* value) const {
        const TJsonValue* tmp = nullptr;
        if (GetValuePointer(index, &tmp)) {
            *value = *tmp;
            return true;
        }
        return false;
    }

    bool TJsonValue::GetValue(const TStringBuf key, TJsonValue* value) const {
        const TJsonValue* tmp = nullptr;
        if (GetValuePointer(key, &tmp)) {
            *value = *tmp;
            return true;
        }
        return false;
    }

    bool TJsonValue::GetValuePointer(const size_t index, const TJsonValue** value) const noexcept {
        if (Type == JSON_ARRAY && index < Value.Array->size()) {
            *value = &(*Value.Array)[index];
            return true;
        }
        return false;
    }

    bool TJsonValue::GetValuePointer(const TStringBuf key, const TJsonValue** value) const noexcept {
        if (Type == JSON_MAP) {
            const TMapType::const_iterator it = Value.Map->find(key);
            if (it != Value.Map->end()) {
                *value = &(it->second);
                return true;
            }
        }
        return false;
    }

    bool TJsonValue::GetValuePointer(const TStringBuf key, TJsonValue** value) noexcept {
        return static_cast<const TJsonValue*>(this)->GetValuePointer(key, const_cast<const TJsonValue**>(value));
    }

    bool TJsonValue::IsNull() const noexcept {
        return Type == JSON_NULL;
    }

    bool TJsonValue::IsBoolean() const noexcept {
        return Type == JSON_BOOLEAN;
    }

    bool TJsonValue::IsInteger() const noexcept {
        switch (Type) {
            case JSON_INTEGER:
                return true;

            case JSON_UINTEGER:
                return (Value.UInteger <= static_cast<unsigned long long>(Max<long long>()));

            case JSON_DOUBLE:
                return ((long long)Value.Double == Value.Double);

            default:
                return false;
        }
    }

    bool TJsonValue::IsUInteger() const noexcept {
        switch (Type) {
            case JSON_UINTEGER:
                return true;

            case JSON_INTEGER:
                return (Value.Integer >= 0);

            case JSON_DOUBLE:
                return ((unsigned long long)Value.Double == Value.Double);

            default:
                return false;
        }
    }

    bool TJsonValue::IsDouble() const noexcept {
        // Check whether we can convert integer to floating-point
        // without precision loss.
        switch (Type) {
            case JSON_DOUBLE:
                return true;

            case JSON_INTEGER:
                return (1ll << std::numeric_limits<double>::digits) >= Abs(Value.Integer);

            case JSON_UINTEGER:
                return (1ull << std::numeric_limits<double>::digits) >= Value.UInteger;

            default:
                return false;
        }
    }

    namespace {
        template <class TPtr, class T>
        TPtr* CreateOrNullptr(TPtr* p, T key, std::true_type /*create*/) {
            return &(*p)[key];
        }

        template <class TPtr, class T>
        TPtr* CreateOrNullptr(const TPtr* p, T key, std::false_type /*create*/) noexcept {
            const TPtr* const next = &(*p)[key];
            return next->IsDefined() ? const_cast<TPtr*>(next) : nullptr;
        }

        template <bool Create, class TJsonPtr>
        TJsonPtr GetValuePtrByPath(TJsonPtr currentJson, TStringBuf path, char delimiter) noexcept(!Create) {
            static_assert(
                !(Create && std::is_const<std::remove_pointer_t<TJsonPtr>>::value),
                "TJsonPtr must be a `TJsonValue*` if `Create` is true");
            constexpr std::integral_constant<bool, Create> create_tag{};

            while (!path.empty()) {
                size_t index = 0;
                const TStringBuf step = path.NextTok(delimiter);
                if (step.size() > 2 && *step.begin() == '[' && step.back() == ']' && TryFromString(step.substr(1, step.size() - 2), index)) {
                    currentJson = CreateOrNullptr(currentJson, index, create_tag);
                } else {
                    currentJson = CreateOrNullptr(currentJson, step, create_tag);
                }

                if (!currentJson) {
                    return nullptr;
                }
            }

            return currentJson;
        }
    } // anonymous namespace

    bool TJsonValue::GetValueByPath(const TStringBuf path, TJsonValue& result, char delimiter) const {
        const TJsonValue* const ptr = GetValuePtrByPath<false>(this, path, delimiter);
        if (ptr) {
            result = *ptr;
            return true;
        }
        return false;
    }

    bool TJsonValue::SetValueByPath(const TStringBuf path, const TJsonValue& value, char delimiter) {
        TJsonValue* const ptr = GetValuePtrByPath<true>(this, path, delimiter);
        if (ptr) {
            *ptr = value;
            return true;
        }
        return false;
    }

    bool TJsonValue::SetValueByPath(const TStringBuf path, TJsonValue&& value, char delimiter) {
        TJsonValue* const ptr = GetValuePtrByPath<true>(this, path, delimiter);
        if (ptr) {
            *ptr = std::move(value);
            return true;
        }
        return false;
    }

    const TJsonValue* TJsonValue::GetValueByPath(const TStringBuf key, char delim) const noexcept {
        return GetValuePtrByPath<false>(this, key, delim);
    }

    TJsonValue* TJsonValue::GetValueByPath(const TStringBuf key, char delim) noexcept {
        return GetValuePtrByPath<false>(this, key, delim);
    }

    void TJsonValue::DoScan(const TString& path, TJsonValue* parent, IScanCallback& callback) {
        if (!callback.Do(path, parent, *this)) {
            return;
        }

        if (Type == JSON_MAP) {
            for (auto&& i : *Value.Map) {
                i.second.DoScan(!!path ? TString::Join(path, ".", i.first) : i.first, this, callback);
            }
        } else if (Type == JSON_ARRAY) {
            for (ui32 i = 0; i < Value.Array->size(); ++i) {
                (*Value.Array)[i].DoScan(TString::Join(path, "[", ToString(i), "]"), this, callback);
            }
        }
    }

    void TJsonValue::Scan(IScanCallback& callback) {
        DoScan("", nullptr, callback);
    }

    bool TJsonValue::IsString() const noexcept {
        return Type == JSON_STRING;
    }

    bool TJsonValue::IsMap() const noexcept {
        return Type == JSON_MAP;
    }

    bool TJsonValue::IsArray() const noexcept {
        return Type == JSON_ARRAY;
    }

    bool TJsonValue::Has(const TStringBuf& key) const noexcept {
        return Type == JSON_MAP && Value.Map->contains(key);
    }

    bool TJsonValue::Has(size_t key) const noexcept {
        return Type == JSON_ARRAY && Value.Array->size() > key;
    }

    bool TJsonValue::operator==(const TJsonValue& rhs) const {
        switch (Type) {
            case JSON_UNDEFINED: {
                return (rhs.GetType() == JSON_UNDEFINED);
            }

            case JSON_NULL: {
                return rhs.IsNull();
            }

            case JSON_BOOLEAN: {
                return (rhs.IsBoolean() && Value.Boolean == rhs.Value.Boolean);
            }

            case JSON_INTEGER: {
                return (rhs.IsInteger() && GetInteger() == rhs.GetInteger());
            }

            case JSON_UINTEGER: {
                return (rhs.IsUInteger() && GetUInteger() == rhs.GetUInteger());
            }

            case JSON_STRING: {
                return (rhs.IsString() && Value.String == rhs.Value.String);
            }

            case JSON_DOUBLE: {
                return (rhs.IsDouble() && fabs(GetDouble() - rhs.GetDouble()) <= FLT_EPSILON);
            }

            case JSON_MAP:
                return AreJsonMapsEqual(*this, rhs);

            case JSON_ARRAY:
                return AreJsonArraysEqual(*this, rhs);

            default:
                Y_ASSERT(false && "Unknown type.");
                return false;
        }
    }

    void TJsonValue::SwapWithUndefined(TJsonValue& output) noexcept {
        if (Type == JSON_STRING) {
            static_assert(std::is_nothrow_move_constructible<TString>::value, "noexcept violation! Add some try {} catch (...) logic");
            new (&output.Value.String) TString(std::move(Value.String));
            Value.String.~TString();
        } else {
            std::memcpy(&output.Value, &Value, sizeof(Value));
        }

        output.Type = Type;
        Type = JSON_UNDEFINED;
    }

    void TJsonValue::Swap(TJsonValue& rhs) noexcept {
        TJsonValue tmp(std::move(*this));
        rhs.SwapWithUndefined(*this);
        tmp.SwapWithUndefined(rhs);
    }

    void TJsonValue::Save(IOutputStream* s) const {
        ::Save(s, static_cast<ui8>(Type));
        switch (Type) {
            case JSON_UNDEFINED:break;
            case JSON_NULL:break;
            case JSON_BOOLEAN:
                ::Save(s, Value.Boolean);
                break;
            case JSON_INTEGER:
                ::Save(s, Value.Integer);
                break;
            case JSON_UINTEGER:
                ::Save(s, Value.UInteger);
                break;
            case JSON_DOUBLE:
                ::Save(s, Value.Double);
                break;
            case JSON_STRING:
                ::Save(s, Value.String);
                break;
            case JSON_MAP:
                ::Save(s, *Value.Map);
                break;
            case JSON_ARRAY:
                ::Save(s, *Value.Array);
                break;
        }
    }

    void TJsonValue::Load(IInputStream* s) {
        {
            ui8 loadedType = {};
            ::Load(s, loadedType);
            SetType(static_cast<EJsonValueType>(loadedType));
        }
        switch (Type) {
            case JSON_UNDEFINED:break;
            case JSON_NULL:break;
            case JSON_BOOLEAN:
                ::Load(s, Value.Boolean);
                break;
            case JSON_INTEGER:
                ::Load(s, Value.Integer);
                break;
            case JSON_UINTEGER:
                ::Load(s, Value.UInteger);
                break;
            case JSON_DOUBLE:
                ::Load(s, Value.Double);
                break;
            case JSON_STRING:
                ::Load(s, Value.String);
                break;
            case JSON_MAP:
                ::Load(s, *Value.Map);
                break;
            case JSON_ARRAY:
                ::Load(s, *Value.Array);
                break;
        }
    }

    //****************************************************************

    bool GetMapPointer(const TJsonValue& jv, const size_t index, const TJsonValue::TMapType** value) {
        const TJsonValue* v;
        if (!jv.GetValuePointer(index, &v) || !v->IsMap())
            return false;

        *value = &v->GetMap();
        return true;
    }

    bool GetArrayPointer(const TJsonValue& jv, const size_t index, const TJsonValue::TArray** value) {
        const TJsonValue* v;
        if (!jv.GetValuePointer(index, &v) || !v->IsArray())
            return false;

        *value = &v->GetArray();
        return true;
    }

    bool GetMapPointer(const TJsonValue& jv, const TStringBuf key, const TJsonValue::TMapType** value) {
        const TJsonValue* v;
        if (!jv.GetValuePointer(key, &v) || !v->IsMap())
            return false;

        *value = &v->GetMap();
        return true;
    }

    bool GetArrayPointer(const TJsonValue& jv, const TStringBuf key, const TJsonValue::TArray** value) {
        const TJsonValue* v;
        if (!jv.GetValuePointer(key, &v) || !v->IsArray())
            return false;

        *value = &v->GetArray();
        return true;
    }

    void TJsonValue::BackChecks() const {
        if (Type != JSON_ARRAY)
            ythrow TJsonException() << "Not an array";

        if (Value.Array->empty())
            ythrow TJsonException() << "Get back on empty array";
    }
}

template <>
void Out<NJson::TJsonValue>(IOutputStream& out, const NJson::TJsonValue& v) {
    NJsonWriter::TBuf buf(NJsonWriter::HEM_DONT_ESCAPE_HTML, &out);
    buf.WriteJsonValue(&v);
}
