#pragma once

#include <library/cpp/json/common/defs.h>

#include <util/generic/string.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/generic/deque.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>

namespace NJson {
    enum EJsonValueType {
        JSON_UNDEFINED /* "Undefined" */,
        JSON_NULL /* "Null" */,
        JSON_BOOLEAN /* "Boolean" */,
        JSON_INTEGER /* "Integer" */,
        JSON_DOUBLE /* "Double" */,
        JSON_STRING /* "String" */,
        JSON_MAP /* "Map" */,
        JSON_ARRAY /* "Array" */,
        JSON_UINTEGER /* "UInteger" */
    };

    class TJsonValue;

    class IScanCallback {
    public:
        virtual ~IScanCallback() = default;

        virtual bool Do(const TString& path, TJsonValue* parent, TJsonValue& value) = 0;
    };

    class TJsonValue {
        void Clear() noexcept;

    public:
        typedef THashMap<TString, TJsonValue> TMapType;
        typedef TDeque<TJsonValue> TArray;

        TJsonValue() noexcept = default;
        TJsonValue(EJsonValueType type);
        TJsonValue(bool value) noexcept;
        TJsonValue(int value) noexcept;
        TJsonValue(unsigned int value) noexcept;
        TJsonValue(long value) noexcept;
        TJsonValue(unsigned long value) noexcept;
        TJsonValue(long long value) noexcept;
        TJsonValue(unsigned long long value) noexcept;
        TJsonValue(double value) noexcept;
        TJsonValue(TString value);
        TJsonValue(const char* value);
        template <class T>
        TJsonValue(const T*) = delete;
        TJsonValue(TStringBuf value);

        TJsonValue(const std::string& s)
            : TJsonValue(TStringBuf(s))
        {
        }

        TJsonValue(const TJsonValue& vval);
        TJsonValue(TJsonValue&& vval) noexcept;

        TJsonValue& operator=(const TJsonValue& val);
        TJsonValue& operator=(TJsonValue&& val) noexcept;

        ~TJsonValue() {
            Clear();
        }

        EJsonValueType GetType() const noexcept;
        TJsonValue& SetType(EJsonValueType type);

        TJsonValue& SetValue(const TJsonValue& value);
        TJsonValue& SetValue(TJsonValue&& value);

        // for Map
        TJsonValue& InsertValue(const TString& key, const TJsonValue& value);
        TJsonValue& InsertValue(TStringBuf key, const TJsonValue& value);
        TJsonValue& InsertValue(const char* key, const TJsonValue& value);
        TJsonValue& InsertValue(const TString& key, TJsonValue&& value);
        TJsonValue& InsertValue(TStringBuf key, TJsonValue&& value);
        TJsonValue& InsertValue(const char* key, TJsonValue&& value);

        // for Array
        TJsonValue& AppendValue(const TJsonValue& value);
        TJsonValue& AppendValue(TJsonValue&& value);
        TJsonValue& Back();
        const TJsonValue& Back() const;

        bool GetValueByPath(TStringBuf path, TJsonValue& result, char delimiter = '.') const;
        bool SetValueByPath(TStringBuf path, const TJsonValue& value, char delimiter = '.');
        bool SetValueByPath(TStringBuf path, TJsonValue&& value, char delimiter = '.');

        // returns NULL on failure
        const TJsonValue* GetValueByPath(TStringBuf path, char delimiter = '.') const noexcept;
        TJsonValue* GetValueByPath(TStringBuf path, char delimiter = '.') noexcept;

        void EraseValue(TStringBuf key);
        void EraseValue(size_t index);

        TJsonValue& operator[](size_t idx);
        TJsonValue& operator[](const TStringBuf& key);
        const TJsonValue& operator[](size_t idx) const noexcept;
        const TJsonValue& operator[](const TStringBuf& key) const noexcept;

        bool GetBoolean() const;
        long long GetInteger() const;
        unsigned long long GetUInteger() const;
        double GetDouble() const;
        const TString& GetString() const;
        const TMapType& GetMap() const;
        const TArray& GetArray() const;

        //throwing TJsonException possible
        bool GetBooleanSafe() const;
        long long GetIntegerSafe() const;
        unsigned long long GetUIntegerSafe() const;
        double GetDoubleSafe() const;
        const TString& GetStringSafe() const;
        const TMapType& GetMapSafe() const;
        TMapType& GetMapSafe();
        const TArray& GetArraySafe() const;
        TArray& GetArraySafe();

        bool GetBooleanSafe(bool defaultValue) const;
        long long GetIntegerSafe(long long defaultValue) const;
        unsigned long long GetUIntegerSafe(unsigned long long defaultValue) const;
        double GetDoubleSafe(double defaultValue) const;
        TString GetStringSafe(const TString& defaultValue) const;

        bool GetBooleanRobust() const noexcept;
        long long GetIntegerRobust() const noexcept;
        unsigned long long GetUIntegerRobust() const noexcept;
        double GetDoubleRobust() const noexcept;
        TString GetStringRobust() const;

        // Exception-free accessors
        bool GetBoolean(bool* value) const noexcept;
        bool GetInteger(long long* value) const noexcept;
        bool GetUInteger(unsigned long long* value) const noexcept;
        bool GetDouble(double* value) const noexcept;
        bool GetMapPointer(const TMapType** value) const noexcept;
        bool GetArrayPointer(const TArray** value) const noexcept;

        bool GetString(TString* value) const;
        bool GetMap(TMapType* value) const;
        bool GetArray(TArray* value) const;
        bool GetValue(size_t index, TJsonValue* value) const;
        bool GetValue(TStringBuf key, TJsonValue* value) const;
        bool GetValuePointer(size_t index, const TJsonValue** value) const noexcept;
        bool GetValuePointer(TStringBuf key, const TJsonValue** value) const noexcept;
        bool GetValuePointer(TStringBuf key, TJsonValue** value) noexcept;

        // Checking for defined non-null value
        bool IsDefined() const noexcept {
            return Type != JSON_UNDEFINED && Type != JSON_NULL;
        }

        bool IsNull() const noexcept;
        bool IsBoolean() const noexcept;
        bool IsDouble() const noexcept;
        bool IsString() const noexcept;
        bool IsMap() const noexcept;
        bool IsArray() const noexcept;

        /// @return true if JSON_INTEGER or (JSON_UINTEGER and Value <= Max<long long>)
        bool IsInteger() const noexcept;

        /// @return true if JSON_UINTEGER or (JSON_INTEGER and Value >= 0)
        bool IsUInteger() const noexcept;

        bool Has(const TStringBuf& key) const noexcept;
        bool Has(size_t key) const noexcept;

        void Scan(IScanCallback& callback);

        /// Non-robust comparison.
        bool operator==(const TJsonValue& rhs) const;

        void Swap(TJsonValue& rhs) noexcept;

        // save using util/ysaveload.h serialization (not to JSON stream)
        void Save(IOutputStream* s) const;

        // load using util/ysaveload.h serialization (not as JSON stream)
        void Load(IInputStream* s);

        static const TJsonValue UNDEFINED;

    private:
        EJsonValueType Type = JSON_UNDEFINED;
        union TValueUnion {
            bool Boolean;
            long long Integer;
            unsigned long long UInteger;
            double Double;
            TString String;
            TMapType* Map;
            TArray* Array;

            TValueUnion() noexcept {
                Zero(*this);
            }
            ~TValueUnion() noexcept {
            }
        };
        TValueUnion Value;
        void DoScan(const TString& path, TJsonValue* parent, IScanCallback& callback);
        void SwapWithUndefined(TJsonValue& output) noexcept;

        /**
            @throw yexception if Back shouldn't be called on the object.
         */
        void BackChecks() const;
    };

    inline bool GetBoolean(const TJsonValue& jv, size_t index, bool* value) noexcept {
        return jv[index].GetBoolean(value);
    }

    inline bool GetInteger(const TJsonValue& jv, size_t index, long long* value) noexcept {
        return jv[index].GetInteger(value);
    }

    inline bool GetUInteger(const TJsonValue& jv, size_t index, unsigned long long* value) noexcept {
        return jv[index].GetUInteger(value);
    }

    inline bool GetDouble(const TJsonValue& jv, size_t index, double* value) noexcept {
        return jv[index].GetDouble(value);
    }

    inline bool GetString(const TJsonValue& jv, size_t index, TString* value) {
        return jv[index].GetString(value);
    }

    bool GetMapPointer(const TJsonValue& jv, size_t index, const TJsonValue::TMapType** value);
    bool GetArrayPointer(const TJsonValue& jv, size_t index, const TJsonValue::TArray** value);

    inline bool GetBoolean(const TJsonValue& jv, TStringBuf key, bool* value) noexcept {
        return jv[key].GetBoolean(value);
    }

    inline bool GetInteger(const TJsonValue& jv, TStringBuf key, long long* value) noexcept {
        return jv[key].GetInteger(value);
    }

    inline bool GetUInteger(const TJsonValue& jv, TStringBuf key, unsigned long long* value) noexcept {
        return jv[key].GetUInteger(value);
    }

    inline bool GetDouble(const TJsonValue& jv, TStringBuf key, double* value) noexcept {
        return jv[key].GetDouble(value);
    }

    inline bool GetString(const TJsonValue& jv, TStringBuf key, TString* value) {
        return jv[key].GetString(value);
    }

    bool GetMapPointer(const TJsonValue& jv, const TStringBuf key, const TJsonValue::TMapType** value);
    bool GetArrayPointer(const TJsonValue& jv, const TStringBuf key, const TJsonValue::TArray** value);

    class TJsonMap: public TJsonValue {
    public:
        TJsonMap()
            : TJsonValue(NJson::JSON_MAP)
        {}

        TJsonMap(const std::initializer_list<std::pair<TString, TJsonValue>>& list)
            : TJsonValue(NJson::JSON_MAP)
        {
            GetMapSafe() = THashMap<TString, TJsonValue>(list);
        }
    };

    class TJsonArray: public TJsonValue {
    public:
        TJsonArray()
            : TJsonValue(NJson::JSON_ARRAY)
        {}

        TJsonArray(const std::initializer_list<TJsonValue>& list)
            : TJsonValue(NJson::JSON_ARRAY)
        {
            GetArraySafe() = TJsonValue::TArray(list);
        }
    };
}
