#include "value.h"
#include "config.h"

#include <library/cpp/string_utils/relaxed_escaper/relaxed_escaper.h>

#include <util/generic/algorithm.h>
#include <util/system/type_name.h>
#include <util/generic/singleton.h>
#include <util/string/cast.h>
#include <util/string/strip.h>
#include <util/string/type.h>

using namespace NConfig;

namespace {
    template <class T>
    class TValue: public IValue {
    public:
        inline TValue(const T& t)
            : T_(t)
        {
        }

        bool IsA(const std::type_info& info) const override {
            return info == typeid(T);
        }

        TString TypeName() const override {
            return ::TypeName<T>();
        }

        void* Ptr() const override {
            return (void*)&T_;
        }

        void ToJson(IOutputStream& out) const override {
            out << AsString();
        }

        bool AsBool() const override {
            return (bool)AsDouble();
        }

    protected:
        T T_;
    };

    class TNullValue: public TValue<TNull> {
    public:
        inline TNullValue()
            : TValue<TNull>(TNull())
        {
            Ref();
        }

        double AsDouble() const override {
            return 0;
        }

        ui64 AsUInt() const override {
            return 0;
        }

        i64 AsInt() const override {
            return 0;
        }

        TString AsString() const override {
            return TString();
        }

        void ToJson(IOutputStream& out) const override {
            out << "null";
        }

        TString TypeName() const override {
            return "null";
        }
    };

    template <class T>
    class TNumericValue: public TValue<T> {
    public:
        inline TNumericValue(const T& t)
            : TValue<T>(t)
        {
        }

        double AsDouble() const override {
            return this->T_;
        }

        ui64 AsUInt() const override {
            return this->T_;
        }

        i64 AsInt() const override {
            return this->T_;
        }
    };

    class TBoolValue: public TNumericValue<bool> {
    public:
        inline TBoolValue(bool v)
            : TNumericValue<bool>(v)
        {
        }

        TString AsString() const override {
            return T_ ? "true" : "false";
        }
    };

    template <class T>
    class TArithmeticValue: public TNumericValue<T> {
    public:
        inline TArithmeticValue(T v)
            : TNumericValue<T>(v)
        {
        }

        TString AsString() const override {
            return ToString(this->T_);
        }
    };

    class TStringValue: public TValue<TString> {
    public:
        inline TStringValue(const TString& v)
            : TValue<TString>(v)
        {
        }

        template <class T>
        inline T AsT() const {
            const TStringBuf s = StripString(TStringBuf(T_));

            if (IsTrue(s)) {
                return true;
            }

            if (IsFalse(s)) {
                return false;
            }

            return FromString<T>(s);
        }

        double AsDouble() const override {
            return AsT<double>();
        }

        ui64 AsUInt() const override {
            return AsT<ui64>();
        }

        i64 AsInt() const override {
            return AsT<i64>();
        }

        TString AsString() const override {
            return T_;
        }

        void ToJson(IOutputStream& out) const override {
            NEscJ::EscapeJ<true, true>(T_, out);
        }

        TString TypeName() const override {
            return "string";
        }
    };

    template <class T>
    class TContainer: public TValue<T> {
    public:
        inline TContainer(const T& t)
            : TValue<T>(t)
        {
        }

        double AsDouble() const override {
            NCfgPrivate::ReportTypeMismatch(this->TypeName(), "double");
        }

        ui64 AsUInt() const override {
            NCfgPrivate::ReportTypeMismatch(this->TypeName(), "uint");
        }

        i64 AsInt() const override {
            NCfgPrivate::ReportTypeMismatch(this->TypeName(), "int");
        }

        bool AsBool() const override {
            NCfgPrivate::ReportTypeMismatch(this->TypeName(), "bool");
        }

        TString AsString() const override {
            NCfgPrivate::ReportTypeMismatch(this->TypeName(), "string");
        }
    };

    class TArrayValue: public TContainer<TArray> {
    public:
        inline TArrayValue(const TArray& v)
            : TContainer<TArray>(v)
        {
        }

        void ToJson(IOutputStream& s) const override {
            s << "[";

            for (TArray::const_iterator it = T_.begin(); it != T_.end(); ++it) {
                if (it != T_.begin()) {
                    s << ",";
                }

                it->ToJson(s);
            }

            s << "]";
        }

        TString TypeName() const override {
            return "array";
        }
    };

    class TDictValue: public TContainer<TDict> {
    public:
        inline TDictValue(const TDict& v)
            : TContainer<TDict>(v)
        {
        }

        void ToJson(IOutputStream& s) const override {
            s << "{";

            TVector<TStringBuf> buf;
            buf.reserve(T_.size());
            for (const auto& t : T_) {
                buf.push_back(t.first);
            }
            Sort(buf.begin(), buf.end());
            for (TVector<TStringBuf>::const_iterator kit = buf.begin(); kit != buf.end(); ++kit) {
                TStringBuf key = *kit;
                TDict::const_iterator it = T_.find(key);

                if (kit != buf.begin()) {
                    s << ",";
                }

                NEscJ::EscapeJ<true, true>(key, s);

                s << ":";

                it->second.ToJson(s);
            }

            s << "}";
        }

        TString TypeName() const override {
            return "dict";
        }
    };
}

#define DECLARE(type1, type2)                    \
    IValue* ConstructValueImpl(const type2& t) { \
        return new type1(t);                     \
    }

namespace NConfig {
    namespace NCfgPrivate {
        DECLARE(TBoolValue, bool)
        DECLARE(TArithmeticValue<double>, double)
        DECLARE(TArithmeticValue<i64>, i64)
        DECLARE(TArithmeticValue<ui64>, ui64)
        DECLARE(TStringValue, TString)
        DECLARE(TArrayValue, TArray)
        DECLARE(TDictValue, TDict)
    }

    IValue* Null() {
        return Singleton<TNullValue>();
    }

    [[noreturn]] void NCfgPrivate::ReportTypeMismatch(TStringBuf realType, TStringBuf expectedType) {
        ythrow TTypeMismatch() << "type mismatch (real: " << realType << ", expected: " << expectedType << ')';
    }
}
