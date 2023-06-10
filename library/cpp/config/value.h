#pragma once

#include <typeinfo>

#include <util/generic/ptr.h>
#include <util/generic/cast.h>
#include <util/generic/string.h>
#include <util/generic/typetraits.h>

class IOutputStream;

namespace NConfig {
    class IValue: public TAtomicRefCount<IValue> {
    public:
        virtual ~IValue() = default;

        virtual bool IsA(const std::type_info& info) const = 0;
        virtual TString TypeName() const = 0;
        virtual void* Ptr() const = 0;

        virtual ui64 AsUInt() const = 0;
        virtual i64 AsInt() const = 0;
        virtual double AsDouble() const = 0;
        virtual bool AsBool() const = 0;
        virtual TString AsString() const = 0;

        virtual void ToJson(IOutputStream& out) const = 0;
    };

    namespace NCfgPrivate {
        struct TDummy {
        };

        template <class T>
        inline IValue* ConstructValueImpl(const T& t, ...) {
            extern IValue* ConstructValueImpl(const T& t);

            return ConstructValueImpl(t);
        }

        template <class T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
        inline IValue* ConstructValueImpl(const T& t, TDummy) {
            extern IValue* ConstructValueImpl(const double& t);

            return ConstructValueImpl(t);
        }

        template <class T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
        inline IValue* ConstructValueImpl(const T& t, TDummy) {
            typedef std::conditional_t<std::is_signed<T>::value, i64, ui64> Type;
            extern IValue* ConstructValueImpl(const Type& t);

            return ConstructValueImpl(t);
        }

        template <class T, std::enable_if_t<std::is_convertible<T, TString>::value>* = nullptr>
        inline IValue* ConstructValueImpl(const T& t, TDummy) {
            extern IValue* ConstructValueImpl(const TString& t);

            return ConstructValueImpl(t);
        }

        inline IValue* ConstructValueImpl(const bool& t, TDummy) {
            extern IValue* ConstructValueImpl(const bool& t);

            return ConstructValueImpl(t);
        }
    }

    template <class T>
    inline IValue* ConstructValue(const T& t) {
        return NCfgPrivate::ConstructValueImpl(t, NCfgPrivate::TDummy());
    }

    IValue* Null();

    namespace NCfgPrivate {
        template <bool Unsigned>
        struct TSelector {
            static inline ui64 Cvt(const IValue* v) {
                return v->AsUInt();
            }
        };

        template <>
        struct TSelector<false> {
            static inline i64 Cvt(const IValue* v) {
                return v->AsInt();
            }
        };

        [[noreturn]] void ReportTypeMismatch(TStringBuf realType, TStringBuf expectedType);
    }

    template <class T>
    inline T ValueAs(const IValue* val) {
        typedef NCfgPrivate::TSelector<std::is_unsigned<T>::value> TCvt;

        return SafeIntegerCast<T>(TCvt::Cvt(val));
    }

    template <>
    inline double ValueAs(const IValue* val) {
        return val->AsDouble();
    }

    template <>
    inline float ValueAs(const IValue* val) {
        return (float)val->AsDouble();
    }

    template <>
    inline bool ValueAs(const IValue* val) {
        return val->AsBool();
    }

    template <>
    inline TString ValueAs(const IValue* val) {
        return val->AsString();
    }
}
