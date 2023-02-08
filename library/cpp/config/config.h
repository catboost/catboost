#pragma once

#include "fwd.h"
#include "value.h"

#include <library/cpp/json/json_value.h>

#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/deque.h>
#include <util/system/type_name.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/generic/bt_exception.h>
#include <util/ysaveload.h>

class IInputStream;
class IOutputStream;

namespace NConfig {
    typedef THashMap<TString, NJson::TJsonValue> TGlobals;

    class TConfigError: public TWithBackTrace<yexception> {
    };

    class TConfigParseError: public TConfigError {
    };

    class TTypeMismatch: public TConfigError {
    };

    struct TArray;
    struct TDict;

    class TConfig {
    public:
        inline TConfig()
            : V_(Null())
        {
        }

        inline TConfig(IValue* v)
            : V_(v)
        {
        }

        TConfig(const TConfig& config) = default;
        TConfig& operator=(const TConfig& config) = default;

        template <class T>
        inline bool IsA() const {
            return V_->IsA(typeid(T));
        }

        inline bool IsNumeric() const {
            return IsA<double>() || IsA<i64>() || IsA<ui64>();
        }

        template <class T>
        inline const T& Get() const {
            return GetNonConstant<T>();
        }

        template <class T>
        inline T& GetNonConstant() const {
            if (this->IsA<T>()) {
                return *(T*)V_->Ptr();
            }

            if constexpr (std::is_same_v<T, ::NConfig::TArray>) {
                NCfgPrivate::ReportTypeMismatch(V_->TypeName(), "array");
            } else if constexpr (std::is_same_v<T, ::NConfig::TDict>) {
                NCfgPrivate::ReportTypeMismatch(V_->TypeName(), "dict");
            } else if constexpr (std::is_same_v<T, TString>) {
                NCfgPrivate::ReportTypeMismatch(V_->TypeName(), "string");
            } else {
                NCfgPrivate::ReportTypeMismatch(V_->TypeName(), ::TypeName<T>());
            }
        }

        template <class T>
        inline T As() const {
            return ValueAs<T>(V_.Get());
        }

        template <class T>
        inline T As(T def) const {
            return IsNull() ? def : As<T>();
        }

        inline bool IsNull() const noexcept {
            return V_.Get() == Null();
        }

        const TConfig& Or(const TConfig& r) const {
            return IsNull() ? r : *this;
        }

        //assume value is dict
        bool Has(const TStringBuf& key) const;
        const TConfig& operator[](const TStringBuf& key) const;
        const TConfig& At(const TStringBuf& key) const;

        //assume value is array
        const TConfig& operator[](size_t index) const;
        size_t GetArraySize() const;

        static TConfig FromIni(IInputStream& in, const TGlobals& g = TGlobals());
        static TConfig FromJson(IInputStream& in, const TGlobals& g = TGlobals());
        static TConfig FromLua(IInputStream& in, const TGlobals& g = TGlobals());
        //load yconf format. unsafe, but natural mapping
        static TConfig FromMarkup(IInputStream& in, const TGlobals& g = TGlobals());

        static TConfig FromStream(IInputStream& in, const TGlobals& g = TGlobals());

        inline void ToJson(IOutputStream& out) const {
            V_->ToJson(out);
        }

        void DumpJson(IOutputStream& out) const;
        void DumpLua(IOutputStream& out) const;

        static TConfig ReadJson(TStringBuf in, const TGlobals& g = TGlobals());
        static TConfig ReadLua(TStringBuf in, const TGlobals& g = TGlobals());
        static TConfig ReadMarkup(TStringBuf in, const TGlobals& g = TGlobals());
        static TConfig ReadIni(TStringBuf in, const TGlobals& g = TGlobals());

        void Load(IInputStream* stream);
        void Save(IOutputStream* stream) const;

    private:
        TIntrusivePtr<IValue> V_;
    };

    struct TArray: public TDeque<TConfig> {
        const TConfig& Index(size_t index) const;
        const TConfig& At(size_t index) const;
    };

    struct TDict: public THashMap<TString, TConfig> {
        const TConfig& Find(const TStringBuf& key) const;
        const TConfig& At(const TStringBuf& key) const;
    };

    THolder<IInputStream> CreatePreprocessor(const TGlobals& g, IInputStream& in);
}
