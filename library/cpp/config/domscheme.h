#pragma once

#include "config.h"

#include <util/generic/algorithm.h>
#include <util/generic/typetraits.h>
#include <util/stream/str.h>

struct TConfigTraits {
    using TValue = NConfig::TConfig;
    using TValueRef = const TValue*;
    using TConstValueRef = TValueRef;
    using TStringType = TString;

    // anyvalue defaults
    template <class T>
    static inline TValue Value(const T& t) {
        return TValue(NConfig::ConstructValue(t));
    }

    template <class T>
    static inline TValue Value(std::initializer_list<T> list) {
        NConfig::TArray result;
        for (const auto& t : list) {
            result.push_back(TValue(NConfig::ConstructValue(t)));
        }
        return TValue(NConfig::ConstructValue(std::move(result)));
    }

    static inline TConstValueRef Ref(const TValue& v) {
        return &v;
    }

    // common ops
    static inline bool IsNull(TConstValueRef v) {
        return v->IsNull();
    }

    static inline TString ToJson(TConstValueRef v) {
        TStringStream str;
        v->ToJson(str);
        return str.Str();
    }

    // struct ops
    static inline TConstValueRef GetField(TConstValueRef v, const TStringBuf& name) {
        return &(*v)[name];
    }

    // array ops
    static bool IsArray(TConstValueRef v) {
        return v->IsA<NConfig::TArray>();
    }

    using TArrayIterator = size_t;

    static inline TConstValueRef ArrayElement(TConstValueRef v, TArrayIterator n) {
        return &(*v)[n];
    }

    static inline size_t ArraySize(TConstValueRef v) {
        return v->GetArraySize();
    }

    static inline TArrayIterator ArrayBegin(TConstValueRef) {
        return 0;
    }

    static inline TArrayIterator ArrayEnd(TConstValueRef v) {
        return ArraySize(v);
    }

    // dict ops
    static bool IsDict(TConstValueRef v) {
        return v->IsA<NConfig::TDict>();
    }

    static inline TConstValueRef DictElement(TConstValueRef v, TStringBuf key) {
        return &(*v)[key];
    }

    static inline size_t DictSize(TConstValueRef v) {
        return v->Get<NConfig::TDict>().size();
    }

    using TDictIterator = NConfig::TDict::const_iterator;

    static inline TDictIterator DictBegin(TConstValueRef v) {
        return v->Get<NConfig::TDict>().begin();
    }

    static inline TDictIterator DictEnd(TConstValueRef v) {
        return v->Get<NConfig::TDict>().end();
    }

    static inline TStringBuf DictIteratorKey(TConstValueRef /*dict*/, const TDictIterator& it) {
        return it->first;
    }

    static inline TConstValueRef DictIteratorValue(TConstValueRef /*dict*/, const TDictIterator& it) {
        return &it->second;
    }

    // generic get
    template <typename T>
    static inline void Get(TConstValueRef v, T def, T& t) {
        t = v->As<T>(def);
    }

    static inline bool Get(TConstValueRef v, double def, double& t) {
        if (v->IsNumeric()) {
            t = v->As<double>(def);
            return true;
        }
        t = def;
        return false;
    }

    template <typename T>
    static inline void Get(TConstValueRef v, T& t) {
        t = v->As<T>();
    }

    template <typename T>
    static inline bool IsValidPrimitive(const T&, TConstValueRef v) {
        if (v->IsNull()) {
            return true;
        }

        try {
            v->As<T>();

            return true;
        } catch (const NConfig::TTypeMismatch&) {
        } catch (const TBadCastException&) {
        }

        return false;
    }

    template <class T>
    static inline void Set(TValueRef v, T&& t) {
        v->GetNonConstant<std::remove_const_t<std::remove_reference_t<T>>>() = t;
    }

    // validation ops
    static inline TVector<TString> GetKeys(TConstValueRef v) {
        TVector<TString> res;
        for (const auto& it : v->Get<NConfig::TDict>()) {
            res.push_back(it.first);
        }
        Sort(res.begin(), res.end());
        return res;
    }
};
