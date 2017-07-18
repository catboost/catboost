#pragma once

#include "json_value.h"
#include "json_reader.h"
#include <util/generic/algorithm.h>

struct TJsonTraits {
    using TValueRef = NJson::TJsonValue*;
    using TConstValueRef = const NJson::TJsonValue*;
    using TStringType = TStringBuf;

    // common ops
    static inline bool IsNull(TConstValueRef v) {
        return v->GetType() == NJson::JSON_UNDEFINED || v->IsNull();
    }

    // struct ops
    static inline TValueRef GetField(TValueRef v, const TStringBuf& name) {
        return &(*v)[name];
    }

    static inline TConstValueRef GetField(TConstValueRef v, const TStringBuf& name) {
        return &(*v)[name];
    }

    // array ops
    static bool IsArray(TConstValueRef v) {
        return v->IsArray();
    }

    static inline void ArrayClear(TValueRef v) {
        v->SetType(NJson::JSON_NULL);
        v->SetType(NJson::JSON_ARRAY);
    }

    static inline TValueRef ArrayElement(TValueRef v, size_t n) {
        return &(*v)[n];
    }

    static inline TConstValueRef ArrayElement(TConstValueRef v, size_t n) {
        return &(*v)[n];
    }

    static inline size_t ArraySize(TConstValueRef v) {
        return v->GetArray().size();
    }

    using TArrayIterator = size_t;

    static inline TArrayIterator ArrayBegin(TConstValueRef) {
        return 0;
    }

    static inline TArrayIterator ArrayEnd(TConstValueRef v) {
        return ArraySize(v);
    }

    // dict ops
    static bool IsDict(TConstValueRef v) {
        return v->IsMap();
    }

    static inline void DictClear(TValueRef v) {
        v->SetType(NJson::JSON_NULL);
        v->SetType(NJson::JSON_MAP);
    }

    static inline TValueRef DictElement(TValueRef v, TStringBuf key) {
        return &(*v)[key];
    }

    static inline TConstValueRef DictElement(TConstValueRef v, TStringBuf key) {
        return &(*v)[key];
    }

    static inline size_t DictSize(TConstValueRef v) {
        return v->GetMap().size();
    }

    using TDictIterator = NJson::TJsonValue::TMap::const_iterator;

    static inline TDictIterator DictBegin(TConstValueRef v) {
        return v->GetMap().begin();
    }

    static inline TDictIterator DictEnd(TConstValueRef v) {
        return v->GetMap().end();
    }

    static inline TStringBuf DictIteratorKey(TConstValueRef /*dict*/, const TDictIterator& it) {
        return it->first;
    }

    static inline TConstValueRef DictIteratorValue(TConstValueRef /*dict*/, const TDictIterator& it) {
        return &it->second;
    }

    // boolean ops
    static inline void Get(TConstValueRef v, bool def, bool& b) {
        b =
            v->GetType() == NJson::JSON_UNDEFINED ? def :
            v->IsNull() ? def :
            v->GetBooleanRobust();
    }

    static inline void Get(TConstValueRef v, bool& b) {
        Get(v, false, b);
    }

    static inline bool IsValidPrimitive(const bool&, TConstValueRef v) {
        return v->IsBoolean();
    }

#define INTEGER_OPS(type, checkOp, getOp)                               \
    static inline void Get(TConstValueRef v, type def, type& i) {        \
        i = v->checkOp() ? v->getOp() : def;                            \
    }                                                                   \
    static inline void Get(TConstValueRef v, type& i) {                  \
        i = v->getOp();                                                 \
    }                                                                   \
    static inline bool IsValidPrimitive(const type&, TConstValueRef v) { \
        return v->checkOp() && v->getOp() >= Min<type>() && v->getOp() <= Max<type>(); \
    }

    INTEGER_OPS(i8, IsInteger, GetInteger)
    INTEGER_OPS(i16, IsInteger, GetInteger)
    INTEGER_OPS(i32, IsInteger, GetInteger)
    INTEGER_OPS(i64, IsInteger, GetInteger)
    INTEGER_OPS(ui8, IsUInteger, GetUInteger)
    INTEGER_OPS(ui16, IsUInteger, GetUInteger)
    INTEGER_OPS(ui32, IsUInteger, GetUInteger)
    INTEGER_OPS(ui64, IsUInteger, GetUInteger)

#undef INTEGER_OPS

    // double ops
    static inline bool Get(TConstValueRef v, double def, double& d) {
        if (v->IsDouble()) {
            d = v->GetDouble();
            return true;
        }
        d = def;
        return false;
    }

    static inline void Get(TConstValueRef v, double& d) {
        d = v->GetDouble();
    }

    static inline bool IsValidPrimitive(const double&, TConstValueRef v) {
        return v->IsDouble();
    }

    // string ops
    static inline void Get(TConstValueRef v, TStringBuf def, TStringBuf& s) {
        s = v->IsString() ? v->GetString() : def;
    }

    static inline void Get(TConstValueRef v, TStringBuf& s) {
        s = v->GetString();
    }

    static inline bool IsValidPrimitive(const TStringBuf&, TConstValueRef v) {
        return v->IsString();
    }

    // generic set
    template <class T>
    static inline void Set(TValueRef v, T&& t) {
        v->SetValue(t);
    }

    static inline void Clear(TValueRef v) {
        v->SetType(NJson::JSON_NULL);
    }

    // validation ops
    static inline yvector<TString> GetKeys(TConstValueRef v) {
        yvector<TString> res;
        for (const auto& it : v->GetMap()) {
            res.push_back(it.first);
        }
        Sort(res.begin(), res.end());
        return res;
    }

    template <typename T>
    static inline bool IsValidPrimitive(const T&, TConstValueRef) {
        return false;
    }
};
