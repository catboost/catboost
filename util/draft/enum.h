#pragma once

#include <bitset>

#include <util/generic/strbuf.h>
#include <util/stream/str.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <utility>

class TEnumNotFoundException: public yexception {
};

#define EnumFromString(key, entries) EnumFromStringImpl(key, entries, Y_ARRAY_SIZE(entries))
#define EnumFromStringWithSize(key, entries, size) EnumFromStringImpl(key, entries, size)
#define FindEnumFromString(key, entries) FindEnumFromStringImpl(key, entries, Y_ARRAY_SIZE(entries))
#define FindEnumFromStringWithSize(key, entries, size) FindEnumFromStringImpl(key, entries, size)
#define EnumToString(key, entries) EnumToStringImpl(key, entries, Y_ARRAY_SIZE(entries))
#define EnumToStringWithSize(key, entries, size) EnumToStringImpl(key, entries, size)
#define PrintEnumItems(entries) PrintEnumItemsImpl(entries, Y_ARRAY_SIZE(entries))

template <class K1, class K2, class V>
const V* FindEnumFromStringImpl(K1 key, const std::pair<K2, V>* entries, size_t arraySize) {
    for (size_t i = 0; i < arraySize; i++)
        if (entries[i].first == key)
            return &entries[i].second;
    return nullptr;
}

// special version for const char*
template <class V>
const V* FindEnumFromStringImpl(const char* key, const std::pair<const char*, V>* entries, size_t arraySize) {
    for (size_t i = 0; i < arraySize; i++)
        if (entries[i].first && key && !strcmp(entries[i].first, key))
            return &entries[i].second;
    return nullptr;
}

template <class K, class V>
TString PrintEnumItemsImpl(const std::pair<K, V>* entries, size_t arraySize) {
    TString result;
    TStringOutput out(result);
    for (size_t i = 0; i < arraySize; i++)
        out << (i ? ", " : "") << "'" << entries[i].first << "'";
    return result;
}

// special version for const char*
template <class V>
TString PrintEnumItemsImpl(const std::pair<const char*, V>* entries, size_t arraySize) {
    TString result;
    TStringOutput out(result);
    for (size_t i = 0; i < arraySize; i++)
        out << (i ? ", " : "") << "'" << (entries[i].first ? entries[i].first : "<null>") << "'";
    return result;
}

template <class K1, class K2, class V>
const V* EnumFromStringImpl(K1 key, const std::pair<K2, V>* entries, size_t arraySize) {
    const V* res = FindEnumFromStringImpl(key, entries, arraySize);
    if (res)
        return res;

    ythrow TEnumNotFoundException() << "Key '" << key << "' not found in enum. Valid options are: " << PrintEnumItemsImpl(entries, arraySize) << ". ";
}

template <class K, class V>
const K* EnumToStringImpl(V value, const std::pair<K, V>* entries, size_t arraySize) {
    for (size_t i = 0; i < arraySize; i++)
        if (entries[i].second == value)
            return &entries[i].first;

    TEnumNotFoundException exc;
    exc << "Value '" << int(value) << "' not found in enum. Valid values are: ";
    for (size_t i = 0; i < arraySize; i++)
        exc << (i ? ", " : "") << int(entries[i].second);
    exc << ". ";
    ythrow exc;
}

///////////////////////////////////

template <class B>
inline void SetEnumFlagsForEmptySpec(B& flags, bool allIfEmpty) {
    if (allIfEmpty) {
        flags.set();
    } else {
        flags.reset();
    }
}

// all set by default
template <class E, size_t N, size_t B>
inline void SetEnumFlags(const std::pair<const char*, E> (&str2Enum)[N], TStringBuf optSpec,
                         std::bitset<B>& flags, bool allIfEmpty = true) {
    if (optSpec.empty()) {
        SetEnumFlagsForEmptySpec(flags, allIfEmpty);
    } else {
        flags.reset();
        for (const auto& it : StringSplitter(optSpec).Split(',')) {
            E e = *EnumFromStringImpl(ToString(it.Token()).data(), str2Enum, N);
            flags.set(e);
        }
    }
}

template <class E, size_t B>
inline void SetEnumFlags(const std::pair<const char*, E>* str2Enum, TStringBuf optSpec,
                         std::bitset<B>& flags, const size_t size,
                         bool allIfEmpty = true) {
    if (optSpec.empty()) {
        SetEnumFlagsForEmptySpec(flags, allIfEmpty);
    } else {
        flags.reset();
        for (const auto& it : StringSplitter(optSpec).Split(',')) {
            E e = *EnumFromStringImpl(ToString(it.Token()).data(), str2Enum, size);
            flags.set(e);
        }
    }
}

// for enums generated with GENERATE_ENUM_SERIALIZATION
template <class E, size_t B>
inline void SetEnumFlags(TStringBuf optSpec, std::bitset<B>& flags, bool allIfEmpty = true) {
    if (optSpec.empty()) {
        SetEnumFlagsForEmptySpec(flags, allIfEmpty);
    } else {
        flags.reset();
        for (const auto& it : StringSplitter(optSpec).Split(',')) {
            E e;
            if (!TryFromString(it.Token(), e))
                ythrow yexception() << "Unknown enum value '" << it.Token() << "'";
            flags.set((size_t)e);
        }
    }
}
