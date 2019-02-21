#pragma once

#include <util/generic/hash.h>
#include <util/generic/strbuf.h>
#include <util/string/split.h>
#include <util/string/iterator.h>
#include <util/charset/utf8.h>

////////////////////////////////////

template <class TVal>
inline bool CheckAndReturnField(TStringBuf field, TStringBuf fieldName, TVal& fieldVal) {
    size_t fnLength = fieldName.length();
    if ((field.length() >= (fnLength + 1)) &&
        field.StartsWith(fieldName) &&
        (field[fnLength] == '=')) {
        fieldVal = FromString<TVal>(field.SubStr(fnLength + 1));
        return true;
    }
    return false;
}

template <class It, class TVal>
bool CheckAndReturnField(It& it, TStringBuf fieldName, TVal& fieldVal) {
    return CheckAndReturnField(*it, fieldName, fieldVal);
}

template <class It, class TVal>
void ReturnField(It& it, TStringBuf fieldName, TVal& fieldVal) {
    if (!CheckAndReturnField(*it, fieldName, fieldVal))
        ythrow yexception() << "ReturnField(" << fieldName << ") failed";
}

template <class It>
bool ExtractField(It& it, TStringBuf& fieldName, TStringBuf& fieldVal) {
    return (*it).TrySplit('=', fieldName, fieldVal);
}

template <class It>
bool SkipUntilFieldFound(It& it, const It& eIt, TStringBuf fieldName, TStringBuf& fieldVal) {
    size_t fnLength = fieldName.length();

    for (; it != eIt; ++it) {
        TStringBuf field = *it;

        if ((field.length() >= (fnLength + 1)) &&
            field.StartsWith(fieldName) &&
            (field[fnLength] == '=')) {
            fieldVal = field.SubStr(fnLength + 1);
            return true;
        }
    }
    return false;
}

template <class TMapType>
void AddToNameValueHash(TStringBuf input, TMapType& nameValueMap) {
    using T = typename TMapType::mapped_type;

    TStringBuf name;
    TStringBuf value;
    for (const auto& it : StringSplitter(input).Split('\t')) {
        if (!it.Token().TrySplit('=', name, value))
            ythrow yexception() << "Failed to extract field from \"" << it.Token() << "\"";

        T val = FromString<T>(value);
        std::pair<typename TMapType::iterator, bool> insertionResult =
            nameValueMap.insert(std::make_pair(ToString(name), val));
        if (!insertionResult.second) // "name" already exists
            insertionResult.first->second += val;
    }
}

/////////////////////////////

bool IsNonBMPUTF8(TStringBuf s);

/////////////////////////////

inline TString CleanupASCIIControl(const TStringBuf text) {
    TTempBuf tempBuf(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        tempBuf.Data()[i] = ((ui8(text[i]) <= 32) ? ' ' : text[i]);
    }
    return TString(tempBuf.Data(), text.size());
}
