#pragma once

#include <util/generic/vector.h>
#include <util/string/split.h>
#include <util/string/strip.h>

//! Converts string to vector of type T variables
template <typename T, typename TStringType, typename TDelim = char>
bool TryParseStringToVector(const TStringType& input, TVector<T>& result, const TDelim delim = ',', const bool useEmpty = true) {
    T currentValue;
    for (const auto& t : StringSplitter(input).Split(delim)) {
        auto sb = StripString(t.Token());
        if (!useEmpty && !sb) {
            continue;
        }
        if (!TryFromString<T>(sb, currentValue)) {
            return false;
        }
        result.push_back(currentValue);
    }
    return true;
}
