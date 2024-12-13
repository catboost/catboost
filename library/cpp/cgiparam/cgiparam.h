#pragma once

#include <library/cpp/iterator/iterate_values.h>

#include <util/generic/iterator_range.h>
#include <util/generic/map.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>

#include <initializer_list>

class TCgiParameters: public TMultiMap<TString, TString> {
public:
    TCgiParameters() = default;

    explicit TCgiParameters(const TStringBuf cgiParamStr) {
        Scan(cgiParamStr);
    }

    TCgiParameters(std::initializer_list<std::pair<TString, TString>> il);

    void Flush() {
        erase(begin(), end());
    }

    size_t EraseAll(const TStringBuf name);

    size_t NumOfValues(const TStringBuf name) const noexcept {
        return count(name);
    }

    TString operator()() const {
        return Print();
    }

    void Scan(const TStringBuf cgiParStr, bool form = true);
    void ScanAdd(const TStringBuf cgiParStr);
    void ScanAddUnescaped(const TStringBuf cgiParStr);
    void ScanAddAllUnescaped(const TStringBuf cgiParStr);
    void ScanAddAll(const TStringBuf cgiParStr);

    /// Returns the string representation of all the stored parameters
    /**
     * @note The returned string has format <name1>=<value1>&<name2>=<value2>&...
     * @note Names and values in the returned string are CGI-escaped.
     */
    TString Print() const;
    char* Print(char* res) const;

    Y_PURE_FUNCTION
    size_t PrintSize() const noexcept;

    /** The same as Print* except that RFC-3986 reserved characters are escaped.
     * @param safe - set of characters to be skipped in escaping
     */
    TString QuotedPrint(const char* safe = "/") const;

    Y_PURE_FUNCTION
    auto Range(const TStringBuf name) const noexcept {
        return IterateValues(MakeIteratorRange(equal_range(name)));
    }

    Y_PURE_FUNCTION
    const_iterator Find(const TStringBuf name, size_t numOfValue = 0) const noexcept Y_LIFETIME_BOUND;

    Y_PURE_FUNCTION
    bool Has(const TStringBuf name, const TStringBuf value) const noexcept;

    Y_PURE_FUNCTION
    bool Has(const TStringBuf name) const noexcept {
        const auto pair = equal_range(name);
        return pair.first != pair.second;
    }
    /// Returns value by name
    /**
     * @note The returned value is CGI-unescaped.
     */
    Y_PURE_FUNCTION
    const TString& Get(const TStringBuf name, size_t numOfValue = 0) const noexcept Y_LIFETIME_BOUND;

    /// Returns the last value by name
    /**
     * @note The returned value is CGI-unescaped.
     */
    Y_PURE_FUNCTION
    const TString& GetLast(const TStringBuf name) const noexcept Y_LIFETIME_BOUND;

    void InsertEscaped(const TStringBuf name, const TStringBuf value);

#if !defined(__GLIBCXX__)
    template <typename TName, typename TValue>
    inline void InsertUnescaped(TName&& name, TValue&& value) {
        // TStringBuf use as TName or TValue is C++17 actually.
        // There is no pair constructor available in C++14 when required type
        // is not implicitly constructible from given type.
        // But libc++ pair allows this with C++14.
        emplace(std::forward<TName>(name), std::forward<TValue>(value));
    }
#else
    template <typename TName, typename TValue>
    inline void InsertUnescaped(TName&& name, TValue&& value) {
        emplace(TString(name), TString(value));
    }
#endif

    // replace all values for a given key with new values
    template <typename TIter>
    void ReplaceUnescaped(const TStringBuf key, TIter valuesBegin, const TIter valuesEnd);

    void ReplaceUnescaped(const TStringBuf key, std::initializer_list<TStringBuf> values) {
        ReplaceUnescaped(key, values.begin(), values.end());
    }

    void ReplaceUnescaped(const TStringBuf key, const TStringBuf value) {
        ReplaceUnescaped(key, {value});
    }

    // join multiple values into a single one using a separator
    // if val is a [possibly empty] non-NULL string, append it as well
    void JoinUnescaped(const TStringBuf key, char sep, TStringBuf val = TStringBuf());

    bool Erase(const TStringBuf name, size_t numOfValue = 0);
    bool Erase(const TStringBuf name, const TStringBuf val);
    bool ErasePattern(const TStringBuf name, const TStringBuf pat);

    inline const char* FormField(const TStringBuf name, size_t numOfValue = 0) const Y_LIFETIME_BOUND {
        const_iterator it = Find(name, numOfValue);

        if (it == end()) {
            return nullptr;
        }

        return it->second.c_str();
    }

    inline TStringBuf FormFieldBuf(const TStringBuf name, size_t numOfValue = 0) const Y_LIFETIME_BOUND {
        const_iterator it = Find(name, numOfValue);

        if (it == end()) {
            return TStringBuf{};
        }

        return it->second;
    }
};

template <typename TIter>
void TCgiParameters::ReplaceUnescaped(const TStringBuf key, TIter valuesBegin, const TIter valuesEnd) {
    const auto oldRange = equal_range(key);
    auto current = oldRange.first;

    // reuse as many existing nodes as possible (probably none)
    for (; valuesBegin != valuesEnd && current != oldRange.second; ++valuesBegin, ++current) {
        current->second = *valuesBegin;
    }

    // if there were more nodes than we need to insert then erase remaining ones
    for (; current != oldRange.second; erase(current++)) {
    }

    // if there were less nodes than we need to insert then emplace the rest of the range
    if (valuesBegin != valuesEnd) {
        const TString keyStr = TString(key);
        for (; valuesBegin != valuesEnd; ++valuesBegin) {
            emplace_hint(oldRange.second, keyStr, TString(*valuesBegin));
        }
    }
}

/** TQuickCgiParam is a faster non-editable version of TCgiParameters.
 * Care should be taken when replacing:
 *  - note that the result of Get() is invalidated when TQuickCgiParam object is destroyed.
 */

class TQuickCgiParam: public TMultiMap<TStringBuf, TStringBuf> {
public:
    TQuickCgiParam() = default;

    explicit TQuickCgiParam(const TStringBuf cgiParamStr);

    Y_PURE_FUNCTION
    bool Has(const TStringBuf name, const TStringBuf value) const noexcept;

    Y_PURE_FUNCTION
    bool Has(const TStringBuf name) const noexcept {
        const auto pair = equal_range(name);
        return pair.first != pair.second;
    }

    Y_PURE_FUNCTION
    TStringBuf Get(const TStringBuf name, size_t numOfValue = 0) const noexcept Y_LIFETIME_BOUND;

private:
    TString UnescapeBuf;
};
