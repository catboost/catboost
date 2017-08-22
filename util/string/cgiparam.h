#pragma once

#include <util/generic/map.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>

#include <initializer_list>

struct TStringLess {
    template <class T1, class T2>
    inline bool operator()(const T1& t1, const T2& t2) const noexcept {
        return TStringBuf(t1) < TStringBuf(t2);
    }
};

class TCgiParameters: public ymultimap<TString, TString> {
public:
    TCgiParameters() {
    }

    explicit TCgiParameters(const TStringBuf cgiParamStr) {
        Scan(cgiParamStr);
    }

    TCgiParameters(std::initializer_list<std::pair<TString, TString>> il);

    void Flush() {
        erase(begin(), end());
    }

    size_t EraseAll(const TStringBuf name);

    size_t NumOfValues(const TStringBuf name) const {
        return count(name);
    }

    TString operator()() const {
        return Print();
    }

    void Scan(const TStringBuf cgiParStr, bool form = true);
    void ScanAdd(const TStringBuf cgiParStr);
    void ScanAddUnescaped(const TStringBuf cgiParStr);
    void ScanAddAll(const TStringBuf cgiParStr);

    /// Returns the string representation of all the stored parameters
    /**
     * @note The returned string has format <name1>=<value1>&<name2>=<value2>&...
     * @note Names and values in the returned string are CGI-escaped.
     */
    TString Print() const;
    char* Print(char* res) const;
    size_t PrintSize() const noexcept;

    std::pair<const_iterator, const_iterator> Range(const TStringBuf name) const {
        return equal_range(name);
    }
    const_iterator Find(const TStringBuf name, size_t numOfValue = 0) const;
    bool Has(const TStringBuf name, const TStringBuf value) const;
    bool Has(const TStringBuf name) const {
        const auto pair = equal_range(name);
        return pair.first != pair.second;
    }
    /// Returns value by name
    /**
     * @note The returned value is CGI-unescaped.
     */
    const TString& Get(const TStringBuf name, size_t numOfValue = 0) const;

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

    // will replace one or more values with a single one
    void ReplaceUnescaped(const TStringBuf key, const TStringBuf val);

    // will join multiple values into a single one using a separator
    // if val is a [possibly empty] non-NULL string, append it as well
    void JoinUnescaped(const TStringBuf key, TStringBuf sep, TStringBuf val = TStringBuf());

    bool Erase(const TStringBuf name, size_t numOfValue = 0);

    inline const char* FormField(const TStringBuf name, size_t numOfValue = 0) const {
        const_iterator it = Find(name, numOfValue);

        if (it == end()) {
            return nullptr;
        }

        return ~it->second;
    }
};
