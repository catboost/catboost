#include "scan.h"
#include "quote.h"
#include "cgiparam.h"

#include <util/generic/singleton.h>

TCgiParameters::TCgiParameters(std::initializer_list<std::pair<TString, TString>> il) {
    for (const auto& item : il) {
        insert(item);
    }
}

const TString& TCgiParameters::Get(const TStringBuf name, size_t numOfValue) const {
    const auto it = Find(name, numOfValue);

    return end() == it ? Default<TString>() : it->second;
}

bool TCgiParameters::Erase(const TStringBuf name, size_t pos) {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it, --pos) {
        if (0 == pos) {
            erase(it);
            return true;
        }
    }

    return false;
}

size_t TCgiParameters::EraseAll(const TStringBuf name) {
    size_t num = 0;

    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; erase(it++), ++num)
        ;

    return num;
}

void TCgiParameters::ReplaceUnescaped(const TStringBuf key, const TStringBuf val) {
    const auto pair = equal_range(key);
    auto it = pair.first;

    if (it == pair.second) { // not found
        emplace_hint(it, TString(key), TString(val));
    } else {
        it->second = val;

        for (++it; it != pair.second; erase(it++))
            ;
    }
}

void TCgiParameters::JoinUnescaped(const TStringBuf key, TStringBuf sep, TStringBuf val) {
    const auto pair = equal_range(key);
    auto it = pair.first;

    if (it == pair.second) { // not found
        if (val.IsInited()) {
            emplace_hint(it, TString(key), TString(val));
        }
    } else {
        TString& dst = it->second;

        for (++it; it != pair.second; erase(it++)) {
            dst += sep;
            dst.AppendNoAlias(~it->second, +it->second);
        }

        if (val.IsInited()) {
            dst += sep;
            dst += val;
        }
    }
}

static inline TString DoUnescape(const TStringBuf s) {
    TString res;

    res.reserve(CgiUnescapeBufLen(+s));
    res.ReserveAndResize(+CgiUnescape(res.begin(), s));

    return res;
}

void TCgiParameters::InsertEscaped(const TStringBuf name, const TStringBuf value) {
    InsertUnescaped(DoUnescape(name), DoUnescape(value));
}

template <bool addAll, class F>
static inline void DoScan(const TStringBuf s, F& f) {
    ScanKeyValue<addAll, '&', '='>(s, f);
}

struct TAddEscaped {
    TCgiParameters* C;

    inline void operator()(const TStringBuf key, const TStringBuf val) {
        C->InsertEscaped(key, val);
    }
};

void TCgiParameters::Scan(const TStringBuf query, bool form) {
    Flush();
    form ? ScanAdd(query) : ScanAddAll(query);
}

void TCgiParameters::ScanAdd(const TStringBuf query) {
    TAddEscaped f = {this};

    DoScan<false>(query, f);
}

void TCgiParameters::ScanAddUnescaped(const TStringBuf query) {
    auto f = [this](const TStringBuf key, const TStringBuf val) {
        this->InsertUnescaped(key, val);
    };

    DoScan<false>(query, f);
}

void TCgiParameters::ScanAddAll(const TStringBuf query) {
    TAddEscaped f = {this};

    DoScan<true>(query, f);
}

TString TCgiParameters::Print() const {
    TString res;

    res.reserve(PrintSize());
    const char* end = Print(res.begin());
    res.ReserveAndResize(end - ~res);

    return res;
}

char* TCgiParameters::Print(char* res) const {
    if (empty()) {
        return res;
    }

    for (auto i = begin();;) {
        res = CGIEscape(res, i->first);
        *res++ = '=';
        res = CGIEscape(res, i->second);

        if (++i == end()) {
            break;
        }

        *res++ = '&';
    }

    return res;
}

size_t TCgiParameters::PrintSize() const noexcept {
    size_t res = size(); // for '&'

    for (const auto& i : *this) {
        res += CgiEscapeBufLen(+i.first + +i.second); // extra zero will be used for '='
    }

    return res;
}

TCgiParameters::const_iterator TCgiParameters::Find(const TStringBuf name, size_t pos) const {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it, --pos) {
        if (0 == pos) {
            return it;
        }
    }

    return end();
}

bool TCgiParameters::Has(const TStringBuf name, const TStringBuf value) const {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it) {
        if (value == it->second) {
            return true;
        }
    }

    return false;
}
