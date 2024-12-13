#include "cgiparam.h"

#include <library/cpp/string_utils/scan/scan.h>
#include <library/cpp/string_utils/quote/quote.h>

#include <util/generic/singleton.h>

TCgiParameters::TCgiParameters(std::initializer_list<std::pair<TString, TString>> il) {
    for (const auto& item : il) {
        insert(item);
    }
}

const TString& TCgiParameters::Get(const TStringBuf name, size_t numOfValue) const noexcept Y_LIFETIME_BOUND {
    const auto it = Find(name, numOfValue);

    return end() == it ? Default<TString>() : it->second;
}

const TString& TCgiParameters::GetLast(const TStringBuf name) const noexcept {
    if (auto it = this->upper_bound(name); it != this->begin()) {
        --it;
        if (it->first == name) {
            return it->second;
        }
    }
    return Default<TString>();
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

bool TCgiParameters::Erase(const TStringBuf name, const TStringBuf val) {
    const auto pair = equal_range(name);

    bool found = false;
    for (auto it = pair.first; it != pair.second;) {
        if (val == it->second) {
            it = erase(it);
            found = true;
        } else {
            ++it;
        }
    }

    return found;
}

bool TCgiParameters::ErasePattern(const TStringBuf name, const TStringBuf pat) {
    const auto pair = equal_range(name);

    bool found = false;
    for (auto it = pair.first; it != pair.second;) {
        bool startsWith = it->second.StartsWith(pat);
        if (startsWith) {
            it = erase(it);
            found = true;
        } else {
            ++it;
        }
    }

    return found;
}

size_t TCgiParameters::EraseAll(const TStringBuf name) {
    size_t num = 0;

    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; erase(it++), ++num)
        ;

    return num;
}

void TCgiParameters::JoinUnescaped(const TStringBuf key, char sep, TStringBuf val) {
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
            dst.AppendNoAlias(it->second.data(), it->second.size());
        }

        if (val.IsInited()) {
            dst += sep;
            dst += val;
        }
    }
}

static inline TString DoUnescape(const TStringBuf s) {
    TString res;

    res.ReserveAndResize(CgiUnescapeBufLen(s.size()));
    res.resize(CgiUnescape(res.begin(), s).size());

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

void TCgiParameters::ScanAddAllUnescaped(const TStringBuf query) {
    auto f = [this](const TStringBuf key, const TStringBuf val) {
        this->InsertUnescaped(key, val);
    };

    DoScan<true>(query, f);
}

void TCgiParameters::ScanAddAll(const TStringBuf query) {
    TAddEscaped f = {this};

    DoScan<true>(query, f);
}

TString TCgiParameters::Print() const {
    TString res;

    res.ReserveAndResize(PrintSize());
    const char* end = Print(res.begin());
    res.ReserveAndResize(end - res.data());

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
        res += CgiEscapeBufLen(i.first.size() + i.second.size()); // extra zero will be used for '='
    }

    return res;
}

TString TCgiParameters::QuotedPrint(const char* safe) const {
    if (empty()) {
        return TString();
    }

    TString res;
    res.ReserveAndResize(PrintSize());

    char* ptr = res.begin();
    for (auto i = begin();;) {
        ptr = Quote(ptr, i->first, safe);
        *ptr++ = '=';
        ptr = Quote(ptr, i->second, safe);

        if (++i == end()) {
            break;
        }

        *ptr++ = '&';
    }

    res.ReserveAndResize(ptr - res.data());
    return res;
}

TCgiParameters::const_iterator TCgiParameters::Find(const TStringBuf name, size_t pos) const noexcept Y_LIFETIME_BOUND {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it, --pos) {
        if (0 == pos) {
            return it;
        }
    }

    return end();
}

bool TCgiParameters::Has(const TStringBuf name, const TStringBuf value) const noexcept {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it) {
        if (value == it->second) {
            return true;
        }
    }

    return false;
}

TQuickCgiParam::TQuickCgiParam(const TStringBuf cgiParamStr) {
    UnescapeBuf.ReserveAndResize(CgiUnescapeBufLen(cgiParamStr.size()));
    char* buf = UnescapeBuf.begin();

    auto f = [this, &buf](const TStringBuf key, const TStringBuf val) {
        TStringBuf name = CgiUnescapeBuf(buf, key);
        buf += name.size() + 1;
        TStringBuf value = CgiUnescapeBuf(buf, val);
        buf += value.size() + 1;
        Y_ASSERT(buf <= UnescapeBuf.begin() + UnescapeBuf.capacity() + 1 /*trailing zero*/);
        emplace(name, value);
    };

    DoScan<false>(cgiParamStr, f);

    if (buf != UnescapeBuf.begin()) {
        UnescapeBuf.ReserveAndResize(buf - UnescapeBuf.begin() - 1 /*trailing zero*/);
    }
}

TStringBuf TQuickCgiParam::Get(const TStringBuf name, size_t pos) const noexcept Y_LIFETIME_BOUND {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it, --pos) {
        if (0 == pos) {
            return it->second;
        }
    }

    return TStringBuf{};
}

bool TQuickCgiParam::Has(const TStringBuf name, const TStringBuf value) const noexcept {
    const auto pair = equal_range(name);

    for (auto it = pair.first; it != pair.second; ++it) {
        if (value == it->second) {
            return true;
        }
    }

    return false;
}
