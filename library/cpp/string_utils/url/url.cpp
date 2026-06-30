#include "url.h"

#include <util/string/cast.h>
#include <util/string/util.h>
#include <util/string/cstriter.h>
#include <util/string/ascii.h>
#include <util/string/strip.h>

#include <util/charset/unidata.h> // for ToLower
#include <util/system/defaults.h>
#include <util/generic/algorithm.h>
#include <util/generic/hash_set.h>
#include <util/generic/yexception.h>
#include <util/generic/singleton.h>

#include <cstdlib>

namespace {
    struct TUncheckedSize {
        static bool Has(size_t) {
            return true;
        }
    };

    struct TKnownSize {
        size_t MySize;
        explicit TKnownSize(size_t sz)
            : MySize(sz)
        {
        }
        bool Has(size_t sz) const {
            return sz <= MySize;
        }
    };

    template <typename TChar1, typename TChar2>
    int Compare1Case2(const TChar1* s1, const TChar2* s2, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if ((TChar1)ToLower(s1[i]) != s2[i])
                return (TChar1)ToLower(s1[i]) < s2[i] ? -1 : 1;
        }
        return 0;
    }

    template <typename TChar, typename TBounds>
    inline size_t GetHttpPrefixSizeImpl(const TChar* url, const TBounds& urlSize, bool ignorehttps) {
        const TChar httpPrefix[] = {'h', 't', 't', 'p', ':', '/', '/', 0};
        const TChar httpsPrefix[] = {'h', 't', 't', 'p', 's', ':', '/', '/', 0};
        if (urlSize.Has(7) && Compare1Case2(url, httpPrefix, 7) == 0)
            return 7;
        if (!ignorehttps && urlSize.Has(8) && Compare1Case2(url, httpsPrefix, 8) == 0)
            return 8;
        return 0;
    }

    template <typename T>
    inline T CutHttpPrefixImpl(const T& url, bool ignorehttps) {
        size_t prefixSize = GetHttpPrefixSizeImpl<typename T::char_type>(url.data(), TKnownSize(url.size()), ignorehttps);
        if (prefixSize)
            return url.substr(prefixSize);
        return url;
    }
}

namespace NUrl {

    TSplitUrlToHostAndPathResult SplitUrlToHostAndPath(const TStringBuf url) {
        TStringBuf host = GetSchemeHostAndPort(url, /*trimHttp=*/false, /*trimDefaultPort=*/false);
        TStringBuf path = url;
        path.SkipPrefix(host);
        return {host, path};
    }

    bool HasLowerHost(const TStringBuf url) {
        for (size_t n = 0; n < url.length(); ++n) {
            if (url[n] == '/')
                break;
            if (isupper(url[n]))
                return false;
        }
        return true;
    }

    TStringBuf CutHttpWwwPrefixes(const TStringBuf url Y_LIFETIME_BOUND) {
        TStringBuf urlCut = CutWWWPrefix(CutHttpPrefix(url));
        if (!urlCut.empty() && urlCut.back() == '/')
            urlCut = urlCut.substr(0, urlCut.length() - 1);
        return urlCut;
    }

    TString MakeLowerHost(const TStringBuf url, size_t shift) {
        TString urlFixed(url);
        for (char *c = urlFixed.begin() + shift; *c && (*c != '/'); ++c) {
            *c = tolower(*c);
        }

        return urlFixed;
    }

    TString MakeNormalized(const TStringBuf url) {
        TStringBuf urlCut = CutHttpWwwPrefixes(url);
        if (HasLowerHost(urlCut)) {
            return ToString(urlCut);
        }
        return MakeLowerHost(urlCut);
    }

} // namespace NUrl

size_t GetHttpPrefixSize(const char* url, bool ignorehttps) noexcept {
    return GetHttpPrefixSizeImpl<char>(url, TUncheckedSize(), ignorehttps);
}

size_t GetHttpPrefixSize(const wchar16* url, bool ignorehttps) noexcept {
    return GetHttpPrefixSizeImpl<wchar16>(url, TUncheckedSize(), ignorehttps);
}

size_t GetHttpPrefixSize(const TStringBuf url, bool ignorehttps) noexcept {
    return GetHttpPrefixSizeImpl<char>(url.data(), TKnownSize(url.size()), ignorehttps);
}

size_t GetHttpPrefixSize(const TWtringBuf url, bool ignorehttps) noexcept {
    return GetHttpPrefixSizeImpl<wchar16>(url.data(), TKnownSize(url.size()), ignorehttps);
}

TStringBuf CutHttpPrefix(const TStringBuf url Y_LIFETIME_BOUND, bool ignorehttps) noexcept {
    return CutHttpPrefixImpl(url, ignorehttps);
}

TWtringBuf CutHttpPrefix(const TWtringBuf url Y_LIFETIME_BOUND, bool ignorehttps) noexcept {
    return CutHttpPrefixImpl(url, ignorehttps);
}

size_t GetSchemePrefixSize(const TStringBuf url) noexcept {
    if (url.empty()) {
        return 0;
    }

    struct TDelim: public str_spn {
        inline TDelim()
            : str_spn("!-/:-@[-`{|}", true)
        {
        }
    };

    const auto& delim = *Singleton<TDelim>();
    const char* n = delim.brk(url.data(), url.end());

    if (n + 2 >= url.end() || *n != ':' || n[1] != '/' || n[2] != '/') {
        return 0;
    }

    return n + 3 - url.begin();
}

TStringBuf GetSchemePrefix(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    return url.Head(GetSchemePrefixSize(url));
}

TStringBuf CutSchemePrefix(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    return url.Tail(GetSchemePrefixSize(url));
}

template <bool KeepPort>
static inline TStringBuf GetHostAndPortImpl(const TStringBuf url) {
    TStringBuf urlNoScheme = url;

    urlNoScheme.Skip(GetHttpPrefixSize(url));

    struct TDelim: public str_spn {
        inline TDelim()
            : str_spn(KeepPort ? "/;?#" : "/:;?#")
        {
        }
    };

    const auto& nonHostCharacters = *Singleton<TDelim>();
    const char* firstNonHostCharacter = nonHostCharacters.brk(urlNoScheme.begin(), urlNoScheme.end());

    if (firstNonHostCharacter != urlNoScheme.end()) {
        return urlNoScheme.substr(0, firstNonHostCharacter - urlNoScheme.data());
    }

    return urlNoScheme;
}

TStringBuf GetHost(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    return GetHostAndPortImpl<false>(url);
}

TStringBuf GetHostAndPort(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    return GetHostAndPortImpl<true>(url);
}

TStringBuf GetSchemeHost(const TStringBuf url Y_LIFETIME_BOUND, bool trimHttp) noexcept {
    const size_t schemeSize = GetSchemePrefixSize(url);
    const TStringBuf scheme = url.Head(schemeSize);

    const bool isHttp = (schemeSize == 0 || scheme == TStringBuf("http://"));

    const TStringBuf host = GetHost(url.Tail(schemeSize));

    if (isHttp && trimHttp) {
        return host;
    } else {
        return TStringBuf(scheme.begin(), host.end());
    }
}

TStringBuf GetSchemeHostAndPort(const TStringBuf url Y_LIFETIME_BOUND, bool trimHttp, bool trimDefaultPort) noexcept {
    const size_t schemeSize = GetSchemePrefixSize(url);
    const TStringBuf scheme = url.Head(schemeSize);

    const bool isHttp = (schemeSize == 0 || scheme == TStringBuf("http://"));

    TStringBuf hostAndPort = GetHostAndPort(url.Tail(schemeSize));

    if (trimDefaultPort) {
        const size_t pos = hostAndPort.find(':');
        if (pos != TStringBuf::npos) {
            const bool isHttps = (scheme == TStringBuf("https://"));

            const TStringBuf port = hostAndPort.Tail(pos + 1);
            if ((isHttp && port == TStringBuf("80")) || (isHttps && port == TStringBuf("443"))) {
                // trimming default port
                hostAndPort = hostAndPort.Head(pos);
            }
        }
    }

    if (isHttp && trimHttp) {
        return hostAndPort;
    } else {
        return TStringBuf(scheme.begin(), hostAndPort.end());
    }
}

void SplitUrlToHostAndPath(const TStringBuf url, TStringBuf& host, TStringBuf& path) {
    auto [hostBuf, pathBuf] = NUrl::SplitUrlToHostAndPath(url);
    host = hostBuf;
    path = pathBuf;
}

void SplitUrlToHostAndPath(const TStringBuf url, TString& host, TString& path) {
    auto [hostBuf, pathBuf] = NUrl::SplitUrlToHostAndPath(url);
    host = hostBuf;
    path = pathBuf;
}

void SeparateUrlFromQueryAndFragment(const TStringBuf url, TStringBuf& sanitizedUrl, TStringBuf& query, TStringBuf& fragment) {
    TStringBuf urlWithoutFragment;
    if (!url.TrySplit('#', urlWithoutFragment, fragment)) {
        fragment = "";
        urlWithoutFragment = url;
    }
    if (!urlWithoutFragment.TrySplit('?', sanitizedUrl, query)) {
        query = "";
        sanitizedUrl = urlWithoutFragment;
    }
}

bool TryGetSchemeHostAndPort(const TStringBuf url, TStringBuf& scheme, TStringBuf& host, ui16& port) {
    const size_t schemeSize = GetSchemePrefixSize(url);
    if (schemeSize != 0) {
        scheme = url.Head(schemeSize);
    }

    TStringBuf portStr;
    TStringBuf hostAndPort = GetHostAndPort(url.Tail(schemeSize));
    if (hostAndPort && hostAndPort.back() != ']' && hostAndPort.TryRSplit(':', host, portStr)) {
        // URL has port
        if (!TryFromString(portStr, port)) {
            return false;
        }
    } else {
        host = hostAndPort;
        if (scheme == TStringBuf("https://")) {
            port = 443;
        } else if (scheme == TStringBuf("http://")) {
            port = 80;
        }
    }
    return true;
}

void GetSchemeHostAndPort(const TStringBuf url, TStringBuf& scheme, TStringBuf& host, ui16& port) {
    bool isOk = TryGetSchemeHostAndPort(url, scheme, host, port);
    Y_ENSURE(isOk, "cannot parse port number from URL: " << url);
}

TStringBuf GetOnlyHost(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    return GetHost(CutSchemePrefix(url));
}

TStringBuf GetPathAndQuery(const TStringBuf url Y_LIFETIME_BOUND, bool trimFragment) noexcept {
    const size_t off = url.find('/', GetHttpPrefixSize(url));
    TStringBuf hostUnused, path;
    if (!url.TrySplitAt(off, hostUnused, path))
        return "/";

    return trimFragment ? path.Before('#') : path;
}

// this strange creature returns 2nd level domain, possibly with port
TStringBuf GetDomain(const TStringBuf host Y_LIFETIME_BOUND) noexcept {
    const char* c = !host ? host.data() : host.end() - 1;
    for (bool wasPoint = false; c != host.data(); --c) {
        if (*c == '.') {
            if (wasPoint) {
                ++c;
                break;
            }
            wasPoint = true;
        }
    }
    return TStringBuf(c, host.end());
}

TStringBuf GetParentDomain(const TStringBuf host Y_LIFETIME_BOUND, size_t level) noexcept {
    size_t pos = host.size();
    for (size_t i = 0; i < level; ++i) {
        pos = host.rfind('.', pos);
        if (pos == TString::npos)
            return host;
    }
    return host.SubStr(pos + 1);
}

TStringBuf GetZone(const TStringBuf host Y_LIFETIME_BOUND) noexcept {
    return GetParentDomain(host, 1);
}

TStringBuf CutWWWPrefix(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    if (url.size() >= 4 && url[3] == '.' && !strnicmp(url.data(), "www", 3))
        return url.substr(4);
    return url;
}

TStringBuf CutWWWNumberedPrefix(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    auto it = url.begin();

    StripRangeBegin(it, url.end(), [](auto& it){ return *it == 'w' || *it == 'W'; });
    if (it == url.begin()) {
        return url;
    }

    StripRangeBegin(it, url.end(), [](auto& it){ return IsAsciiDigit(*it); });
    if (it == url.end()) {
        return url;
    }

    if (*it++ == '.') {
        return url.Tail(it - url.begin());
    }

    return url;
}

TStringBuf CutMPrefix(const TStringBuf url Y_LIFETIME_BOUND) noexcept {
    if (url.size() >= 2 && url[1] == '.' && (url[0] == 'm' || url[0] == 'M')) {
        return url.substr(2);
    }
    return url;
}

static inline bool IsSchemeChar(char c) noexcept {
    return IsAsciiAlnum(c); //what about '+' ?..
}

static bool HasPrefix(const TStringBuf url) noexcept {
    TStringBuf scheme, unused;
    if (!url.TrySplit(TStringBuf("://"), scheme, unused))
        return false;

    return AllOf(scheme, IsSchemeChar);
}

TString AddSchemePrefix(const TString& url) {
    return AddSchemePrefix(url, TStringBuf("http"));
}

TString AddSchemePrefix(const TString& url, TStringBuf scheme) {
    if (HasPrefix(url)) {
        return url;
    }

    return TString::Join(scheme, TStringBuf("://"), url);
}

#define X(c) (c >= 'A' ? ((c & 0xdf) - 'A') + 10 : (c - '0'))

static inline int x2c(unsigned char* x) {
    if (!IsAsciiHex(x[0]) || !IsAsciiHex(x[1]))
        return -1;
    return X(x[0]) * 16 + X(x[1]);
}

#undef X

static inline int Unescape(char* str) {
    char *to, *from;
    int dlen = 0;
    if ((str = strchr(str, '%')) == nullptr)
        return dlen;
    for (to = str, from = str; *from; from++, to++) {
        if ((*to = *from) == '%') {
            int c = x2c((unsigned char*)from + 1);
            *to = char((c > 0) ? c : '0');
            from += 2;
            dlen += 2;
        }
    }
    *to = 0; /* terminate it at the new length */
    return dlen;
}

size_t NormalizeUrlName(char* dest, const TStringBuf source, size_t dest_size) {
    if (source.empty() || source[0] == '?')
        return strlcpy(dest, "/", dest_size);
    size_t len = Min(dest_size - 1, source.length());
    memcpy(dest, source.data(), len);
    dest[len] = 0;
    len -= Unescape(dest);
    strlwr(dest);
    return len;
}

size_t NormalizeHostName(char* dest, const TStringBuf source, size_t dest_size, ui16 defport) {
    size_t len = Min(dest_size - 1, source.length());
    memcpy(dest, source.data(), len);
    dest[len] = 0;
    char buf[8] = ":";
    size_t buflen = 1 + ToString(defport, buf + 1, sizeof(buf) - 2);
    buf[buflen] = '\0';
    char* ptr = strstr(dest, buf);
    if (ptr && ptr[buflen] == 0) {
        len -= buflen;
        *ptr = 0;
    }
    strlwr(dest);
    return len;
}

TStringBuf RemoveFinalSlash(TStringBuf str Y_LIFETIME_BOUND) noexcept {
    if (str.EndsWith('/')) {
        str.Chop(1);
    }
    return str;
}

TStringBuf CutUrlPrefixes(TStringBuf url Y_LIFETIME_BOUND) noexcept {
    url = CutSchemePrefix(url);
    url = CutWWWPrefix(url);
    return url;
}

bool DoesUrlPathStartWithToken(TStringBuf url, const TStringBuf token) noexcept {
    url = CutSchemePrefix(url);
    const TStringBuf noHostSuffix = url.After('/');
    if (noHostSuffix == url) {
        // no slash => no suffix with token info
        return false;
    }
    const bool suffixHasPrefix = noHostSuffix.StartsWith(token);
    if (!suffixHasPrefix) {
        return false;
    }
    const bool slashAfterPrefix = noHostSuffix.find("/", token.length()) == token.length();
    const bool qMarkAfterPrefix = noHostSuffix.find("?", token.length()) == token.length();
    const bool nothingAfterPrefix = noHostSuffix.length() <= token.length();
    const bool prefixIsToken = slashAfterPrefix || qMarkAfterPrefix || nothingAfterPrefix;
    return prefixIsToken;
}
