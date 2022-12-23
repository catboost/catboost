#pragma once

#include <util/generic/fwd.h>
#include <util/generic/strbuf.h>

namespace NUrl {

    /**
     * Splits URL to host and path
     * Example:
     * auto [host, path] = SplitUrlToHostAndPath(url);
     *
     * @param[in] url                   any URL
     * @param[out] <host, path>     parsed host and path
     */
    struct TSplitUrlToHostAndPathResult {
        TStringBuf host;
        TStringBuf path;
    };

    Y_PURE_FUNCTION
    TSplitUrlToHostAndPathResult SplitUrlToHostAndPath(const TStringBuf url);

} // namespace NUrl

Y_PURE_FUNCTION
size_t GetHttpPrefixSize(const char* url, bool ignorehttps = false) noexcept;
Y_PURE_FUNCTION
size_t GetHttpPrefixSize(const wchar16* url, bool ignorehttps = false) noexcept;

Y_PURE_FUNCTION
size_t GetHttpPrefixSize(const TStringBuf url, bool ignorehttps = false) noexcept;

Y_PURE_FUNCTION
size_t GetHttpPrefixSize(const TWtringBuf url, bool ignorehttps = false) noexcept;

/** BEWARE of TStringBuf! You can not use operator ~ or c_str() like in TString
    !!!!!!!!!!!! */
Y_PURE_FUNCTION
size_t GetSchemePrefixSize(const TStringBuf url) noexcept;

Y_PURE_FUNCTION
TStringBuf GetSchemePrefix(const TStringBuf url) noexcept;

//! removes protocol prefixes 'http://' and 'https://' from given URL
//! @note if URL has no prefix or some other prefix the function does nothing
//! @param url    URL from which the prefix should be removed
//! @param ignorehttps if true, leaves https://
//! @return a new URL without protocol prefix
Y_PURE_FUNCTION
TStringBuf CutHttpPrefix(const TStringBuf url, bool ignorehttps = false) noexcept;

Y_PURE_FUNCTION
TWtringBuf CutHttpPrefix(const TWtringBuf url, bool ignorehttps = false) noexcept;

Y_PURE_FUNCTION
TStringBuf CutSchemePrefix(const TStringBuf url) noexcept;

//! adds specified scheme prefix if URL has no scheme
//! @note if URL has scheme prefix already the function returns unchanged URL
TString AddSchemePrefix(const TString& url, const TStringBuf scheme);

//! Same as `AddSchemePrefix(url, "http")`.
TString AddSchemePrefix(const TString& url);

Y_PURE_FUNCTION
TStringBuf GetHost(const TStringBuf url) noexcept;

Y_PURE_FUNCTION
TStringBuf GetHostAndPort(const TStringBuf url) noexcept;

Y_PURE_FUNCTION
TStringBuf GetSchemeHost(const TStringBuf url, bool trimHttp = true) noexcept;

Y_PURE_FUNCTION
TStringBuf GetSchemeHostAndPort(const TStringBuf url, bool trimHttp = true, bool trimDefaultPort = true) noexcept;

/**
 * Splits URL to host and path
 *
 * @param[in] url       any URL
 * @param[out] host     parsed host
 * @param[out] path     parsed path
 */
void SplitUrlToHostAndPath(const TStringBuf url, TStringBuf& host, TStringBuf& path);
void SplitUrlToHostAndPath(const TStringBuf url, TString& host, TString& path);

/**
 * Separates URL into url prefix, query (aka cgi params list), and fragment (aka part after #)
 *
 * @param[in] url               any URL
 * @param[out] sanitizedUrl     parsed URL without query and fragment parts
 * @param[out] query            parsed query
 * @param[out] fragment         parsed fragment
 */
void SeparateUrlFromQueryAndFragment(const TStringBuf url, TStringBuf& sanitizedUrl, TStringBuf& query, TStringBuf& fragment);

/**
 * Extracts scheme, host and port from URL.
 *
 * Port will be parsed from URL with checks against ui16 overflow. If URL doesn't
 * contain port it will be determined by one of the known schemes (currently
 * https:// and http:// only).
 * Given parameters will not be modified if URL has no appropriate components.
 *
 * @param[in] url       any URL
 * @param[out] scheme   URL scheme
 * @param[out] host     host name
 * @param[out] port     parsed port number
 * @return false if present port number cannot be parsed into ui16
 *         true  otherwise.
 */
bool TryGetSchemeHostAndPort(const TStringBuf url, TStringBuf& scheme, TStringBuf& host, ui16& port);

/**
 * Extracts scheme, host and port from URL.
 *
 * This function perform the same actions as TryGetSchemeHostAndPort(), but in
 * case of impossibility to parse port number throws yexception.
 *
 * @param[in] url       any URL
 * @param[out] scheme   URL scheme
 * @param[out] host     host name
 * @param[out] port     parsed port number
 * @throws yexception  if present port number cannot be parsed into ui16.
 */
void GetSchemeHostAndPort(const TStringBuf url, TStringBuf& scheme, TStringBuf& host, ui16& port);

Y_PURE_FUNCTION
TStringBuf GetPathAndQuery(const TStringBuf url, bool trimFragment = true) noexcept;
/**
 * Extracts host from url and cuts http(https) protocol prefix and port if any.
 * @param[in] url   any URL
 * @return          host without port and http(https) prefix.
 */
Y_PURE_FUNCTION
TStringBuf GetOnlyHost(const TStringBuf url) noexcept;

Y_PURE_FUNCTION
TStringBuf GetParentDomain(const TStringBuf host, size_t level) noexcept; // ("www.ya.ru", 2) -> "ya.ru"

Y_PURE_FUNCTION
TStringBuf GetZone(const TStringBuf host) noexcept;

Y_PURE_FUNCTION
TStringBuf CutWWWPrefix(const TStringBuf url) noexcept;

Y_PURE_FUNCTION
TStringBuf CutWWWNumberedPrefix(const TStringBuf url) noexcept;

/**
 * Cuts 'm.' prefix from url if and only if the url starts with it
 * Example: 'm.some-domain.com' -> 'some-domain.com'.
 * 'http://m.some-domain.com' is not changed
 *
 * @param[in] url   any URL
 * @return          url without 'm.' or 'M.' prefix.
 */
Y_PURE_FUNCTION
TStringBuf CutMPrefix(const TStringBuf url) noexcept;

Y_PURE_FUNCTION
TStringBuf GetDomain(const TStringBuf host) noexcept; // should not be used

size_t NormalizeUrlName(char* dest, const TStringBuf source, size_t dest_size);
size_t NormalizeHostName(char* dest, const TStringBuf source, size_t dest_size, ui16 defport = 80);

Y_PURE_FUNCTION
TStringBuf RemoveFinalSlash(TStringBuf str) noexcept;

TStringBuf CutUrlPrefixes(TStringBuf url) noexcept;
bool DoesUrlPathStartWithToken(TStringBuf url, const TStringBuf& token) noexcept;

