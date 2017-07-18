#pragma once

#include <util/generic/fwd.h>

size_t GetHttpPrefixSize(const char* url, bool ignorehttps = false);
size_t GetHttpPrefixSize(const wchar16* url, bool ignorehttps = false);

size_t GetHttpPrefixSize(const TStringBuf url, bool ignorehttps = false);
size_t GetHttpPrefixSize(const TWtringBuf url, bool ignorehttps = false);

/** BEWARE of TStringBuf! You can not use operator ~ or c_str() like in TString
    !!!!!!!!!!!! */

size_t GetSchemePrefixSize(const TStringBuf url);

TStringBuf GetSchemePrefix(const TStringBuf url);

//! removes protocol prefixes 'http://' and 'https://' from given URL
//! @note if URL has no prefix or some other prefix the function does nothing
//! @param url    URL from which the prefix should be removed
//! @param ignorehttps if true, leaves https://
//! @return a new URL without protocol prefix
TStringBuf CutHttpPrefix(const TStringBuf url, bool ignorehttps = false);
TWtringBuf CutHttpPrefix(const TWtringBuf url, bool ignorehttps = false);

TStringBuf CutSchemePrefix(const TStringBuf url);

//! adds specified scheme prefix if URL has no scheme
//! @note if URL has scheme prefix already the function returns unchanged URL
TString AddSchemePrefix(const TString& url, TString scheme);

//! Same as `AddSchemePrefix(url, "http")`.
TString AddSchemePrefix(const TString& url);

TStringBuf GetHost(const TStringBuf url);
TStringBuf GetHostAndPort(const TStringBuf url);
TStringBuf GetSchemeHostAndPort(const TStringBuf url, bool trimHttp = true, bool trimDefaultPort = true);

TStringBuf GetPathAndQuery(const TStringBuf url, bool trimFragment = true);
/**
 * Extracts host from url and cuts http(https) protocol prefix and port if any.
 * @param[in] url   any URL
 * @return          host without port and http(https) prefix.
 */
TStringBuf GetOnlyHost(const TStringBuf url);
TStringBuf GetParentDomain(const TStringBuf host, size_t level); // ("www.ya.ru", 2) -> "ya.ru"
TStringBuf GetZone(const TStringBuf host);
TStringBuf CutWWWPrefix(const TStringBuf url);
TStringBuf GetDomain(const TStringBuf host); // should not be used

size_t NormalizeUrlName(char* dest, const TStringBuf source, size_t dest_size);
size_t NormalizeHostName(char* dest, const TStringBuf source, size_t dest_size, ui16 defport = 80);
