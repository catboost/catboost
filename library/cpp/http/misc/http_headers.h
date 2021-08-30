#pragma once

#include <util/generic/strbuf.h>


/* Taken from SpringFramework's HttpHeaders. Docs:
 * https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/http/HttpHeaders.html
 * Source:
 * https://github.com/spring-projects/spring-framework/blob/816bbee8de584676250e2bc5dcff6da6cd81623f/spring-web/src/main/java/org/springframework/http/HttpHeaders.java
 */
namespace NHttpHeaders {
    constexpr TStringBuf ACCEPT = "Accept";
    constexpr TStringBuf ACCEPT_CHARSET = "Accept-Charset";
    constexpr TStringBuf ACCEPT_ENCODING = "Accept-Encoding";
    constexpr TStringBuf ACCEPT_LANGUAGE = "Accept-Language";
    constexpr TStringBuf ACCEPT_RANGES = "Accept-Ranges";
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_CREDENTIALS = "Access-Control-Allow-Credentials";
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_HEADERS = "Access-Control-Allow-Headers";
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_METHODS = "Access-Control-Allow-Methods";
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_ORIGIN = "Access-Control-Allow-Origin";
    constexpr TStringBuf ACCESS_CONTROL_EXPOSE_HEADERS = "Access-Control-Expose-Headers";
    constexpr TStringBuf ACCESS_CONTROL_MAX_AGE = "Access-Control-Max-Age";
    constexpr TStringBuf ACCESS_CONTROL_REQUEST_HEADERS = "Access-Control-Request-Headers";
    constexpr TStringBuf ACCESS_CONTROL_REQUEST_METHOD = "Access-Control-Request-Method";
    constexpr TStringBuf AGE = "Age";
    constexpr TStringBuf ALLOW = "Allow";
    constexpr TStringBuf AUTHORIZATION = "Authorization";
    constexpr TStringBuf CACHE_CONTROL = "Cache-Control";
    constexpr TStringBuf CONNECTION = "Connection";
    constexpr TStringBuf CONTENT_ENCODING = "Content-Encoding";
    constexpr TStringBuf CONTENT_DISPOSITION = "Content-Disposition";
    constexpr TStringBuf CONTENT_LANGUAGE = "Content-Language";
    constexpr TStringBuf CONTENT_LENGTH = "Content-Length";
    constexpr TStringBuf CONTENT_LOCATION = "Content-Location";
    constexpr TStringBuf CONTENT_RANGE = "Content-Range";
    constexpr TStringBuf CONTENT_TYPE = "Content-Type";
    constexpr TStringBuf COOKIE = "Cookie";
    constexpr TStringBuf DATE = "Date";
    constexpr TStringBuf ETAG = "ETag";
    constexpr TStringBuf EXPECT = "Expect";
    constexpr TStringBuf EXPIRES = "Expires";
    constexpr TStringBuf FROM = "From";
    constexpr TStringBuf HOST = "Host";
    constexpr TStringBuf IF_MATCH = "If-Match";
    constexpr TStringBuf IF_MODIFIED_SINCE = "If-Modified-Since";
    constexpr TStringBuf IF_NONE_MATCH = "If-None-Match";
    constexpr TStringBuf IF_RANGE = "If-Range";
    constexpr TStringBuf IF_UNMODIFIED_SINCE = "If-Unmodified-Since";
    constexpr TStringBuf LAST_MODIFIED = "Last-Modified";
    constexpr TStringBuf LINK = "Link";
    constexpr TStringBuf LOCATION = "Location";
    constexpr TStringBuf MAX_FORWARDS = "Max-Forwards";
    constexpr TStringBuf ORIGIN = "Origin";
    constexpr TStringBuf PRAGMA = "Pragma";
    constexpr TStringBuf PROXY_AUTHENTICATE = "Proxy-Authenticate";
    constexpr TStringBuf PROXY_AUTHORIZATION = "Proxy-Authorization";
    constexpr TStringBuf RANGE = "Range";
    constexpr TStringBuf REFERER = "Referer";
    constexpr TStringBuf RETRY_AFTER = "Retry-After";
    constexpr TStringBuf SERVER = "Server";
    constexpr TStringBuf SET_COOKIE = "Set-Cookie";
    constexpr TStringBuf SET_COOKIE2 = "Set-Cookie2";
    constexpr TStringBuf TE = "TE";
    constexpr TStringBuf TRAILER = "Trailer";
    constexpr TStringBuf TRANSFER_ENCODING = "Transfer-Encoding";
    constexpr TStringBuf UPGRADE = "Upgrade";
    constexpr TStringBuf USER_AGENT = "User-Agent";
    constexpr TStringBuf VARY = "Vary";
    constexpr TStringBuf VIA = "Via";
    constexpr TStringBuf WARNING = "Warning";
    constexpr TStringBuf WWW_AUTHENTICATE = "WWW-Authenticate";
} // namespace HttpHeaders
