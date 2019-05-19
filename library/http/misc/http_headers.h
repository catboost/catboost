#pragma once

#include <util/generic/strbuf.h>


/* Taken from SpringFramework's HttpHeaders. Docs:
 * https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/http/HttpHeaders.html
 * Source:
 * https://github.com/spring-projects/spring-framework/blob/816bbee8de584676250e2bc5dcff6da6cd81623f/spring-web/src/main/java/org/springframework/http/HttpHeaders.java
 */
namespace NHttpHeaders {
    constexpr TStringBuf ACCEPT = AsStringBuf("Accept");
    constexpr TStringBuf ACCEPT_CHARSET = AsStringBuf("Accept-Charset");
    constexpr TStringBuf ACCEPT_ENCODING = AsStringBuf("Accept-Encoding");
    constexpr TStringBuf ACCEPT_LANGUAGE = AsStringBuf("Accept-Language");
    constexpr TStringBuf ACCEPT_RANGES = AsStringBuf("Accept-Ranges");
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_CREDENTIALS = AsStringBuf("Access-Control-Allow-Credentials");
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_HEADERS = AsStringBuf("Access-Control-Allow-Headers");
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_METHODS = AsStringBuf("Access-Control-Allow-Methods");
    constexpr TStringBuf ACCESS_CONTROL_ALLOW_ORIGIN = AsStringBuf("Access-Control-Allow-Origin");
    constexpr TStringBuf ACCESS_CONTROL_EXPOSE_HEADERS = AsStringBuf("Access-Control-Expose-Headers");
    constexpr TStringBuf ACCESS_CONTROL_MAX_AGE = AsStringBuf("Access-Control-Max-Age");
    constexpr TStringBuf ACCESS_CONTROL_REQUEST_HEADERS = AsStringBuf("Access-Control-Request-Headers");
    constexpr TStringBuf ACCESS_CONTROL_REQUEST_METHOD = AsStringBuf("Access-Control-Request-Method");
    constexpr TStringBuf AGE = AsStringBuf("Age");
    constexpr TStringBuf ALLOW = AsStringBuf("Allow");
    constexpr TStringBuf AUTHORIZATION = AsStringBuf("Authorization");
    constexpr TStringBuf CACHE_CONTROL = AsStringBuf("Cache-Control");
    constexpr TStringBuf CONNECTION = AsStringBuf("Connection");
    constexpr TStringBuf CONTENT_ENCODING = AsStringBuf("Content-Encoding");
    constexpr TStringBuf CONTENT_DISPOSITION = AsStringBuf("Content-Disposition");
    constexpr TStringBuf CONTENT_LANGUAGE = AsStringBuf("Content-Language");
    constexpr TStringBuf CONTENT_LENGTH = AsStringBuf("Content-Length");
    constexpr TStringBuf CONTENT_LOCATION = AsStringBuf("Content-Location");
    constexpr TStringBuf CONTENT_RANGE = AsStringBuf("Content-Range");
    constexpr TStringBuf CONTENT_TYPE = AsStringBuf("Content-Type");
    constexpr TStringBuf COOKIE = AsStringBuf("Cookie");
    constexpr TStringBuf DATE = AsStringBuf("Date");
    constexpr TStringBuf ETAG = AsStringBuf("ETag");
    constexpr TStringBuf EXPECT = AsStringBuf("Expect");
    constexpr TStringBuf EXPIRES = AsStringBuf("Expires");
    constexpr TStringBuf FROM = AsStringBuf("From");
    constexpr TStringBuf HOST = AsStringBuf("Host");
    constexpr TStringBuf IF_MATCH = AsStringBuf("If-Match");
    constexpr TStringBuf IF_MODIFIED_SINCE = AsStringBuf("If-Modified-Since");
    constexpr TStringBuf IF_NONE_MATCH = AsStringBuf("If-None-Match");
    constexpr TStringBuf IF_RANGE = AsStringBuf("If-Range");
    constexpr TStringBuf IF_UNMODIFIED_SINCE = AsStringBuf("If-Unmodified-Since");
    constexpr TStringBuf LAST_MODIFIED = AsStringBuf("Last-Modified");
    constexpr TStringBuf LINK = AsStringBuf("Link");
    constexpr TStringBuf LOCATION = AsStringBuf("Location");
    constexpr TStringBuf MAX_FORWARDS = AsStringBuf("Max-Forwards");
    constexpr TStringBuf ORIGIN = AsStringBuf("Origin");
    constexpr TStringBuf PRAGMA = AsStringBuf("Pragma");
    constexpr TStringBuf PROXY_AUTHENTICATE = AsStringBuf("Proxy-Authenticate");
    constexpr TStringBuf PROXY_AUTHORIZATION = AsStringBuf("Proxy-Authorization");
    constexpr TStringBuf RANGE = AsStringBuf("Range");
    constexpr TStringBuf REFERER = AsStringBuf("Referer");
    constexpr TStringBuf RETRY_AFTER = AsStringBuf("Retry-After");
    constexpr TStringBuf SERVER = AsStringBuf("Server");
    constexpr TStringBuf SET_COOKIE = AsStringBuf("Set-Cookie");
    constexpr TStringBuf SET_COOKIE2 = AsStringBuf("Set-Cookie2");
    constexpr TStringBuf TE = AsStringBuf("TE");
    constexpr TStringBuf TRAILER = AsStringBuf("Trailer");
    constexpr TStringBuf TRANSFER_ENCODING = AsStringBuf("Transfer-Encoding");
    constexpr TStringBuf UPGRADE = AsStringBuf("Upgrade");
    constexpr TStringBuf USER_AGENT = AsStringBuf("User-Agent");
    constexpr TStringBuf VARY = AsStringBuf("Vary");
    constexpr TStringBuf VIA = AsStringBuf("Via");
    constexpr TStringBuf WARNING = AsStringBuf("Warning");
    constexpr TStringBuf WWW_AUTHENTICATE = AsStringBuf("WWW-Authenticate");
} // namespace HttpHeaders
