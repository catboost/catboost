#pragma once

#include <util/generic/strbuf.h>
#include <util/generic/string.h>

//CGIEscape*:
// ' ' converted to '+',
// Some punctuation and chars outside [32, 126] range are converted to %xx
// Use function CgiEscapeBufLen to determine number of characters needed for 'char* to' parameter.
// Returns pointer to the end of the result string
char* CGIEscape(char* to, const char* from);
char* CGIEscape(char* to, const char* from, size_t len);
inline char* CGIEscape(char* to, const TStringBuf from) {
    return CGIEscape(to, from.data(), from.size());
}
void CGIEscape(TString& url);
TString CGIEscapeRet(const TStringBuf url);
TString& AppendCgiEscaped(const TStringBuf value, TString& to);

inline TStringBuf CgiEscapeBuf(char* to, const TStringBuf from) {
    return TStringBuf(to, CGIEscape(to, from.data(), from.size()));
}
inline TStringBuf CgiEscape(void* tmp, const TStringBuf s) {
    return CgiEscapeBuf(static_cast<char*>(tmp), s);
}

//CgiUnescape*:
// Decodes '%xx' to bytes, '+' to space.
// Use function CgiUnescapeBufLen to determine number of characters needed for 'char* to' parameter.
// If pointer returned, then this is pointer to the end of the result string.
char* CGIUnescape(char* to, const char* from);
char* CGIUnescape(char* to, const char* from, size_t len);
void CGIUnescape(TString& url);
TString CGIUnescapeRet(const TStringBuf from);

inline TStringBuf CgiUnescapeBuf(char* to, const TStringBuf from) {
    return TStringBuf(to, CGIUnescape(to, from.data(), from.size()));
}
inline TStringBuf CgiUnescape(void* tmp, const TStringBuf s) {
    return CgiUnescapeBuf(static_cast<char*>(tmp), s);
}

//Quote:
// Is like CGIEscape, also skips encoding of user-supplied 'safe' characters.
char* Quote(char* to, const char* from, const char* safe = "/");
char* Quote(char* to, const TStringBuf s, const char* safe = "/");
void Quote(TString& url, const char* safe = "/");

//UrlEscape:
// Can't be used for cgi parameters ('&' character is not escaped)!
// escapes only '%' not followed by two hex-digits or if forceEscape set to ture,
// and chars outside [32, 126] range.
// Can't handle '\0'-chars in TString.
char* UrlEscape(char* to, const char* from, bool forceEscape = false);
void UrlEscape(TString& url, bool forceEscape = false);
TString UrlEscapeRet(const TStringBuf from, bool forceEscape = false);

//UrlUnescape:
// '+' is NOT converted to space!
// %xx converted to bytes, other characters are copied unchanged.
char* UrlUnescape(char* to, TStringBuf from);
void UrlUnescape(TString& url);
TString UrlUnescapeRet(const TStringBuf from);

//*BufLen: how much characters you should allocate for 'char* to' buffers.
constexpr size_t CgiEscapeBufLen(const size_t len) noexcept {
    return 3 * len + 1;
}

constexpr size_t CgiUnescapeBufLen(const size_t len) noexcept {
    return len + 1;
}
