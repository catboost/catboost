# Relaxed escaper

`EscapeJ` performs byte-wise C-style escaping. Despite its name, it preserves UTF-8 and may emit `\xXX` or octal escapes that are not valid JSON.

`EscapeJsonStringToAscii` decodes UTF-8 and returns string contents without surrounding quotes. It escapes JSON special characters, represents every non-ASCII code point with `\uXXXX`, and uses UTF-16 surrogate pairs above `U+FFFF`. Invalid UTF-8 bytes are escaped individually as `\u00XX`. For compatibility with `Poco::UTF8::escape`, vertical tab and bell use the non-JSON escapes `\v` and `\a`.

```cpp
NEscJ::EscapeJ<false, true>("Привет");
// Привет

NEscJ::EscapeJsonStringToAscii("Привет 🪏");
// \u041F\u0440\u0438\u0432\u0435\u0442 \uD83E\uDE8F
```
