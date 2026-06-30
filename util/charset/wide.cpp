#include "wide.h"

#include <util/generic/mem_copy.h>
#include <util/string/strip.h>

namespace {
    //! the constants are not zero-terminated
    const wchar16 LT[] = {'&', 'l', 't', ';'};
    const wchar16 GT[] = {'&', 'g', 't', ';'};
    const wchar16 AMP[] = {'&', 'a', 'm', 'p', ';'};
    const wchar16 BR[] = {'<', 'B', 'R', '>'};
    const wchar16 QUOT[] = {'&', 'q', 'u', 'o', 't', ';'};

    template <bool insertBr>
    inline size_t EscapedLen(wchar16 c) {
        switch (c) {
            case '<':
                return Y_ARRAY_SIZE(LT);
            case '>':
                return Y_ARRAY_SIZE(GT);
            case '&':
                return Y_ARRAY_SIZE(AMP);
            case '\"':
                return Y_ARRAY_SIZE(QUOT);
            default:
                if (insertBr && (c == '\r' || c == '\n')) {
                    return Y_ARRAY_SIZE(BR);
                } else {
                    return 1;
                }
        }
    }
} // namespace

void Collapse(TUtf16String& w) {
    CollapseImpl(w, w, 0, IsWhitespace);
}

size_t Collapse(wchar16* s, size_t n) {
    return CollapseImpl(s, n, IsWhitespace);
}

TWtringBuf StripLeft(const TWtringBuf text) noexcept {
    const auto* p = text.data();
    const auto* const pe = text.data() + text.size();

    for (; p != pe && IsWhitespace(*p); ++p) {
    }

    return {p, pe};
}

void StripLeft(TUtf16String& text) {
    const auto stripped = StripLeft(TWtringBuf(text));
    if (stripped.size() == text.size()) {
        return;
    }

    text = stripped;
}

TWtringBuf StripRight(const TWtringBuf text) noexcept {
    if (!text) {
        return {};
    }

    const auto* const pe = text.data() - 1;
    const auto* p = text.data() + text.size() - 1;

    for (; p != pe && IsWhitespace(*p); --p) {
    }

    return {pe + 1, p + 1};
}

void StripRight(TUtf16String& text) {
    const auto stripped = StripRight(TWtringBuf(text));
    if (stripped.size() == text.size()) {
        return;
    }

    text.resize(stripped.size());
}

TWtringBuf Strip(const TWtringBuf text) noexcept {
    return StripRight(StripLeft(text));
}

void Strip(TUtf16String& text) {
    StripLeft(text);
    StripRight(text);
}

template <typename T>
static bool IsReductionOnSymbolsTrue(const TWtringBuf text, T&& f) {
    const auto* p = text.data();
    const auto* const pe = text.data() + text.length();
    while (p != pe) {
        const auto symbol = ReadSymbolAndAdvance(p, pe);
        if (!f(symbol)) {
            return false;
        }
    }

    return true;
}

bool IsLowerWord(const TWtringBuf text) noexcept {
    return IsReductionOnSymbolsTrue(text, [](const wchar32 s) { return IsLower(s); });
}

bool IsUpperWord(const TWtringBuf text) noexcept {
    return IsReductionOnSymbolsTrue(text, [](const wchar32 s) { return IsUpper(s); });
}

bool IsLower(const TWtringBuf text) noexcept {
    return IsReductionOnSymbolsTrue(text, [](const wchar32 s) {
        if (IsAlpha(s)) {
            return IsLower(s);
        }
        return true;
    });
}

bool IsUpper(const TWtringBuf text) noexcept {
    return IsReductionOnSymbolsTrue(text, [](const wchar32 s) {
        if (IsAlpha(s)) {
            return IsUpper(s);
        }
        return true;
    });
}

bool IsTitleWord(const TWtringBuf text) noexcept {
    if (!text) {
        return false;
    }

    const auto* p = text.data();
    const auto* pe = text.data() + text.size();

    const auto firstSymbol = ReadSymbolAndAdvance(p, pe);
    if (firstSymbol != ToTitle(firstSymbol)) {
        return false;
    }

    return IsLowerWord({p, pe});
}

template <bool stopOnFirstModification, typename TCharType, typename F>
static bool ModifySequence(TCharType*& p, const TCharType* const pe, F&& f) {
    while (p != pe) {
        const auto symbol = ReadSymbol(p, pe);
        const auto modified = f(symbol);
        if (symbol != modified) {
            if (stopOnFirstModification) {
                return true;
            }

            WriteSymbol(modified, p); // also moves `p` forward
        } else {
            p = SkipSymbol(p, pe);
        }
    }

    return false;
}

template <bool stopOnFirstModification, typename TCharType, typename F>
static bool ModifySequence(const TCharType*& p, const TCharType* const pe, TCharType*& out, F&& f) {
    while (p != pe) {
        const auto symbol = stopOnFirstModification ? ReadSymbol(p, pe) : ReadSymbolAndAdvance(p, pe);
        const auto modified = f(symbol);

        if (stopOnFirstModification) {
            if (symbol != modified) {
                return true;
            }

            p = SkipSymbol(p, pe);
        }

        WriteSymbol(modified, out);
    }

    return false;
}

template <class TStringType>
static void DetachAndFixPointers(TStringType& text, typename TStringType::value_type*& p, const typename TStringType::value_type*& pe) {
    const auto pos = p - text.data();
    const auto count = pe - p;
    p = text.Detach() + pos;
    pe = p + count;
}

template <class TStringType, typename F>
static bool ModifyStringSymbolwise(TStringType& text, size_t pos, size_t count, F&& f) {
    // TODO(yazevnul): this is done for consistency with `TUtf16String::to_lower` and friends
    // at r2914050, maybe worth replacing them with asserts. Also see the same code in `ToTitle`.
    pos = pos < text.size() ? pos : text.size();
    count = count < text.size() - pos ? count : text.size() - pos;

    // TUtf16String is refcounted and it's `data` method return pointer to the constant memory.
    // To simplify the code we do a `const_cast`, though first write to the memory will be done only
    // after we call `Detach()` and get pointer to a writable piece of memory.
    auto* p = const_cast<typename TStringType::value_type*>(text.data() + pos);
    const auto* pe = text.data() + pos + count;

    if (ModifySequence<true>(p, pe, f)) {
        DetachAndFixPointers(text, p, pe);
        ModifySequence<false>(p, pe, f);
        return true;
    }

    return false;
}

bool ToLower(TUtf16String& text, size_t pos, size_t count) {
    const auto f = [](const wchar32 s) { return ToLower(s); };
    return ModifyStringSymbolwise(text, pos, count, f);
}

bool ToUpper(TUtf16String& text, size_t pos, size_t count) {
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    return ModifyStringSymbolwise(text, pos, count, f);
}

bool ToLower(TUtf32String& text, size_t pos, size_t count) {
    const auto f = [](const wchar32 s) { return ToLower(s); };
    return ModifyStringSymbolwise(text, pos, count, f);
}

bool ToUpper(TUtf32String& text, size_t pos, size_t count) {
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    return ModifyStringSymbolwise(text, pos, count, f);
}

bool ToTitle(TUtf16String& text, size_t pos, size_t count) {
    if (!text) {
        return false;
    }

    pos = pos < text.size() ? pos : text.size();
    count = count < text.size() - pos ? count : text.size() - pos;

    const auto toLower = [](const wchar32 s) { return ToLower(s); };

    auto* p = const_cast<wchar16*>(text.data() + pos);
    const auto* pe = text.data() + pos + count;

    const auto firstSymbol = ReadSymbol(p, pe);
    if (firstSymbol == ToTitle(firstSymbol)) {
        p = SkipSymbol(p, pe);
        if (ModifySequence<true>(p, pe, toLower)) {
            DetachAndFixPointers(text, p, pe);
            ModifySequence<false>(p, pe, toLower);
            return true;
        }
    } else {
        DetachAndFixPointers(text, p, pe);
        WriteSymbol(ToTitle(ReadSymbol(p, pe)), p); // also moves `p` forward
        ModifySequence<false>(p, pe, toLower);
        return true;
    }

    return false;
}

bool ToTitle(TUtf32String& text, size_t pos, size_t count) {
    if (!text) {
        return false;
    }

    pos = pos < text.size() ? pos : text.size();
    count = count < text.size() - pos ? count : text.size() - pos;

    const auto toLower = [](const wchar32 s) { return ToLower(s); };

    auto* p = const_cast<wchar32*>(text.data() + pos);
    const auto* pe = text.data() + pos + count;

    const auto firstSymbol = *p;
    if (firstSymbol == ToTitle(firstSymbol)) {
        p += 1;
        if (ModifySequence<true>(p, pe, toLower)) {
            DetachAndFixPointers(text, p, pe);
            ModifySequence<false>(p, pe, toLower);
            return true;
        }
    } else {
        DetachAndFixPointers(text, p, pe);
        WriteSymbol(ToTitle(ReadSymbol(p, pe)), p); // also moves `p` forward
        ModifySequence<false>(p, pe, toLower);
        return true;
    }

    return false;
}

TUtf16String ToLowerRet(TUtf16String text, size_t pos, size_t count) {
    ToLower(text, pos, count);
    return text;
}

TUtf16String ToUpperRet(TUtf16String text, size_t pos, size_t count) {
    ToUpper(text, pos, count);
    return text;
}

TUtf16String ToTitleRet(TUtf16String text, size_t pos, size_t count) {
    ToTitle(text, pos, count);
    return text;
}

TUtf32String ToLowerRet(TUtf32String text, size_t pos, size_t count) {
    ToLower(text, pos, count);
    return text;
}

TUtf32String ToUpperRet(TUtf32String text, size_t pos, size_t count) {
    ToUpper(text, pos, count);
    return text;
}

TUtf32String ToTitleRet(TUtf32String text, size_t pos, size_t count) {
    ToTitle(text, pos, count);
    return text;
}

bool ToLower(const wchar16* text, size_t length, wchar16* out) noexcept {
    // TODO(yazevnul): get rid of `text == out` case (it is probably used only in lemmer) and then
    // we can declare text and out as `__restrict__`
    Y_ASSERT(text == out || !(out >= text && out < text + length));
    const auto f = [](const wchar32 s) { return ToLower(s); };
    const auto* p = text;
    const auto* const pe = text + length;
    if (ModifySequence<true>(p, pe, out, f)) {
        ModifySequence<false>(p, pe, out, f);
        return true;
    }
    return false;
}

bool ToUpper(const wchar16* text, size_t length, wchar16* out) noexcept {
    Y_ASSERT(text == out || !(out >= text && out < text + length));
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    const auto* p = text;
    const auto* const pe = text + length;
    if (ModifySequence<true>(p, pe, out, f)) {
        ModifySequence<false>(p, pe, out, f);
        return true;
    }
    return false;
}

bool ToTitle(const wchar16* text, size_t length, wchar16* out) noexcept {
    if (!length) {
        return false;
    }

    Y_ASSERT(text == out || !(out >= text && out < text + length));

    const auto* const textEnd = text + length;
    const auto firstSymbol = ReadSymbolAndAdvance(text, textEnd);
    const auto firstSymbolTitle = ToTitle(firstSymbol);

    WriteSymbol(firstSymbolTitle, out);

    return ToLower(text, textEnd - text, out) || firstSymbol != firstSymbolTitle;
}

bool ToLower(wchar16* text, size_t length) noexcept {
    const auto f = [](const wchar32 s) { return ToLower(s); };
    const auto* const textEnd = text + length;
    if (ModifySequence<true>(text, textEnd, f)) {
        ModifySequence<false>(text, textEnd, f);
        return true;
    }
    return false;
}

bool ToUpper(wchar16* text, size_t length) noexcept {
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    const auto* const textEnd = text + length;
    if (ModifySequence<true>(text, textEnd, f)) {
        ModifySequence<false>(text, textEnd, f);
        return true;
    }
    return false;
}

bool ToTitle(wchar16* text, size_t length) noexcept {
    if (!length) {
        return false;
    }

    const auto* textEnd = text + length;
    const auto firstSymbol = ReadSymbol(text, textEnd);
    const auto firstSymbolTitle = ToTitle(firstSymbol);

    // avoid unnacessary writes to the memory
    if (firstSymbol != firstSymbolTitle) {
        WriteSymbol(firstSymbolTitle, text);
    } else {
        text = SkipSymbol(text, textEnd);
    }

    return ToLower(text, textEnd - text) || firstSymbol != firstSymbolTitle;
}

bool ToLower(const wchar32* text, size_t length, wchar32* out) noexcept {
    // TODO(yazevnul): get rid of `text == out` case (it is probably used only in lemmer) and then
    // we can declare text and out as `__restrict__`
    Y_ASSERT(text == out || !(out >= text && out < text + length));
    const auto f = [](const wchar32 s) { return ToLower(s); };
    const auto* p = text;
    const auto* const pe = text + length;
    if (ModifySequence<true>(p, pe, out, f)) {
        ModifySequence<false>(p, pe, out, f);
        return true;
    }
    return false;
}

bool ToUpper(const wchar32* text, size_t length, wchar32* out) noexcept {
    Y_ASSERT(text == out || !(out >= text && out < text + length));
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    const auto* p = text;
    const auto* const pe = text + length;
    if (ModifySequence<true>(p, pe, out, f)) {
        ModifySequence<false>(p, pe, out, f);
        return true;
    }
    return false;
}

bool ToTitle(const wchar32* text, size_t length, wchar32* out) noexcept {
    if (!length) {
        return false;
    }

    Y_ASSERT(text == out || !(out >= text && out < text + length));

    const auto* const textEnd = text + length;
    const auto firstSymbol = ReadSymbolAndAdvance(text, textEnd);
    const auto firstSymbolTitle = ToTitle(firstSymbol);

    WriteSymbol(firstSymbolTitle, out);

    return ToLower(text, textEnd - text, out) || firstSymbol != firstSymbolTitle;
}

bool ToLower(wchar32* text, size_t length) noexcept {
    const auto f = [](const wchar32 s) { return ToLower(s); };
    const auto* const textEnd = text + length;
    if (ModifySequence<true>(text, textEnd, f)) {
        ModifySequence<false>(text, textEnd, f);
        return true;
    }
    return false;
}

bool ToUpper(wchar32* text, size_t length) noexcept {
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    const auto* const textEnd = text + length;
    if (ModifySequence<true>(text, textEnd, f)) {
        ModifySequence<false>(text, textEnd, f);
        return true;
    }
    return false;
}

bool ToTitle(wchar32* text, size_t length) noexcept {
    if (!length) {
        return false;
    }

    const auto* textEnd = text + length;
    const auto firstSymbol = ReadSymbol(text, textEnd);
    const auto firstSymbolTitle = ToTitle(firstSymbol);

    // avoid unnacessary writes to the memory
    if (firstSymbol != firstSymbolTitle) {
        WriteSymbol(firstSymbolTitle, text);
    } else {
        text = SkipSymbol(text, textEnd);
    }

    return ToLower(text, textEnd - text) || firstSymbol != firstSymbolTitle;
}

template <typename F>
static TUtf16String ToSmthRet(const TWtringBuf text, size_t pos, size_t count, F&& f) {
    pos = pos < text.size() ? pos : text.size();
    count = count < text.size() - pos ? count : text.size() - pos;

    auto res = TUtf16String::Uninitialized(text.size());
    auto* const resBegin = res.Detach();

    if (pos) {
        MemCopy(resBegin, text.data(), pos);
    }

    f(text.data() + pos, count, resBegin + pos);

    if (count - pos != text.size()) {
        MemCopy(resBegin + pos + count, text.data() + pos + count, text.size() - pos - count);
    }

    return res;
}

template <typename F>
static TUtf32String ToSmthRet(const TUtf32StringBuf text, size_t pos, size_t count, F&& f) {
    pos = pos < text.size() ? pos : text.size();
    count = count < text.size() - pos ? count : text.size() - pos;

    auto res = TUtf32String::Uninitialized(text.size());
    auto* const resBegin = res.Detach();

    if (pos) {
        MemCopy(resBegin, text.data(), pos);
    }

    f(text.data() + pos, count, resBegin + pos);

    if (count - pos != text.size()) {
        MemCopy(resBegin + pos + count, text.data() + pos + count, text.size() - pos - count);
    }

    return res;
}

TUtf16String ToLowerRet(const TWtringBuf text, size_t pos, size_t count) {
    return ToSmthRet(text, pos, count, [](const wchar16* theText, size_t length, wchar16* out) {
        ToLower(theText, length, out);
    });
}

TUtf16String ToUpperRet(const TWtringBuf text, size_t pos, size_t count) {
    return ToSmthRet(text, pos, count, [](const wchar16* theText, size_t length, wchar16* out) {
        ToUpper(theText, length, out);
    });
}

TUtf16String ToTitleRet(const TWtringBuf text, size_t pos, size_t count) {
    return ToSmthRet(text, pos, count, [](const wchar16* theText, size_t length, wchar16* out) {
        ToTitle(theText, length, out);
    });
}

TUtf32String ToLowerRet(const TUtf32StringBuf text, size_t pos, size_t count) {
    return ToSmthRet(text, pos, count, [](const wchar32* theText, size_t length, wchar32* out) {
        ToLower(theText, length, out);
    });
}

TUtf32String ToUpperRet(const TUtf32StringBuf text, size_t pos, size_t count) {
    return ToSmthRet(text, pos, count, [](const wchar32* theText, size_t length, wchar32* out) {
        ToUpper(theText, length, out);
    });
}

TUtf32String ToTitleRet(const TUtf32StringBuf text, size_t pos, size_t count) {
    return ToSmthRet(text, pos, count, [](const wchar32* theText, size_t length, wchar32* out) {
        ToTitle(theText, length, out);
    });
}

template <bool insertBr>
void EscapeHtmlChars(TUtf16String& str) {
    static const TUtf16String lt(LT, Y_ARRAY_SIZE(LT));
    static const TUtf16String gt(GT, Y_ARRAY_SIZE(GT));
    static const TUtf16String amp(AMP, Y_ARRAY_SIZE(AMP));
    static const TUtf16String br(BR, Y_ARRAY_SIZE(BR));
    static const TUtf16String quot(QUOT, Y_ARRAY_SIZE(QUOT));

    size_t escapedLen = 0;

    const TUtf16String& cs = str;

    for (size_t i = 0; i < cs.size(); ++i) {
        escapedLen += EscapedLen<insertBr>(cs[i]);
    }

    if (escapedLen == cs.size()) {
        return;
    }

    TUtf16String res;
    res.reserve(escapedLen);

    size_t start = 0;

    for (size_t i = 0; i < cs.size(); ++i) {
        const TUtf16String* ent = nullptr;
        switch (cs[i]) {
            case '<':
                ent = &lt;
                break;
            case '>':
                ent = &gt;
                break;
            case '&':
                ent = &amp;
                break;
            case '\"':
                ent = &quot;
                break;
            default:
                if (insertBr && (cs[i] == '\r' || cs[i] == '\n')) {
                    ent = &br;
                    break;
                } else {
                    continue;
                }
        }

        res.append(cs.begin() + start, cs.begin() + i);
        res.append(ent->begin(), ent->end());
        start = i + 1;
    }

    res.append(cs.begin() + start, cs.end());
    res.swap(str);
}

template void EscapeHtmlChars<false>(TUtf16String& str);
template void EscapeHtmlChars<true>(TUtf16String& str);
