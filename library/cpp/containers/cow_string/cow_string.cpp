#include "cow_string.h"

#include <util/string/ascii.h>
#include <util/system/sanitizers.h>
#include <util/system/sys_alloc.h>
#include <util/charset/wide.h>

#include <iostream>

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

std::ostream& operator<<(std::ostream& os, const TCowString& s) {
    return os.write(s.data(), s.size());
}

std::istream& operator>>(std::istream& is, TCowString& s) {
    return is >> s.MutRef();
}

template <>
bool TBasicCowString<char, std::char_traits<char>>::to_lower(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToLower(c); }, pos, n);
}

template <>
bool TBasicCowString<char, std::char_traits<char>>::to_upper(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToUpper(c); }, pos, n);
}

template <>
bool TBasicCowString<char, std::char_traits<char>>::to_title(size_t pos, size_t n) {
    if (n == 0) {
        return false;
    }
    bool changed = to_upper(pos, 1);
    return to_lower(pos + 1, n - 1) || changed;
}

template <>
TUtf16CowString&
TBasicCowString<wchar16, std::char_traits<wchar16>>::AppendAscii(const ::TStringBuf& s) {
    ReserveAndResize(size() + s.size());

    auto dst = begin() + size() - s.size();

    for (const char* src = s.data(); dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar16>(*src);
    }

    return *this;
}

template <>
TUtf16CowString&
TBasicCowString<wchar16, std::char_traits<wchar16>>::AppendUtf8(const ::TStringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.data(), s.size(), begin() + oldSize, written);
    if (pos != s.size()) {
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.data(), s.size());
    }
    resize(oldSize + written);

    return *this;
}

template <>
bool TBasicCowString<wchar16, std::char_traits<wchar16>>::to_lower(size_t pos, size_t n) {
    const auto f = [](const wchar32 s) { return ToLower(s); };
    return ModifyStringSymbolwise(*this, pos, n, f);
}

template <>
bool TBasicCowString<wchar16, std::char_traits<wchar16>>::to_upper(size_t pos, size_t n) {
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    return ModifyStringSymbolwise(*this, pos, n, f);
}

template <>
bool TBasicCowString<wchar16, std::char_traits<wchar16>>::to_title(size_t pos, size_t nn) {
    if (!*this) {
        return false;
    }

    pos = pos < this->size() ? pos : this->size();
    nn = nn < this->size() - pos ? nn : this->size() - pos;

    const auto toLower = [](const wchar32 s) { return ToLower(s); };

    auto* p = const_cast<wchar16*>(this->data() + pos);
    const auto* pe = this->data() + pos + nn;

    const auto firstSymbol = ReadSymbol(p, pe);
    if (firstSymbol == ToTitle(firstSymbol)) {
        p = SkipSymbol(p, pe);
        if (ModifySequence<true>(p, pe, toLower)) {
            DetachAndFixPointers(*this, p, pe);
            ModifySequence<false>(p, pe, toLower);
            return true;
        }
    } else {
        DetachAndFixPointers(*this, p, pe);
        WriteSymbol(ToTitle(ReadSymbol(p, pe)), p); // also moves `p` forward
        ModifySequence<false>(p, pe, toLower);
        return true;
    }

    return false;
}

template <>
TUtf32CowString&
TBasicCowString<wchar32, std::char_traits<wchar32>>::AppendAscii(const ::TStringBuf& s) {
    ReserveAndResize(size() + s.size());

    auto dst = begin() + size() - s.size();

    for (const char* src = s.data(); dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar32>(*src);
    }

    return *this;
}

template <>
TBasicCowString<char, std::char_traits<char>>&
TBasicCowString<char, std::char_traits<char>>::AppendUtf16(const ::TWtringBuf& s) {
    const size_t oldSize = size();
    ReserveAndResize(size() + WideToUTF8BufferSize(s.size()));

    size_t written = 0;
    WideToUTF8(s.data(), s.size(), begin() + oldSize, written);

    resize(oldSize + written);

    return *this;
}

template <>
TUtf32CowString&
TBasicCowString<wchar32, std::char_traits<wchar32>>::AppendUtf8(const ::TStringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.data(), s.size(), begin() + oldSize, written);
    if (pos != s.size()) {
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.data(), s.size());
    }
    resize(oldSize + written);

    return *this;
}

template <>
TUtf32CowString&
TBasicCowString<wchar32, std::char_traits<wchar32>>::AppendUtf16(const ::TWtringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 2);

    wchar32* oldEnd = begin() + oldSize;
    wchar32* end = oldEnd;
    NDetail::UTF16ToUTF32ImplScalar(s.data(), s.data() + s.size(), end);
    size_t written = end - oldEnd;

    resize(oldSize + written);

    return *this;
}

template <>
bool TBasicCowString<wchar32, std::char_traits<wchar32>>::to_lower(size_t pos, size_t n) {
    const auto f = [](const wchar32 s) { return ToLower(s); };
    return ModifyStringSymbolwise(*this, pos, n, f);
}

template <>
bool TBasicCowString<wchar32, std::char_traits<wchar32>>::to_upper(size_t pos, size_t n) {
    const auto f = [](const wchar32 s) { return ToUpper(s); };
    return ModifyStringSymbolwise(*this, pos, n, f);
}

template <>
bool TBasicCowString<wchar32, std::char_traits<wchar32>>::to_title(size_t pos, size_t n) {
    if (!*this) {
        return false;
    }

    pos = pos < this->size() ? pos : this->size();
    n = n < this->size() - pos ? n : this->size() - pos;

    const auto toLower = [](const wchar32 s) { return ToLower(s); };

    auto* p = const_cast<wchar32*>(this->data() + pos);
    const auto* pe = this->data() + pos + n;

    const auto firstSymbol = *p;
    if (firstSymbol == ToTitle(firstSymbol)) {
        p += 1;
        if (ModifySequence<true>(p, pe, toLower)) {
            DetachAndFixPointers(*this, p, pe);
            ModifySequence<false>(p, pe, toLower);
            return true;
        }
    } else {
        DetachAndFixPointers(*this, p, pe);
        WriteSymbol(ToTitle(ReadSymbol(p, pe)), p); // also moves `p` forward
        ModifySequence<false>(p, pe, toLower);
        return true;
    }

    return false;
}
