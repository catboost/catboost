#include "string.h"

#include <util/string/ascii.h>
#include <util/system/sys_alloc.h>
#include <util/charset/wide.h>

#include <iostream>
#include <cctype>

namespace NDetail {
    struct TStaticData {
        TStringData Data;
        size_t Buf[4];
    };

    static const TStaticData STATIC_DATA = {{0, 0, 0}, {0, 0, 0, 0}};
    void const* STRING_DATA_NULL = STATIC_DATA.Buf;

    template <typename TCharType>
    TCharType* Allocate(size_t oldLen, size_t newLen, TStringData* oldData) {
        static_assert(offsetof(TStaticData, Buf) == sizeof(TStringData), "expect offsetof(TStaticData, Buf) == sizeof(TStringData)");
        static_assert(sizeof(STATIC_DATA.Buf) >= sizeof(TCharType), "expect sizeof(STATIC_DATA.Buf) >= sizeof(TCharType)");

        using TData = TStringData;
        using TDataTraits = TStringDataTraits<TCharType>;

        if (0 == newLen) {
            return TDataTraits::GetNull();
        }

        const size_t bufLen = Max(FastClp2(newLen), newLen);

        if (bufLen >= TDataTraits::MaxSize) {
            ThrowLengthError("Allocate() will fail");
        }

        const size_t dataSize = TDataTraits::CalcAllocationSize(bufLen);

        TData* ret = (TData*)(oldData == nullptr ? y_allocate(dataSize) : y_reallocate(oldData, dataSize));

        ret->Refs = 1;
        ret->BufLen = bufLen;
        ret->Length = oldLen;

        TCharType* chars = TDataTraits::GetChars(ret);

        chars[oldLen] = TCharType();

        return chars;
    }

    template char* Allocate<char>(size_t oldLen, size_t newLen, TStringData* oldData);
    template wchar16* Allocate<wchar16>(size_t oldLen, size_t newLen, TStringData* oldData);

    void Deallocate(void* data) {
        y_deallocate(data);
    }
}

TOStream& operator<<(TOStream& os, const TString& s) {
    return os.write(~s, +s);
}

bool TString::to_lower(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToLower(c); }, pos, n);
}

bool TString::to_upper(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToUpper(c); }, pos, n);
}

bool TString::to_title(size_t pos, size_t n) {
    if (n == 0) {
        return false;
    }
    bool changed = to_upper(pos, 1);
    return to_lower(pos + 1, n - 1) || changed;
}

TUtf16String& TUtf16String::AppendUtf8(const ::TFixedString<char>& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.Length * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.Start, s.Length, begin() + oldSize, written);
    if (pos != s.Length)
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.Start, s.Length);
    remove(oldSize + written);
    return *this;
}

TUtf16String& TUtf16String::AppendAscii(const ::TFixedString<char>& s) {
    ReserveAndResize(size() + s.Length);

    auto dst = begin() + size() - s.Length;

    for (const char *src = s.Start; dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar16>(*src);
    }

    return *this;
}

#define POS_RANGE(pos, len) pos = ((pos) < (len) ? (pos) : (len))
#define NUM_RANGE(num, len, pos) num = ((num) < ((len) - (pos)) ? (num) : ((len) - (pos)))

bool TUtf16String::to_lower(size_t pos, size_t n) {
    bool changed = false;
    size_t len = length();
    POS_RANGE(pos, len);
    NUM_RANGE(n, len, pos);

    wchar16* begin = Data_ + pos;
    wchar16* end = begin + n;
    for (wchar16* i = begin; i != end;) {
        wchar32 s = ReadSymbol(i, end);
        wchar32 c = ToLower(s);
        if (c != s) {
            if (Y_UNLIKELY(!changed)) {
                Detach();
                end = Data_ + pos + n;
                i = Data_ + pos + (i - begin);
                changed = true;
            }
            WriteSymbol(c, i); // also skipping symbol
        } else {
            i = SkipSymbol(i, end);
        }
    }
    return changed;
}

bool TUtf16String::to_upper(size_t pos, size_t n) {
    bool changed = false;
    size_t len = length();
    POS_RANGE(pos, len);
    NUM_RANGE(n, len, pos);

    wchar16* begin = Data_ + pos;
    wchar16* end = begin + n;
    for (wchar16* i = begin; i != end;) {
        wchar32 s = ReadSymbol(i, end);
        wchar32 c = ToUpper(s);
        if (c != s) {
            if (Y_UNLIKELY(!changed)) {
                Detach();
                end = Data_ + pos + n;
                i = Data_ + pos + (i - begin);
                changed = true;
            }
            WriteSymbol(c, i); // also skipping symbol
        } else {
            i = SkipSymbol(i, end);
        }
    }
    return changed;
}

bool TUtf16String::to_title() {
    size_t len = length();
    if (len == 0)
        return false;

    bool changed = false;

    wchar16* begin = Data_;
    wchar16* end = Data_ + len;
    wchar16* i = Data_;

    wchar32 s = ReadSymbol(i, end);
    wchar32 c = ::ToTitle(s);
    if (c != s) {
        Y_ASSERT(!changed);
        Detach();
        end = Data_ + len;
        i = Data_ + (i - begin);
        changed = true;
        WriteSymbol(c, i); // also skipping symbol
    } else {
        i = SkipSymbol(i, end);
    }
    Y_ASSERT(i <= end);

    while (i != end) {
        s = ReadSymbol(i, end);
        c = ToLower(s);
        if (c != s) {
            if (Y_UNLIKELY(!changed)) {
                Detach();
                end = Data_ + len;
                i = Data_ + (i - begin);
                changed = true;
            }
            WriteSymbol(c, i); // also skipping symbol
        } else {
            i = SkipSymbol(i, end);
        }
    }
    return changed;
}
