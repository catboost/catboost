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

        if (Y_UNLIKELY(newLen >= TDataTraits::MaxSize)) {
            throw std::length_error("Allocate() will fail");
        }

        size_t bufLen = newLen;
        const size_t dataSize = TDataTraits::CalcAllocationSizeAndCapacity(bufLen);
        Y_ASSERT(bufLen >= newLen);

        auto ret = reinterpret_cast<TData*>(oldData == nullptr ? y_allocate(dataSize) : y_reallocate(oldData, dataSize));

        ret->Refs = 1;
        ret->BufLen = bufLen;
        ret->Length = oldLen;

        TCharType* chars = TDataTraits::GetChars(ret);

        chars[oldLen] = TCharType();

        return chars;
    }

    template char* Allocate<char>(size_t oldLen, size_t newLen, TStringData* oldData);
    template wchar16* Allocate<wchar16>(size_t oldLen, size_t newLen, TStringData* oldData);
    template wchar32* Allocate<wchar32>(size_t oldLen, size_t newLen, TStringData* oldData);

    void Deallocate(void* data) {
        y_deallocate(data);
    }
}

std::ostream& operator<<(std::ostream& os, const TString& s) {
    return os.write(s.data(), s.size());
}

template<>
bool TBasicString<char, TCharTraits<char>>::to_lower(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToLower(c); }, pos, n);
}

template<>
bool TBasicString<char, TCharTraits<char>>::to_upper(size_t pos, size_t n) {
    return Transform([](size_t, char c) { return AsciiToUpper(c); }, pos, n);
}

template<>
bool TBasicString<char, TCharTraits<char>>::to_title(size_t pos, size_t n) {
    if (n == 0) {
        return false;
    }
    bool changed = to_upper(pos, 1);
    return to_lower(pos + 1, n - 1) || changed;
}

template<>
TUtf16String&
TBasicString<wchar16, TCharTraits<wchar16>>::AppendAscii(const ::TStringBuf& s) {
    ReserveAndResize(size() + s.size());

    auto dst = begin() + size() - s.size();

    for (const char* src = s.data(); dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar16>(*src);
    }

    return *this;
}

template<>
TUtf16String&
TBasicString<wchar16, TCharTraits<wchar16>>::AppendUtf8(const ::TStringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.data(), s.size(), begin() + oldSize, written);
    if (pos != s.size())
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.data(), s.size());
    remove(oldSize + written);

    return *this;
}

template<>
bool TBasicString<wchar16, TCharTraits<wchar16>>::to_lower(size_t pos, size_t n) {
    return ToLower(*this, pos, n);
}

template<>
bool TBasicString<wchar16, TCharTraits<wchar16>>::to_upper(size_t pos, size_t n) {
    return ToUpper(*this, pos, n);
}

template<>
bool TBasicString<wchar16, TCharTraits<wchar16>>::to_title(size_t pos, size_t n) {
    return ToTitle(*this, pos, n);
}

template<>
TUtf32String&
TBasicString<wchar32, TCharTraits<wchar32>>::AppendAscii(const ::TStringBuf& s) {
    ReserveAndResize(size() + s.size());

    auto dst = begin() + size() - s.size();

    for (const char* src = s.data(); dst != end(); ++dst, ++src) {
        *dst = static_cast<wchar32>(*src);
    }

    return *this;
}

template<>
TUtf32String&
TBasicString<wchar32, TCharTraits<wchar32>>::AppendUtf8(const ::TStringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 4);
    size_t written = 0;
    size_t pos = UTF8ToWideImpl(s.data(), s.size(), begin() + oldSize, written);
    if (pos != s.size())
        ythrow yexception() << "failed to decode UTF-8 string at pos " << pos << ::NDetail::InStringMsg(s.data(), s.size());
    remove(oldSize + written);

    return *this;
}

template<>
TUtf32String&
TBasicString<wchar32, TCharTraits<wchar32>>::AppendUtf16(const ::TWtringBuf& s) {
    size_t oldSize = size();
    ReserveAndResize(size() + s.size() * 2);

    wchar32* oldEnd = begin() + oldSize;
    wchar32* end = oldEnd;
    NDetail::UTF16ToUTF32ImplScalar(s.data(), s.data() + s.size(), end);
    size_t written = end - oldEnd;

    remove(oldSize + written);

    return *this;
}


template<>
bool TBasicString<wchar32, TCharTraits<wchar32>>::to_lower(size_t pos, size_t n) {
    return ToLower(*this, pos, n);
}

template<>
bool TBasicString<wchar32, TCharTraits<wchar32>>::to_upper(size_t pos, size_t n) {
    return ToUpper(*this, pos, n);
}

template<>
bool TBasicString<wchar32, TCharTraits<wchar32>>::to_title(size_t pos, size_t n) {
    return ToTitle(*this, pos, n);
}
