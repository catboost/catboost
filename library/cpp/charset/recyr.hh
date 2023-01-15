#pragma once

#include <cstdlib>

#include <util/charset/recode_result.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>

#include "codepage.h"
#include "doccodes.h"
#include "iconv.h"
#include "recyr_int.hh"

///////////////////////////////////////////////////////////////////////////////////////
//     input buf -> output buf                                                       //
///////////////////////////////////////////////////////////////////////////////////////
template <class TCharType>
inline RECODE_RESULT RecodeToUnicode(ECharset from, const char* in, TCharType* out, size_t inSize, size_t outSize, size_t& inRead, size_t& outWritten) {
    static_assert(sizeof(TCharType) > 1, "expect wide character type");

    return NCodepagePrivate::_recodeToUnicode(from, in, out, inSize, outSize, inRead, outWritten);
}

template <class TCharType>
inline RECODE_RESULT RecodeFromUnicode(ECharset to, const TCharType* in, char* out, size_t inSize, size_t outSize, size_t& inRead, size_t& outWritten) {
    static_assert(sizeof(TCharType) > 1, "expect wide character type");

    return NCodepagePrivate::_recodeFromUnicode(to, in, out, inSize, outSize, inRead, outWritten);
}

inline RECODE_RESULT RecodeFromUnicode(ECharset to, wchar32 rune, char* out, size_t outSize, size_t& outWritten) {
    return NCodepagePrivate::_recodeFromUnicode(to, rune, out, outSize, outWritten);
}

template <class TCharType>
inline RECODE_RESULT RecodeToUnicode(ECharset from, const char* in, TCharType* out, size_t inSize, size_t outSize) {
    size_t inRead = 0;
    size_t outWritten = 0;
    return RecodeToUnicode(from, in, out, inSize, outSize, inRead, outWritten);
}

template <class TCharType>
inline RECODE_RESULT RecodeFromUnicode(ECharset to, const TCharType* in, char* out, size_t inSize, size_t outSize) {
    size_t inRead = 0;
    size_t outWritten = 0;
    return RecodeFromUnicode(to, in, out, inSize, outSize, inRead, outWritten);
}

inline RECODE_RESULT RecodeFromUnicode(ECharset theEncoding, const wchar16* chars, size_t length,
                                       char* bytes, size_t size, size_t* read = nullptr, size_t* written = nullptr) {
    size_t w = 0, r = 0;
    RECODE_RESULT rc = ::RecodeFromUnicode(theEncoding, chars, bytes, length, size, r, w);
    if (read)
        *read = r;
    if (written)
        *written = w;
    return rc;
}

inline RECODE_RESULT Recode(ECharset from, ECharset to, const char* in, char* out, size_t inSize, size_t outSize, size_t& inRead, size_t& outWritten) {
    inRead = 0;
    outWritten = 0;

    if (!ValidCodepage(to) || !ValidCodepage(from))
        return RECODE_ERROR;

    if (to == from)
        return NCodepagePrivate::_recodeCopy(in, out, inSize, outSize, inRead, outWritten);

    if (NCodepagePrivate::NativeCodepage(from) && NCodepagePrivate::NativeCodepage(to)) {
        if (from == CODES_UTF8)
            return NCodepagePrivate::_recodeFromUTF8(to, in, out, inSize, outSize, inRead, outWritten);
        if (to == CODES_UTF8)
            return NCodepagePrivate::_recodeToUTF8(from, in, out, inSize, outSize, inRead, outWritten);
        if (from == CODES_YANDEX)
            return NCodepagePrivate::_recodeFromYandex(to, in, out, inSize, outSize, inRead, outWritten);
        if (to == CODES_YANDEX)
            return NCodepagePrivate::_recodeToYandex(from, in, out, inSize, outSize, inRead, outWritten);
    } else if (NICONVPrivate::CanConvert(from, to)) {
        return NICONVPrivate::RecodeNoThrow(from, to, in, out, inSize, outSize, inRead, outWritten);
    }

    size_t wideSize = inSize * 3;
    TArrayHolder<wchar16> wide(new wchar16[wideSize]);

    size_t wideRead = 0;
    size_t wideWritten = 0;

    RECODE_RESULT res = RecodeToUnicode(from, in, wide.Get(), inSize, wideSize, inRead, wideWritten);
    if (res != RECODE_OK)
        return res;

    res = RecodeFromUnicode(to, wide.Get(), out, wideWritten, outSize, wideRead, outWritten);

    return res;
}

inline RECODE_RESULT Recode(ECharset from, ECharset to, const char* in, char* out, size_t inSize, size_t outSize) {
    size_t inRead = 0;
    size_t outWritten = 0;
    return Recode(from, to, in, out, inSize, outSize, inRead, outWritten);
}

/**
 * Recode from one charset to another; throw an exception if conversion failed
 * @param[in] from the source character set
 * @param[in] to the target character set
 * @param[in] in    the input string buffer
 * @param[out] out  the output string object if conversion was successful
 * @return false if conversion was not attempted (charsets were the same),
 *         true if successful
 */
inline bool Recode(ECharset from, ECharset to, const TStringBuf& in, TString& out) {
    if (to == from)
        return false;

    const size_t inSize = in.length();
    const size_t outSize = SingleByteCodepage(to) ? inSize : 3 * inSize;
    out.clear(); // so we don't copy stuff around when resizing
    out.ReserveAndResize(outSize);

    size_t inRead = 0;
    size_t outWritten = 0;
    const RECODE_RESULT res = Recode(from, to, in.data(), out.begin(), inSize, outSize, inRead, outWritten);
    Y_ENSURE(RECODE_OK == res, "Recode failed. ");
    if (outWritten > outSize)
        ythrow yexception() << "Recode overrun the buffer: size="
                            << outSize << " need=" << outWritten;

    out.remove(outWritten);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////
//     TString -> TString                                                              //
///////////////////////////////////////////////////////////////////////////////////////
inline TString Recode(ECharset from, ECharset to, const TString& in) {
    TString out;
    return to != from && Recode(from, to, in, out) ? out : in;
}
inline TString RecodeToYandex(ECharset from, const TString& in) {
    return Recode(from, CODES_YANDEX, in);
}
inline TString RecodeFromYandex(ECharset to, const TString& in) {
    return Recode(CODES_YANDEX, to, in);
}

inline TString RecodeToHTMLEntities(ECharset from, const TString& in) {
    RECODE_RESULT res;
    size_t outWritten, inRead;
    TString out;
    out.resize(in.length() * (4 + 4));
    res = NCodepagePrivate::_recodeToHTMLEntities(from, in.c_str(), out.begin(), in.length(), out.length(), inRead, outWritten);
    if (res == RECODE_EOOUTPUT) { //input contains many 8-byte characters?
        out.resize(in.length() * (4 + 8));
        res = NCodepagePrivate::_recodeToHTMLEntities(from, in.c_str(), out.begin(), in.length(), out.length(), inRead, outWritten);
    }
    if (res != RECODE_OK) {
        ythrow yexception() << "Recode to HTML entities failed";
    }

    out.resize(outWritten - 1);
    return out;
}
