#pragma once

#include <util/charset/recode_result.h>
#include <util/charset/utf8.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/system/defaults.h>

#include "codepage.h"
#include "doccodes.h"
#include "iconv.h"
#include "wide.h"

namespace NCodepagePrivate {
    inline RECODE_RESULT _recodeCopy(const char* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        in_readed = in_size;
        RECODE_RESULT res = RECODE_OK;
        if (in_readed > out_size) {
            res = RECODE_EOOUTPUT;
            in_readed = out_size;
        }
        if (in != out)
            memcpy(out, in, in_readed);
        out_writed = in_readed;
        return res;
    }

    inline RECODE_RESULT _recodeToUTF8(ECharset From, const char* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        if (From == CODES_UTF8)
            return _recodeCopy(in, out, in_size, out_size, in_readed, out_writed);
        const CodePage* cp = CodePageByCharset(From);

        const unsigned char* in_start = (const unsigned char*)in;
        const unsigned char* in_end = in_start + in_size;
        const unsigned char* out_start = (unsigned char*)out;
        const unsigned char* out_end = out_start + out_size;

        size_t rune_len;
        RECODE_RESULT res = RECODE_OK;
        while ((unsigned char*)in < in_end && res == RECODE_OK) {
            res = SafeWriteUTF8Char(cp->unicode[(unsigned char)(*in++)], rune_len, (unsigned char*)out, out_end);
            out += rune_len;
        }
        in_readed = (unsigned char*)in - in_start;
        out_writed = (unsigned char*)out - out_start;
        return res;
    }

    inline RECODE_RESULT _recodeFromUTF8(ECharset to, const char* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        if (to == CODES_UTF8)
            return _recodeCopy(in, out, in_size, out_size, in_readed, out_writed);
        Y_ASSERT(CODES_UNKNOWN < to && to < CODES_MAX);
        const Encoder* enc = &EncoderByCharset(to);

        const unsigned char* in_start = (const unsigned char*)in;
        const unsigned char* in_end = in_start + in_size;
        const unsigned char* out_start = (unsigned char*)out;
        const unsigned char* out_end = out_start + out_size;

        wchar32 rune;
        size_t rune_len;
        RECODE_RESULT res = RECODE_OK;
        while ((const unsigned char*)in < in_end && (res == RECODE_OK || res == RECODE_BROKENSYMBOL)) {
            res = SafeReadUTF8Char(rune, rune_len, (const unsigned char*)in, in_end);
            if (res == RECODE_BROKENSYMBOL)
                rune_len = 1;
            if (res != RECODE_EOINPUT)
                *out++ = enc->Tr(rune);
            in += rune_len;
            if (res == RECODE_OK && (const unsigned char*)in < in_end && (unsigned char*)out >= out_end)
                res = RECODE_EOOUTPUT;
        }
        in_readed = (unsigned char*)in - in_start;
        out_writed = (unsigned char*)out - out_start;
        return res;
    }

    inline RECODE_RESULT _recodeToYandex(ECharset From, const char* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        if (From == CODES_YANDEX)
            return _recodeCopy(in, out, in_size, out_size, in_readed, out_writed);
        if (From == CODES_UTF8)
            return _recodeFromUTF8(CODES_YANDEX, in, out, in_size, out_size, in_readed, out_writed);
        in_readed = (out_size > in_size) ? in_size : out_size;
        const Recoder& rcdr = NCodepagePrivate::TCodePageData::rcdr_to_yandex[From];
        rcdr.Tr(in, out, in_readed);
        out_writed = in_readed;
        if (out_size < in_size)
            return RECODE_EOOUTPUT;
        return RECODE_OK;
    }
    inline RECODE_RESULT _recodeFromYandex(ECharset To, const char* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        if (To == CODES_YANDEX)
            return _recodeCopy(in, out, in_size, out_size, in_readed, out_writed);
        if (To == CODES_UTF8)
            return _recodeToUTF8(CODES_YANDEX, in, out, in_size, out_size, in_readed, out_writed);
        in_readed = (out_size > in_size) ? in_size : out_size;
        const Recoder& rcdr = NCodepagePrivate::TCodePageData::rcdr_from_yandex[To];
        rcdr.Tr(in, out, in_readed);
        out_writed = in_readed;
        if (out_size < in_size)
            return RECODE_EOOUTPUT;
        return RECODE_OK;
    }

    template <class TCharType>
    inline RECODE_RESULT _recodeUTF8ToUnicode(const char* in, TCharType* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        const unsigned char* inp = (const unsigned char*)in;
        const unsigned char* in_end = inp + in_size;
        TCharType* outp = out;
        const TCharType* out_end = outp + out_size;
        size_t rune_len;
        wchar32 rune;
        RECODE_RESULT res = RECODE_OK;
        while ((res == RECODE_OK || res == RECODE_BROKENSYMBOL) && inp < in_end && outp < out_end) {
            res = SafeReadUTF8Char(rune, rune_len, inp, in_end);
            if (res == RECODE_BROKENSYMBOL)
                rune_len = 1;
            if (res == RECODE_OK || res == RECODE_BROKENSYMBOL) {
                if (!WriteSymbol(rune, outp, out_end)) {
                    break;
                }
                inp += rune_len;
            }
        }
        in_readed = inp - (const unsigned char*)in;
        out_writed = outp - out;

        if ((res == RECODE_OK || res == RECODE_BROKENSYMBOL) && in_readed != in_size)
            return RECODE_EOOUTPUT;

        return res;
    }

    template <class TCharType>
    inline RECODE_RESULT _recodeSBToUnicode(ECharset From, const char* in, TCharType* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        const CodePage* cp = CodePageByCharset(From);
        const unsigned char* inp = (const unsigned char*)in;
        const unsigned char* in_end = inp + in_size;
        TCharType* outp = out;
        const TCharType* out_end = outp + out_size;
        while (inp < in_end && outp < out_end)
            *outp++ = static_cast<TCharType>(cp->unicode[*inp++]);
        in_readed = inp - (const unsigned char*)in;
        out_writed = outp - out;
        if (in_readed != in_size)
            return RECODE_EOOUTPUT;
        return RECODE_OK;
    }

    template <class TCharType>
    inline RECODE_RESULT _recodeUnicodeToUTF8Impl(const TCharType* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        const TCharType* inp = in;
        const TCharType* in_end = in + in_size;
        unsigned char* outp = (unsigned char*)out;
        const unsigned char* out_end = outp + out_size;
        size_t rune_len;
        wchar32 rune;
        RECODE_RESULT res = RECODE_OK;

        while ((res == RECODE_OK || res == RECODE_BROKENSYMBOL) && inp != in_end) {
            rune = ReadSymbolAndAdvance(inp, in_end);
            res = SafeWriteUTF8Char(rune, rune_len, outp, out_end);
            if (outp >= out_end && (res == RECODE_OK || res == RECODE_BROKENSYMBOL))
                res = RECODE_EOOUTPUT;
            outp += rune_len;
        }
        in_readed = inp - in;
        out_writed = outp - (const unsigned char*)out;
        return res;
    }

    inline RECODE_RESULT _recodeUnicodeToUTF8(wchar32 rune, char* out, size_t out_size, size_t& nwritten) {
        return SafeWriteUTF8Char(rune, nwritten, (unsigned char*)out, out_size);
    }

    template <class TCharType, int Size = sizeof(TCharType)>
    struct TCharTypeSwitch;

    template <class TCharType>
    struct TCharTypeSwitch<TCharType, 2> {
        using TRealCharType = wchar16;
    };

    template <class TCharType>
    struct TCharTypeSwitch<TCharType, 4> {
        using TRealCharType = wchar32;
    };

    template <class TCharType>
    inline RECODE_RESULT _recodeUnicodeToUTF8(const TCharType* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        static_assert(sizeof(TCharType) > 1, "expect some wide type");

        using TRealCharType = typename TCharTypeSwitch<TCharType>::TRealCharType;

        return _recodeUnicodeToUTF8Impl(reinterpret_cast<const TRealCharType*>(in), out, in_size, out_size, in_readed, out_writed);
    }

    template <class TCharType>
    inline RECODE_RESULT _recodeUnicodeToSB(ECharset To, const TCharType* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        const TCharType* inp = in;
        const TCharType* in_end = in + in_size;
        const char* out_begin = out;
        const char* out_end = out + out_size;

        const Encoder* enc = &EncoderByCharset(To);
        while (inp != in_end && out != out_end) {
            *out++ = enc->Tr(ReadSymbolAndAdvance(inp, in_end));
        }

        in_readed = inp - in;
        out_writed = out - out_begin;

        if (in_readed != in_size)
            return RECODE_EOOUTPUT;

        return RECODE_OK;
    }

    inline RECODE_RESULT _recodeUnicodeToSB(ECharset To, wchar32 rune, char* out, size_t out_size, size_t& nwritten) {
        if (0 == out_size)
            return RECODE_EOOUTPUT;
        *out = EncoderByCharset(To).Tr(rune);
        nwritten = 1;
        return RECODE_OK;
    }

    inline RECODE_RESULT _rune2hex(wchar32 in, char* out, size_t out_size, size_t& out_writed) {
        static const char hex_digs[] = "0123456789ABCDEF";
        out_writed = 0;
        RECODE_RESULT res = RECODE_OK;
        for (int i = 7; i >= 0; i--) {
            unsigned char h = (unsigned char)(in >> (i * 4) & 0x0F);
            if (h || i == 0) {
                if (out_writed + 1 >= out_size) {
                    res = RECODE_EOOUTPUT;
                    break;
                }
                out[out_writed++] = hex_digs[h];
            }
        }
        return res;
    }

    inline RECODE_RESULT _recodeUnicodeToHTMLEntities(const wchar32* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        const wchar32* in_end = in + in_size;
        const char* out_beg = out;
        const wchar32* in_beg = in;
        RECODE_RESULT res = RECODE_OK;

        const char* out_end = out + out_size - 1;
        while (in < in_end && out < out_end) {
            if (*in < 0x80 && *in != '<' && *in != '&' && *in != '>') { //ascii
                *out++ = char(*in & 0x00FF);
            } else { //entity
                char* ent = out;
                size_t ent_writed;
                if (ent > out_end - 6) {
                    res = RECODE_EOOUTPUT;
                    break;
                }
                memcpy(ent, "&#x", 3);
                ent += 3;
                res = _rune2hex(*in, ent, out_end - 1 - ent, ent_writed);
                if (res != RECODE_OK)
                    break;
                ent += ent_writed;
                *ent++ = ';';
                out = ent;
            }
            in++;
        }
        *out++ = '\x00';
        out_writed = out - out_beg;
        in_readed = in - in_beg;
        return res;
    }

    template <class TCharType>
    inline RECODE_RESULT _recodeToUnicode(ECharset From, const char* in, TCharType* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        if (!ValidCodepage(From))
            return RECODE_ERROR;

        if (!NCodepagePrivate::NativeCodepage(From))
            return NICONVPrivate::RecodeToUnicodeNoThrow(From, in, out, in_size, out_size, in_readed, out_writed);

        if (From == CODES_UTF8)
            return _recodeUTF8ToUnicode(in, out, in_size, out_size, in_readed, out_writed);

        return _recodeSBToUnicode(From, in, out, in_size, out_size, in_readed, out_writed);
    }

    template <class TCharType>
    inline RECODE_RESULT _recodeFromUnicode(ECharset To, const TCharType* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        if (!ValidCodepage(To))
            return RECODE_ERROR;

        if (!NCodepagePrivate::NativeCodepage(To))
            return NICONVPrivate::RecodeFromUnicodeNoThrow(To, in, out, in_size, out_size, in_readed, out_writed);

        if (To == CODES_UTF8)
            return NCodepagePrivate::_recodeUnicodeToUTF8(in, out, in_size, out_size, in_readed, out_writed);

        return NCodepagePrivate::_recodeUnicodeToSB(To, in, out, in_size, out_size, in_readed, out_writed);
    }

    inline RECODE_RESULT _recodeFromUnicode(ECharset To, wchar32 rune, char* out, size_t out_size, size_t& nwritten) {
        if (!ValidCodepage(To))
            return RECODE_ERROR;

        if (!NCodepagePrivate::NativeCodepage(To)) {
            size_t nread = 0;
            return NICONVPrivate::RecodeFromUnicodeNoThrow(To, &rune, out, 1, out_size, nread, nwritten);
        }

        if (To == CODES_UTF8)
            return NCodepagePrivate::_recodeUnicodeToUTF8(rune, out, out_size, nwritten);

        return NCodepagePrivate::_recodeUnicodeToSB(To, rune, out, out_size, nwritten);
    }

    inline RECODE_RESULT _recodeToHTMLEntities(ECharset From, const char* in, char* out, size_t in_size, size_t out_size, size_t& in_readed, size_t& out_writed) {
        TArrayHolder<wchar32> bufHolder(new wchar32[in_size]);
        wchar32* buf = bufHolder.Get();
        size_t unicode_size;
        RECODE_RESULT res1, res2;

        //first pass - to unicode
        res1 = _recodeToUnicode(From, in, buf, in_size, in_size, in_readed, unicode_size);

        //second pass - to entities
        res2 = _recodeUnicodeToHTMLEntities(buf, out, in_size, out_size, in_readed, out_writed);

        return (res2 != RECODE_OK) ? res2 : res1;
    }

}
