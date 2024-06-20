#include "base64.h"

#include <contrib/libs/base64/avx2/libbase64.h>
#include <contrib/libs/base64/neon32/libbase64.h>
#include <contrib/libs/base64/neon64/libbase64.h>
#include <contrib/libs/base64/plain32/libbase64.h>
#include <contrib/libs/base64/plain64/libbase64.h>
#include <contrib/libs/base64/ssse3/libbase64.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>
#include <util/system/cpu_id.h>
#include <util/system/platform.h>

#include <array>

using namespace std::string_view_literals;

#define BASE64_UT_DECLARE_BASE64_IMPL(prefix, encFunction, decFunction)                                                        \
    Y_DECLARE_UNUSED                                                                                                           \
    static size_t prefix##Base64Decode(void* dst, const char* b, const char* e) {                                              \
        const auto size = e - b;                                                                                               \
        Y_ENSURE(!(size % 4), "incorrect input length for base64 decode");                                                     \
                                                                                                                               \
        size_t outLen;                                                                                                         \
        decFunction(b, size, (char*)dst, &outLen);                                                                             \
        return outLen;                                                                                                         \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static inline TStringBuf prefix##Base64Decode(const TStringBuf& src, void* dst) {                                          \
        return TStringBuf((const char*)dst, ::NB64Etalon::prefix##Base64Decode(dst, src.begin(), src.end()));                  \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static inline void prefix##Base64Decode(const TStringBuf& src, TString& dst) {                                             \
        dst.ReserveAndResize(Base64DecodeBufSize(src.size()));                                                                 \
        dst.resize(::NB64Etalon::prefix##Base64Decode(src, dst.begin()).size());                                               \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static inline TString prefix##Base64Decode(const TStringBuf& s) {                                                          \
        TString ret;                                                                                                           \
        prefix##Base64Decode(s, ret);                                                                                          \
        return ret;                                                                                                            \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static char* prefix##Base64Encode(char* outstr, const unsigned char* instr, size_t len) {                                  \
        size_t outLen;                                                                                                         \
        encFunction((char*)instr, len, outstr, &outLen);                                                                       \
        *(outstr + outLen) = '\0';                                                                                             \
        return outstr + outLen;                                                                                                \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static inline TStringBuf prefix##Base64Encode(const TStringBuf& src, void* tmp) {                                          \
        return TStringBuf((const char*)tmp, ::NB64Etalon::prefix##Base64Encode((char*)tmp, (const unsigned char*)src.data(), src.size())); \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static inline void prefix##Base64Encode(const TStringBuf& src, TString& dst) {                                             \
        dst.ReserveAndResize(Base64EncodeBufSize(src.size()));                                                                 \
        dst.resize(::NB64Etalon::prefix##Base64Encode(src, dst.begin()).size());                                               \
    }                                                                                                                          \
                                                                                                                               \
    Y_DECLARE_UNUSED                                                                                                           \
    static inline TString prefix##Base64Encode(const TStringBuf& s) {                                                          \
        TString ret;                                                                                                           \
        prefix##Base64Encode(s, ret);                                                                                          \
        return ret;                                                                                                            \
    }

namespace NB64Etalon {
    BASE64_UT_DECLARE_BASE64_IMPL(PLAIN32, plain32_base64_encode, plain32_base64_decode)
    BASE64_UT_DECLARE_BASE64_IMPL(PLAIN64, plain64_base64_encode, plain64_base64_decode)
    BASE64_UT_DECLARE_BASE64_IMPL(NEON32, neon32_base64_encode, neon32_base64_decode)
    BASE64_UT_DECLARE_BASE64_IMPL(NEON64, neon64_base64_encode, neon64_base64_decode)
    BASE64_UT_DECLARE_BASE64_IMPL(AVX2, avx2_base64_encode, avx2_base64_decode)
    BASE64_UT_DECLARE_BASE64_IMPL(SSSE3, ssse3_base64_encode, ssse3_base64_decode)

#undef BASE64_UT_DECLARE_BASE64_IMPL

    struct TImpls {
        enum EImpl : size_t {
            PLAIN32_IMPL,
            PLAIN64_IMPL,
            NEON32_IMPL,
            NEON64_IMPL,
            AVX2_IMPL,
            SSSE3_IMPL,
            MAX_IMPL
        };

        using TEncodeF = void (*)(const TStringBuf&, TString&);
        using TDecodeF = void (*)(const TStringBuf&, TString&);

        struct TImpl {
            TEncodeF Encode = nullptr;
            TDecodeF Decode = nullptr;
        };

        std::array<TImpl, MAX_IMPL> Impl;

        TImpls() {
            Impl[PLAIN32_IMPL].Encode = PLAIN32Base64Encode;
            Impl[PLAIN32_IMPL].Decode = PLAIN32Base64Decode;
            Impl[PLAIN64_IMPL].Encode = PLAIN64Base64Encode;
            Impl[PLAIN64_IMPL].Decode = PLAIN64Base64Decode;
#if defined(_arm32_)
            Impl[NEON32_IMPL].Encode = NEON32Base64Encode;
            Impl[NEON32_IMPL].Decode = NEON32Base64Decode;
#elif defined(_arm64_)
            Impl[NEON64_IMPL].Encode = NEON64Base64Encode;
            Impl[NEON64_IMPL].Decode = NEON64Base64Decode;
#elif defined(_x86_64_)
            if (NX86::HaveSSSE3()) {
                Impl[SSSE3_IMPL].Encode = SSSE3Base64Encode;
                Impl[SSSE3_IMPL].Decode = SSSE3Base64Decode;
            }

            if (NX86::HaveAVX2()) {
                Impl[AVX2_IMPL].Encode = AVX2Base64Encode;
                Impl[AVX2_IMPL].Decode = AVX2Base64Decode;
            }
#else
            ythrow yexception() << "Failed to identify the platform";
#endif
        }
    };

    TImpls GetImpls() {
        static const TImpls IMPLS;
        return IMPLS;
    }
}

template <>
void Out<NB64Etalon::TImpls::EImpl>(IOutputStream& o, typename TTypeTraits<NB64Etalon::TImpls::EImpl>::TFuncParam v) {
    switch (v) {
        case NB64Etalon::TImpls::PLAIN32_IMPL:
            o << TStringBuf{"PLAIN32"};
            return;
        case NB64Etalon::TImpls::PLAIN64_IMPL:
            o << TStringBuf{"PLAIN64"};
            return;
        case NB64Etalon::TImpls::NEON64_IMPL:
            o << TStringBuf{"NEON64"};
            return;
        case NB64Etalon::TImpls::NEON32_IMPL:
            o << TStringBuf{"NEON32"};
            return;
        case NB64Etalon::TImpls::SSSE3_IMPL:
            o << TStringBuf{"SSSE3"};
            return;
        case NB64Etalon::TImpls::AVX2_IMPL:
            o << TStringBuf{"AVX2"};
            return;
        default:
            ythrow yexception() << "invalid";
    }
}

static void TestEncodeDecodeIntoString(const TString& plain, const TString& encoded, const TString& encodedUrl, const TString& encodedNoPadding, const TString& encodedUrlNoPadding) {
    TString a, b;

    Base64Encode(plain, a);
    UNIT_ASSERT_VALUES_EQUAL(a, encoded);

    Base64Decode(a, b);
    UNIT_ASSERT_VALUES_EQUAL(b, plain);

    Base64EncodeUrl(plain, a);
    UNIT_ASSERT_VALUES_EQUAL(a, encodedUrl);

    Base64Decode(a, b);
    UNIT_ASSERT_VALUES_EQUAL(b, plain);

    Base64EncodeNoPadding(plain, a);
    UNIT_ASSERT_VALUES_EQUAL(a, encodedNoPadding);

    TString c = Base64DecodeUneven(a);
    UNIT_ASSERT_VALUES_EQUAL(c, plain);

    Base64EncodeUrlNoPadding(plain, a);
    UNIT_ASSERT_VALUES_EQUAL(a, encodedUrlNoPadding);

    TString d = Base64DecodeUneven(a);
    UNIT_ASSERT_VALUES_EQUAL(d, plain);
}

static void TestEncodeStrictDecodeIntoString(const TString& plain, const TString& encoded, const TString& encodedUrl) {
    TString a, b;

    Base64Encode(plain, a);
    UNIT_ASSERT_VALUES_EQUAL(a, encoded);

    Base64StrictDecode(a, b);
    UNIT_ASSERT_VALUES_EQUAL(b, plain);

    Base64EncodeUrl(plain, a);
    UNIT_ASSERT_VALUES_EQUAL(a, encodedUrl);

    Base64StrictDecode(a, b);
    UNIT_ASSERT_VALUES_EQUAL(b, plain);
}

Y_UNIT_TEST_SUITE(TBase64) {
    Y_UNIT_TEST(TestEncode) {
        UNIT_ASSERT_VALUES_EQUAL(Base64Encode("12z"), "MTJ6");
        UNIT_ASSERT_VALUES_EQUAL(Base64Encode("123"), "MTIz");
        UNIT_ASSERT_VALUES_EQUAL(Base64Encode("12"), "MTI=");
        UNIT_ASSERT_VALUES_EQUAL(Base64Encode("1"), "MQ==");
    }

    Y_UNIT_TEST(TestIntoString) {
        {
            TString str;
            for (size_t i = 0; i < 256; ++i)
                str += char(i);

            const TString base64 =
                "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJy"
                "gpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9Q"
                "UVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eH"
                "l6e3x9fn+AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6Ch"
                "oqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIyc"
                "rLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvs7e7v8PHy"
                "8/T19vf4+fr7/P3+/w==";
            const TString base64Url =
                "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJy"
                "gpKissLS4vMDEyMzQ1Njc4OTo7PD0-P0BBQkNERUZHSElKS0xNTk9Q"
                "UVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eH"
                "l6e3x9fn-AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6Ch"
                "oqOkpaanqKmqq6ytrq-wsbKztLW2t7i5uru8vb6_wMHCw8TFxsfIyc"
                "rLzM3Oz9DR0tPU1dbX2Nna29zd3t_g4eLj5OXm5-jp6uvs7e7v8PHy"
                "8_T19vf4-fr7_P3-_w,,";
            const TString base64WithoutPadding =
                "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJy"
                "gpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9Q"
                "UVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eH"
                "l6e3x9fn+AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6Ch"
                "oqOkpaanqKmqq6ytrq+wsbKztLW2t7i5uru8vb6/wMHCw8TFxsfIyc"
                "rLzM3Oz9DR0tPU1dbX2Nna29zd3t/g4eLj5OXm5+jp6uvs7e7v8PHy"
                "8/T19vf4+fr7/P3+/w";
            const TString base64UrlWithoutPadding =
                "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJy"
                "gpKissLS4vMDEyMzQ1Njc4OTo7PD0-P0BBQkNERUZHSElKS0xNTk9Q"
                "UVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eH"
                "l6e3x9fn-AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6Ch"
                "oqOkpaanqKmqq6ytrq-wsbKztLW2t7i5uru8vb6_wMHCw8TFxsfIyc"
                "rLzM3Oz9DR0tPU1dbX2Nna29zd3t_g4eLj5OXm5-jp6uvs7e7v8PHy"
                "8_T19vf4-fr7_P3-_w";

            TestEncodeDecodeIntoString(str, base64, base64Url, base64WithoutPadding, base64UrlWithoutPadding);
            TestEncodeStrictDecodeIntoString(str, base64, base64Url);
        }

        {
            const TString str = "http://yandex.ru:1234/request?param=value&lll=fff#fragment";

            const TString base64 = "aHR0cDovL3lhbmRleC5ydToxMjM0L3JlcXVlc3Q/cGFyYW09dmFsdWUmbGxsPWZmZiNmcmFnbWVudA==";
            const TString base64Url = "aHR0cDovL3lhbmRleC5ydToxMjM0L3JlcXVlc3Q_cGFyYW09dmFsdWUmbGxsPWZmZiNmcmFnbWVudA,,";
            const TString base64WithoutPadding = "aHR0cDovL3lhbmRleC5ydToxMjM0L3JlcXVlc3Q/cGFyYW09dmFsdWUmbGxsPWZmZiNmcmFnbWVudA";
            const TString base64UrlWithoutPadding = "aHR0cDovL3lhbmRleC5ydToxMjM0L3JlcXVlc3Q_cGFyYW09dmFsdWUmbGxsPWZmZiNmcmFnbWVudA";

            TestEncodeDecodeIntoString(str, base64, base64Url, base64WithoutPadding, base64UrlWithoutPadding);
            TestEncodeStrictDecodeIntoString(str, base64, base64Url);
        }
    }

    Y_UNIT_TEST(TestDecode) {
        UNIT_ASSERT_EXCEPTION(Base64Decode("a"), yexception);
        UNIT_ASSERT_EXCEPTION(Base64StrictDecode("a"), yexception);

        UNIT_ASSERT_VALUES_EQUAL(Base64Decode(""), "");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecode(""), "");

        UNIT_ASSERT_VALUES_EQUAL(Base64Decode("MTI="), "12");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecode("MTI="), "12");

        UNIT_ASSERT_VALUES_EQUAL(Base64Decode("QQ=="), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecode("QQ=="), "A");

        UNIT_ASSERT_EXCEPTION(Base64StrictDecode("M=I="), yexception);

        UNIT_ASSERT_VALUES_EQUAL(Base64Decode("dnluZHg="), "vyndx");
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecode("dnluZHg="), "vyndx");

        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecode("dnluZHg=dmlkZW8="), "vyndxvideo");

        UNIT_ASSERT_EXCEPTION(Base64StrictDecode("aHR0cDovL2ltZy5tZWdhLXBvcm5vLnJ1Lw=a"), yexception);

        UNIT_ASSERT_EXCEPTION(Base64StrictDecode("aHh=="), yexception);
        UNIT_ASSERT_EXCEPTION(Base64StrictDecode("\1\1\1\2"), yexception);
    }

    Y_UNIT_TEST(TestDecodeUneven) {
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven(""), "");

        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("YWFh"), "aaa");

        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("MTI="), "12");
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("MTI,"), "12");
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("MTI"), "12");

        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("QQ=="), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("QQ,,"), "A");
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("QQ"), "A");

        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("dnluZHg="), "vyndx");
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("dnluZHg,"), "vyndx");
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven("dnluZHg"), "vyndx");
    }

    Y_UNIT_TEST(TestDecodeRandom) {
        TString input;
        constexpr size_t testSize = 240000;
        for (size_t i = 0; i < testSize; ++i) {
            input.push_back(rand() % 256);
        }
        TString output;
        TString encoded = Base64Encode(input);
        TString encodedUrl = TString::Uninitialized(Base64EncodeBufSize(input.length()));
        Base64EncodeUrlNoPadding(input, encodedUrl);
        UNIT_ASSERT_VALUES_EQUAL(Base64Decode(encoded), input);
        UNIT_ASSERT_VALUES_EQUAL(Base64StrictDecode(encoded), input);
        UNIT_ASSERT_VALUES_EQUAL(Base64DecodeUneven(encodedUrl), input);
    }

    Y_UNIT_TEST(TestAllPossibleOctets) {
        const TString x("\0\x01\x02\x03\x04\x05\x06\x07\b\t\n\x0B\f\r\x0E\x0F\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1A\x1B\x1C\x1D\x1E\x1F !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7F"sv);
        const TString xEnc = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn8=";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestTwoPaddingCharacters) {
        const TString x("a");
        const TString xEnc = "YQ==";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestOnePaddingCharacter) {
        const TString x("aa");
        const TString xEnc = "YWE=";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestNoPaddingCharacters) {
        const TString x("aaa");
        const TString xEnc = "YWFh";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestTrailingZero) {
        const TString x("foo\0"sv);
        const TString xEnc = "Zm9vAA==";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestTwoTrailingZeroes) {
        const TString x("foo\0\0"sv);
        const TString xEnc = "Zm9vAAA=";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestZero) {
        const TString x("\0"sv);
        const TString xEnc = "AA==";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestSymbolsAfterZero) {
        const TString x("\0a"sv);
        const TString xEnc = "AGE=";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestEmptyString) {
        const TString x = "";
        const TString xEnc = "";
        const TString y = Base64Decode(xEnc);
        const TString yEnc = Base64Encode(x);
        UNIT_ASSERT_VALUES_EQUAL(x, y);
        UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
    }

    Y_UNIT_TEST(TestBackendsConsistencyOnRandomData) {
        constexpr size_t TEST_CASES_COUNT = 1000;
        constexpr size_t MAX_DATA_SIZE = 1000;
        TFastRng<ui32> prng{42};
        TVector<TString> xs{TEST_CASES_COUNT};
        TString xEnc;
        TString xDec;
        TString yEnc;
        TString yDec;

        for (auto& x : xs) {
            const size_t size = prng() % MAX_DATA_SIZE;
            for (size_t j = 0; j < size; ++j) {
                x += static_cast<char>(prng() % 256);
            }
        }

        static const auto IMPLS = NB64Etalon::GetImpls();
        for (size_t i = 0; i < static_cast<size_t>(NB64Etalon::TImpls::MAX_IMPL); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(NB64Etalon::TImpls::MAX_IMPL); ++j) {
                const auto ei = static_cast<NB64Etalon::TImpls::EImpl>(i);
                const auto ej = static_cast<NB64Etalon::TImpls::EImpl>(j);
                const auto impl = IMPLS.Impl[i];
                const auto otherImpl = IMPLS.Impl[j];
                if (!impl.Encode && !impl.Decode || !otherImpl.Encode && !otherImpl.Decode) {
                    continue;
                }

                for (const auto& x : xs) {
                    impl.Encode(x, xEnc);
                    impl.Decode(xEnc, xDec);
                    Y_ENSURE(x == xDec, "something is wrong with " << ei << " implementation");

                    otherImpl.Encode(x, yEnc);
                    otherImpl.Decode(xEnc, yDec);
                    Y_ENSURE(x == yDec, "something is wrong with " << ej << " implementation");

                    UNIT_ASSERT_VALUES_EQUAL(xEnc, yEnc);
                    UNIT_ASSERT_VALUES_EQUAL(xDec, yDec);
                }
            }
        }
    }

    Y_UNIT_TEST(TestIfEncodedDataIsZeroTerminatedOnRandomData) {
        constexpr size_t TEST_CASES_COUNT = 1000;
        constexpr size_t MAX_DATA_SIZE = 1000;
        TFastRng<ui32> prng{42};
        TString x;
        TVector<char> buf;
        for (size_t i = 0; i < TEST_CASES_COUNT; ++i) {
            const size_t size = prng() % MAX_DATA_SIZE;
            x.clear();
            for (size_t j = 0; j < size; ++j) {
                x += static_cast<char>(prng() % 256);
            }

            buf.assign(Base64EncodeBufSize(x.size()), Max<char>());
            const auto* const xEncEnd = Base64Encode(buf.data(), (const unsigned char*)x.data(), x.size());
            UNIT_ASSERT_VALUES_EQUAL(*xEncEnd, '\0');
        }
    }

    Y_UNIT_TEST(TestDecodeURLEncodedNoPadding) {
        const auto x = "123";
        const auto xDec = Base64Decode("MTIz");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeURLEncodedOnePadding) {
        const auto x = "12";
        const auto xDec = Base64Decode("MTI,");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeURLEncodedTwoPadding) {
        const auto x = "1";
        const auto xDec = Base64Decode("MQ,,");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeURLEncodedWithoutPadding) {
        const auto x = "1";
        const auto xDec = Base64DecodeUneven("MQ");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeNoPaddingLongString) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?a";
        const auto xDec = Base64Decode("SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz9h");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeOnePaddingLongString) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?";
        const auto xDec = Base64Decode("SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz8=");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeTwoPaddingLongString) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?aa";
        const auto xDec = Base64Decode("SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz9hYQ==");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeURLEncodedNoPaddingLongString) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?a";
        const auto xDec = Base64Decode("SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz9h");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeURLEncodedOnePaddingLongString) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?";
        const auto xDec = Base64Decode("SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz8,");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeURLEncodedTwoPaddingLongString) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?aa";
        const auto xDec = Base64Decode("SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz9hYQ,,");
        UNIT_ASSERT_VALUES_EQUAL(x, xDec);
    }

    Y_UNIT_TEST(TestDecodeUnevenDst) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?aa";
        TString b64 = "SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz9hYQ";
        TVector<char> buf(Base64DecodeBufSize(b64.Size()), '\0');
        Base64DecodeUneven(buf.begin(), b64);
        TString res(buf.data());
        UNIT_ASSERT_VALUES_EQUAL(x, res);
    }

    Y_UNIT_TEST(TestDecodeUnevenDst2) {
        const auto x = "How do I convert between big-endian and little-endian values in C++?";
        TString b64 = "SG93IGRvIEkgY29udmVydCBiZXR3ZWVuIGJpZy1lbmRpYW4gYW5kIGxpdHRsZS1lbmRpYW4gdmFsdWVzIGluIEMrKz8";
        TVector<char> buf(Base64DecodeBufSize(b64.Size()), '\0');
        Base64DecodeUneven(buf.begin(), b64);
        TString res(buf.data());
        UNIT_ASSERT_VALUES_EQUAL(x, res);
    }
}
