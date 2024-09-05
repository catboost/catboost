#include "cast.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/wide.h>
#include <util/system/defaults.h>

#include <limits>

// positive test (return true or no exception)
#define test1(t, v)       \
    F<t>().CheckTryOK(v); \
    F<t>().CheckOK(v)

// negative test (return false or exception)
#define test2(t, v)         \
    F<t>().CheckTryFail(v); \
    F<t>().CheckExc(v)

#define EPS 10E-7

#define HEX_MACROS_MAP(mac, type, val) mac(type, val, 2) mac(type, val, 8) mac(type, val, 10) mac(type, val, 16)

#define OK_HEX_CHECK(type, val, base) UNIT_ASSERT_EQUAL((IntFromStringForCheck<base>(IntToString<base>(val))), val);
#define EXC_HEX_CHECK(type, val, base) UNIT_ASSERT_EXCEPTION((IntFromString<type, base>(IntToString<base>(val))), yexception);

#define TRY_HEX_MACROS_MAP(mac, type, val, result, def) \
    mac(type, val, result, def, 2)                      \
        mac(type, val, result, def, 8)                  \
            mac(type, val, result, def, 10)             \
                mac(type, val, result, def, 16)

#define TRY_OK_HEX_CHECK(type, val, result, def, base)                                       \
    result = def;                                                                            \
    UNIT_ASSERT_EQUAL(TryIntFromStringForCheck<base>(IntToString<base>(val), result), true); \
    UNIT_ASSERT_EQUAL(result, val);

#define TRY_FAIL_HEX_CHECK(type, val, result, def, base)                                             \
    result = def;                                                                                    \
    UNIT_ASSERT_VALUES_EQUAL(TryIntFromStringForCheck<base>(IntToString<base>(val), result), false); \
    UNIT_ASSERT_VALUES_EQUAL(result, def);

template <class A>
struct TRet {
    template <int base>
    inline A IntFromStringForCheck(const TString& str) {
        return IntFromString<A, base>(str);
    }

    template <int base>
    inline bool TryIntFromStringForCheck(const TString& str, A& result) {
        return TryIntFromString<base>(str, result);
    }

    template <class B>
    inline void CheckOK(B v) {
        UNIT_ASSERT_VALUES_EQUAL(FromString<A>(ToString(v)), v); // char
        UNIT_ASSERT_VALUES_EQUAL(FromString<A>(ToWtring(v)), v); // wide char
        HEX_MACROS_MAP(OK_HEX_CHECK, A, v);
    }

    template <class B>
    inline void CheckExc(B v) {
        UNIT_ASSERT_EXCEPTION(FromString<A>(ToString(v)), yexception); // char
        UNIT_ASSERT_EXCEPTION(FromString<A>(ToWtring(v)), yexception); // wide char
        HEX_MACROS_MAP(EXC_HEX_CHECK, A, v);
    }

    template <class B>
    inline void CheckTryOK(B v) {
        static const A defaultV = 42;
        A convV;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<A>(ToString(v), convV), true); // char
        UNIT_ASSERT_VALUES_EQUAL(v, convV);
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<A>(ToWtring(v), convV), true); // wide char
        UNIT_ASSERT_VALUES_EQUAL(v, convV);

        TRY_HEX_MACROS_MAP(TRY_OK_HEX_CHECK, A, v, convV, defaultV);
    }

    template <class B>
    inline void CheckTryFail(B v) {
        static const A defaultV = 42;
        A convV = defaultV;                                                    // to check that original value is not trashed on bad cast
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<A>(ToString(v), convV), false); // char
        UNIT_ASSERT_VALUES_EQUAL(defaultV, convV);
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<A>(ToWtring(v), convV), false); // wide char
        UNIT_ASSERT_VALUES_EQUAL(defaultV, convV);

        TRY_HEX_MACROS_MAP(TRY_FAIL_HEX_CHECK, A, v, convV, defaultV);
    }
};

template <>
struct TRet<bool> {
    template <class B>
    inline void CheckOK(B v) {
        UNIT_ASSERT_VALUES_EQUAL(FromString<bool>(ToString(v)), v);
    }

    template <class B>
    inline void CheckTryOK(B v) {
        B convV;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<bool>(ToString(v), convV), true);
        UNIT_ASSERT_VALUES_EQUAL(v, convV);
    }

    template <class B>
    inline void CheckExc(B v) {
        UNIT_ASSERT_EXCEPTION(FromString<bool>(ToString(v)), yexception);
    }

    template <class B>
    inline void CheckTryFail(B v) {
        static const bool defaultV = false;
        bool convV = defaultV;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<bool>(ToString(v), convV), false);
        UNIT_ASSERT_VALUES_EQUAL(defaultV, convV);
    }
};

template <class A>
inline TRet<A> F() {
    return TRet<A>();
}

#if 0
template <class T>
inline void CheckConvertToBuffer(const T& value, const size_t size, const TString& canonValue) {
    const size_t maxSize = 256;
    char buffer[maxSize];
    const char magic = 0x7F;
    memset(buffer, magic, maxSize);
    size_t length = 0;
    if (canonValue.size() > size) { // overflow will occur
        UNIT_ASSERT_EXCEPTION(length = ToString(value, buffer, size), yexception);
        // check that no bytes after size was trashed
        for (size_t i = size; i < maxSize; ++i)
            UNIT_ASSERT_VALUES_EQUAL(buffer[i], magic);
    } else {
        length = ToString(value, buffer, size);
        UNIT_ASSERT(length < maxSize);
        // check that no bytes after length was trashed
        for (size_t i = length; i < maxSize; ++i)
            UNIT_ASSERT_VALUES_EQUAL(buffer[i], magic);
        TStringBuf result(buffer, length);
        UNIT_ASSERT_VALUES_EQUAL(result, TStringBuf(canonValue));
    }
}
#endif

Y_UNIT_TEST_SUITE(TCastTest) {
    template <class A>
    inline TRet<A> F() {
        return TRet<A>();
    }

    template <class TFloat>
    void GoodFloatTester(const char* str, const TFloat canonValue, const double eps) {
        TFloat f = canonValue + 42.0; // shift value to make it far from proper
        UNIT_ASSERT_VALUES_EQUAL(TryFromString<TFloat>(str, f), true);
        UNIT_ASSERT_DOUBLES_EQUAL(f, canonValue, eps);
        f = FromString<TFloat>(str);
        UNIT_ASSERT_DOUBLES_EQUAL(f, canonValue, eps);
    }

    template <class TFloat>
    void BadFloatTester(const char* str) {
        const double eps = 10E-5;
        TFloat f = 42.0; // make it far from proper
        auto res = TryFromString<TFloat>(str, f);

        UNIT_ASSERT_VALUES_EQUAL(res, false);
        UNIT_ASSERT_DOUBLES_EQUAL(f, 42.0, eps); // check value was not trashed
        UNIT_ASSERT_EXCEPTION(f = FromString<TFloat>(str), TFromStringException);
        Y_UNUSED(f); // shut up compiler about 'assigned value that is not used'
    }

    Y_UNIT_TEST(TestToFrom) {
        test1(bool, true);
        test1(bool, false);
        test2(bool, "");
        test2(bool, "a");

        test2(ui8, -1);
        test1(i8, -1);
        test1(i8, SCHAR_MAX);
        test1(i8, SCHAR_MIN);
        test1(i8, SCHAR_MAX - 1);
        test1(i8, SCHAR_MIN + 1);
        test2(i8, (int)SCHAR_MAX + 1);
        test2(i8, (int)SCHAR_MIN - 1);
        test1(ui8, UCHAR_MAX);
        test1(ui8, UCHAR_MAX - 1);
        test2(ui8, (int)UCHAR_MAX + 1);
        test2(ui8, -1);
        test1(int, -1);
        test2(unsigned int, -1);
        test1(short int, -1);
        test2(unsigned short int, -1);
        test1(long int, -1);
        test2(unsigned long int, -1);
        test1(int, INT_MAX);
        test1(int, INT_MIN);
        test1(int, INT_MAX - 1);
        test1(int, INT_MIN + 1);
        test2(int, (long long int)INT_MAX + 1);
        test2(int, (long long int)INT_MIN - 1);
        test1(unsigned int, UINT_MAX);
        test1(unsigned int, UINT_MAX - 1);
        test2(unsigned int, (long long int)UINT_MAX + 1);
        test1(short int, SHRT_MAX);
        test1(short int, SHRT_MIN);
        test1(short int, SHRT_MAX - 1);
        test1(short int, SHRT_MIN + 1);
        test2(short int, (long long int)SHRT_MAX + 1);
        test2(short int, (long long int)SHRT_MIN - 1);
        test1(unsigned short int, USHRT_MAX);
        test1(unsigned short int, USHRT_MAX - 1);
        test2(unsigned short int, (long long int)USHRT_MAX + 1);
        test1(long int, LONG_MAX);
        test1(long int, LONG_MIN);
        test1(long int, LONG_MAX - 1);
        test1(long int, LONG_MIN + 1);

        test1(long long int, LLONG_MAX);
        test1(long long int, LLONG_MIN);
        test1(long long int, LLONG_MAX - 1);
        test1(long long int, LLONG_MIN + 1);
    }

    Y_UNIT_TEST(TestVolatile) {
        volatile int x = 1;
        UNIT_ASSERT_VALUES_EQUAL(ToString(x), "1");
    }

    Y_UNIT_TEST(TestStrToD) {
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("1.1", nullptr), 1.1, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("1.12345678", nullptr), 1.12345678, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("10E-5", nullptr), 10E-5, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("1.1E+5", nullptr), 1.1E+5, EPS);

        char* ret = nullptr;

        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("1.1y", &ret), 1.1, EPS);
        UNIT_ASSERT_VALUES_EQUAL(*ret, 'y');
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("1.12345678z", &ret), 1.12345678, EPS);
        UNIT_ASSERT_VALUES_EQUAL(*ret, 'z');
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("10E-5y", &ret), 10E-5, EPS);
        UNIT_ASSERT_VALUES_EQUAL(*ret, 'y');
        UNIT_ASSERT_DOUBLES_EQUAL(StrToD("1.1E+5z", &ret), 1.1E+5, EPS);
        UNIT_ASSERT_VALUES_EQUAL(*ret, 'z');
    }

    Y_UNIT_TEST(TestFloats) {
        // "%g" mode
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.1f, PREC_NDIGITS, 6), "0.1"); // drop trailing zeroes
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.12345678f, PREC_NDIGITS, 6), "0.123457");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1e-20f, PREC_NDIGITS, 6), "1e-20");
        // "%f" mode
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.1f, PREC_POINT_DIGITS, 6), "0.100000");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.12345678f, PREC_POINT_DIGITS, 6), "0.123457");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1e-20f, PREC_POINT_DIGITS, 6), "0.000000");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(12.34f, PREC_POINT_DIGITS, 0), "12"); // rounding to integers drops '.'
        // strip trailing zeroes
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.1f, PREC_POINT_DIGITS_STRIP_ZEROES, 6), "0.1");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.12345678f, PREC_POINT_DIGITS_STRIP_ZEROES, 6), "0.123457");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1e-20f, PREC_POINT_DIGITS_STRIP_ZEROES, 6), "0");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(12.34f, PREC_POINT_DIGITS_STRIP_ZEROES, 0), "12"); // rounding to integers drops '.'
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(10000.0f, PREC_POINT_DIGITS_STRIP_ZEROES, 0), "10000");
        // automatic selection of ndigits
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.1f), "0.1");               // drop trailing zeroes
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(0.12345678f), "0.12345678"); // 8 valid digits
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1000.00006f), "1000.00006"); // 9 valid digits
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1e-45f), "1e-45");           // denormalized: 1 valid digit
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(-0.0f), "-0");               // sign must be preserved
        // version for double
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1.0 / 10000), "0.0001");                    // trailing zeroes
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1.2345678901234567), "1.2345678901234567"); // no truncation
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(5e-324), "5e-324");                         // denormalized
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(-0.0), "-0");                               // sign must be preserved

        UNIT_ASSERT_STRINGS_EQUAL(FloatToString(std::numeric_limits<double>::quiet_NaN()), "nan");
        UNIT_ASSERT_STRINGS_EQUAL(FloatToString(std::numeric_limits<double>::infinity()), "inf");
        UNIT_ASSERT_STRINGS_EQUAL(FloatToString(-std::numeric_limits<double>::infinity()), "-inf");

        UNIT_ASSERT_STRINGS_EQUAL(FloatToString(std::numeric_limits<float>::quiet_NaN()), "nan");
        UNIT_ASSERT_STRINGS_EQUAL(FloatToString(std::numeric_limits<float>::infinity()), "inf");
        UNIT_ASSERT_STRINGS_EQUAL(FloatToString(-std::numeric_limits<float>::infinity()), "-inf");
    }

    Y_UNIT_TEST(TestReadFloats) {
        GoodFloatTester<float>("0.0001", 0.0001f, EPS);
        GoodFloatTester<double>("0.0001", 0.0001, EPS);
        GoodFloatTester<long double>("0.0001", 0.0001, EPS);
        GoodFloatTester<float>("10E-5", 10E-5f, EPS);
        GoodFloatTester<double>("1.0001E5", 1.0001E5, EPS);
        GoodFloatTester<long double>("1.0001e5", 1.0001e5, EPS);
        GoodFloatTester<long double>(".0001e5", .0001e5, EPS);
        BadFloatTester<float>("a10E-5");
        BadFloatTester<float>("10 ");
        BadFloatTester<float>("10\t");
        // BadFloatTester<float>("10E");
        // BadFloatTester<float>("10.E");
        BadFloatTester<float>("..0");
        BadFloatTester<float>(""); // IGNIETFERRO-300
        BadFloatTester<double>("1.00.01");
        BadFloatTester<double>("1.0001E5b");
        BadFloatTester<double>("1.0001s");
        BadFloatTester<double>("1..01");
        BadFloatTester<double>(""); // IGNIETFERRO-300
        BadFloatTester<long double>(".1.00");
        BadFloatTester<long double>("1.00.");
        BadFloatTester<long double>("1.0001e5-");
        BadFloatTester<long double>("10e 2");
        BadFloatTester<long double>(""); // IGNIETFERRO-300
    }

    Y_UNIT_TEST(TestLiteral) {
        UNIT_ASSERT_VALUES_EQUAL(ToString("abc"), TString("abc"));
    }

    Y_UNIT_TEST(TestFromStringStringBuf) {
        TString a = "xyz";
        TStringBuf b = FromString<TStringBuf>(a);
        UNIT_ASSERT_VALUES_EQUAL(a, b);
        UNIT_ASSERT_VALUES_EQUAL((void*)a.data(), (void*)b.data());
    }

#if 0
    Y_UNIT_TEST(TestBufferOverflow) {
        CheckConvertToBuffer<float>(1.f, 5, "1");
        CheckConvertToBuffer<float>(1.005f, 3, "1.005");
        CheckConvertToBuffer<float>(1.00000000f, 3, "1");

        CheckConvertToBuffer<double>(1.f, 5, "1");
        CheckConvertToBuffer<double>(1.005f, 3, "1.005");
        CheckConvertToBuffer<double>(1.00000000f, 3, "1");

        CheckConvertToBuffer<int>(2, 5, "2");
        CheckConvertToBuffer<int>(1005, 3, "1005");

        CheckConvertToBuffer<size_t>(2, 5, "2");
        CheckConvertToBuffer<ui64>(1005000000000000ull, 32, "1005000000000000");
        CheckConvertToBuffer<ui64>(1005000000000000ull, 3, "1005000000000000");

        // TString longNumber = TString("1.") + TString(1 << 20, '1');
        // UNIT_ASSERT_EXCEPTION(FromString<double>(longNumber), yexception);
    }
#endif

    Y_UNIT_TEST(TestWide) {
        TUtf16String iw = u"-100500";
        int iv = 0;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString(iw, iv), true);
        UNIT_ASSERT_VALUES_EQUAL(iv, -100500);

        ui64 uv = 0;
        TUtf16String uw = u"21474836470";
        UNIT_ASSERT_VALUES_EQUAL(TryFromString(uw, uv), true);
        UNIT_ASSERT_VALUES_EQUAL(uv, 21474836470ull);

        TWtringBuf bw(uw.data(), uw.size());
        uv = 0;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString(uw, uv), true);
        UNIT_ASSERT_VALUES_EQUAL(uv, 21474836470ull);

        const wchar16* beg = uw.data();
        uv = 0;
        UNIT_ASSERT_VALUES_EQUAL(TryFromString(beg, uw.size(), uv), true);
        UNIT_ASSERT_VALUES_EQUAL(uv, 21474836470ull);
    }

    Y_UNIT_TEST(TestDefault) {
        size_t res = 0;
        const size_t def1 = 42;

        TString s1("100500");
        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s1, res, def1), true);
        UNIT_ASSERT_VALUES_EQUAL(res, 100500);

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s1, res), true);
        UNIT_ASSERT_VALUES_EQUAL(res, 100500);

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault("100500", res, def1), true);
        UNIT_ASSERT_VALUES_EQUAL(res, 100500);

        UNIT_CHECK_GENERATED_NO_EXCEPTION(FromStringWithDefault(s1, def1), yexception);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault(s1, def1), 100500);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault<size_t>(s1), 100500);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault("100500", def1), 100500);

        TString s2("100q500");
        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s2, res), false);
        UNIT_ASSERT_VALUES_EQUAL(res, size_t());

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s2, res, def1), false);
        UNIT_ASSERT_VALUES_EQUAL(res, def1);

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault("100q500", res), false);
        UNIT_ASSERT_VALUES_EQUAL(res, size_t());

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault("100 500", res), false);
        UNIT_ASSERT_VALUES_EQUAL(res, size_t());

        UNIT_CHECK_GENERATED_NO_EXCEPTION(FromStringWithDefault(s2, def1), yexception);
        UNIT_CHECK_GENERATED_NO_EXCEPTION(FromStringWithDefault("100q500", def1), yexception);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault(s2, def1), def1);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault<size_t>(s2), size_t());
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault<size_t>("100q500"), size_t());
        UNIT_CHECK_GENERATED_EXCEPTION(FromString<size_t>(s2), TFromStringException);

        int res2 = 0;
        const int def2 = -6;

        TUtf16String s3 = u"-100500";
        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s3, res2, def2), true);
        UNIT_ASSERT_VALUES_EQUAL(res2, -100500);

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s3, res2), true);
        UNIT_ASSERT_VALUES_EQUAL(res2, -100500);

        UNIT_CHECK_GENERATED_NO_EXCEPTION(FromStringWithDefault(s3, def1), yexception);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault(s3, def2), -100500);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault<size_t>(s3), size_t());

        TUtf16String s4 = u"-f100500";
        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s4, res2, def2), false);
        UNIT_ASSERT_VALUES_EQUAL(res2, def2);

        UNIT_ASSERT_VALUES_EQUAL(TryFromStringWithDefault(s4, res2), false);
        UNIT_ASSERT_VALUES_EQUAL(res2, size_t());

        UNIT_CHECK_GENERATED_NO_EXCEPTION(FromStringWithDefault(s4, def2), yexception);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault(s4, def2), def2);
        UNIT_CHECK_GENERATED_EXCEPTION(FromString<size_t>(s4), yexception);
        UNIT_ASSERT_VALUES_EQUAL(FromStringWithDefault<size_t>(s4), size_t());
    }

    Y_UNIT_TEST(TestMaybe) {
        TMaybe<int> res;

        TString s1("100500");
        UNIT_CHECK_GENERATED_NO_EXCEPTION(res = TryFromString<int>(s1), yexception);
        UNIT_ASSERT_VALUES_EQUAL(res, 100500);

        UNIT_ASSERT_VALUES_EQUAL(TryFromString<int>("100500"), 100500);

        TString s2("100q500");
        UNIT_CHECK_GENERATED_NO_EXCEPTION(res = TryFromString<int>(s2), yexception);
        UNIT_ASSERT(res.Empty());

        TUtf16String s3 = u"-100500";
        UNIT_CHECK_GENERATED_NO_EXCEPTION(res = TryFromString<size_t>(s3), yexception);
        UNIT_ASSERT(res.Empty());

        TUtf16String s4 = u"-f100500";
        UNIT_CHECK_GENERATED_NO_EXCEPTION(res = TryFromString<int>(s4), yexception);
        UNIT_ASSERT(res.Empty());

        std::string s5 = "100500";
        UNIT_CHECK_GENERATED_NO_EXCEPTION(res = TryFromString<int>(s5), yexception);
        UNIT_ASSERT_VALUES_EQUAL(res, 100500);
    }

    Y_UNIT_TEST(TestBool) {
        // True cases
        UNIT_ASSERT_VALUES_EQUAL(FromString<bool>("yes"), true);
        UNIT_ASSERT_VALUES_EQUAL(FromString<bool>("1"), true);
        // False cases
        UNIT_ASSERT_VALUES_EQUAL(FromString<bool>("no"), false);
        UNIT_ASSERT_VALUES_EQUAL(FromString<bool>("0"), false);
        // Strange cases
        UNIT_ASSERT_EXCEPTION(FromString<bool>(""), yexception);
        UNIT_ASSERT_EXCEPTION(FromString<bool>("something"), yexception);
    }

    Y_UNIT_TEST(TestAutoDetectType) {
        UNIT_ASSERT_DOUBLES_EQUAL((float)FromString("0.0001"), 0.0001, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL((double)FromString("0.0015", sizeof("0.0015") - 2), 0.001, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL((long double)FromString(TStringBuf("0.0001")), 0.0001, EPS);
        UNIT_ASSERT_DOUBLES_EQUAL((float)FromString(TString("10E-5")), 10E-5, EPS);
        UNIT_ASSERT_VALUES_EQUAL((bool)FromString("da"), true);
        UNIT_ASSERT_VALUES_EQUAL((bool)FromString("no"), false);
        UNIT_ASSERT_VALUES_EQUAL((short)FromString(u"9000"), 9000);
        UNIT_ASSERT_VALUES_EQUAL((int)FromString(u"-100500"), -100500);
        UNIT_ASSERT_VALUES_EQUAL((unsigned long long)FromString(TWtringBuf(u"42", 1)), 4);
        int integer = FromString("125");
        ui16 wideCharacterCode = FromString(u"125");
        UNIT_ASSERT_VALUES_EQUAL(integer, wideCharacterCode);
    }

    Y_UNIT_TEST(ErrorMessages) {
        UNIT_ASSERT_EXCEPTION_CONTAINS(FromString<ui32>(""), TFromStringException, "empty string as number");
        UNIT_ASSERT_EXCEPTION_CONTAINS(FromString<ui32>("-"), TFromStringException, "Unexpected symbol \"-\" at pos 0 in string \"-\"");
        UNIT_ASSERT_EXCEPTION_CONTAINS(FromString<i32>("-"), TFromStringException, "Cannot parse string \"-\" as number");
        UNIT_ASSERT_EXCEPTION_CONTAINS(FromString<i32>("+"), TFromStringException, "Cannot parse string \"+\" as number");
        UNIT_ASSERT_EXCEPTION_CONTAINS(FromString<i32>("0.328413745072"), TFromStringException, "Unexpected symbol \".\" at pos 1 in string \"0.328413745072\"");
    }

    Y_UNIT_TEST(TryStringBuf) {
        {
            constexpr TStringBuf hello = "hello";
            TStringBuf out;
            UNIT_ASSERT(TryFromString(hello, out));
            UNIT_ASSERT_VALUES_EQUAL(hello, out);
        }
        {
            constexpr TStringBuf empty = "";
            TStringBuf out;
            UNIT_ASSERT(TryFromString(empty, out));
            UNIT_ASSERT_VALUES_EQUAL(empty, out);
        }
        {
            constexpr TStringBuf empty;
            TStringBuf out;
            UNIT_ASSERT(TryFromString(empty, out));
            UNIT_ASSERT_VALUES_EQUAL(empty, out);
        }
        {
            const auto hello = u"hello";
            TWtringBuf out;
            UNIT_ASSERT(TryFromString(hello, out));
            UNIT_ASSERT_VALUES_EQUAL(hello, out);
        }
        {
            const TUtf16String empty;
            TWtringBuf out;
            UNIT_ASSERT(TryFromString(empty, out));
            UNIT_ASSERT_VALUES_EQUAL(empty, out);
        }
        {
            constexpr TWtringBuf empty;
            TWtringBuf out;
            UNIT_ASSERT(TryFromString(empty, out));
            UNIT_ASSERT_VALUES_EQUAL(empty, out);
        }
    }

    Y_UNIT_TEST(Nan) {
        double xx = 0;

        UNIT_ASSERT(!TryFromString("NaN", xx));
        UNIT_ASSERT(!TryFromString("NAN", xx));
        UNIT_ASSERT(!TryFromString("nan", xx));
    }

    Y_UNIT_TEST(Infinity) {
        double xx = 0;

        UNIT_ASSERT(!TryFromString("Infinity", xx));
        UNIT_ASSERT(!TryFromString("INFINITY", xx));
        UNIT_ASSERT(!TryFromString("infinity", xx));
    }

    Y_UNIT_TEST(TestBorderCases) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(0.0), "0");
        UNIT_ASSERT_VALUES_EQUAL(ToString(1.0), "1");
        UNIT_ASSERT_VALUES_EQUAL(ToString(10.0), "10");
        UNIT_ASSERT_VALUES_EQUAL(ToString(NAN), "nan");
        UNIT_ASSERT_VALUES_EQUAL(ToString(-NAN), "nan");
        UNIT_ASSERT_VALUES_EQUAL(ToString(INFINITY), "inf");
        UNIT_ASSERT_VALUES_EQUAL(ToString(-INFINITY), "-inf");
        UNIT_ASSERT_VALUES_EQUAL(ToString(1.1e+100), "1.1e+100");
        UNIT_ASSERT_VALUES_EQUAL(ToString(1e+100), "1e+100");
        UNIT_ASSERT_VALUES_EQUAL(ToString(87423.2031250000001), "87423.20313");
        UNIT_ASSERT_VALUES_EQUAL(FloatToString(1.0e60, PREC_POINT_DIGITS_STRIP_ZEROES, 0), "1e+60");
    }

    Y_UNIT_TEST(TestChar) {
        // Given a character ch, ToString(ch) returns
        // the decimal representation of its integral value

        // char
        UNIT_ASSERT_VALUES_EQUAL(ToString('\0'), "0");
        UNIT_ASSERT_VALUES_EQUAL(ToString('0'), "48");

        // wchar16
        UNIT_ASSERT_VALUES_EQUAL(ToString(u'\0'), "0");
        UNIT_ASSERT_VALUES_EQUAL(ToString(u'0'), "48");
        UNIT_ASSERT_VALUES_EQUAL(ToString(u'я'), "1103");
        UNIT_ASSERT_VALUES_EQUAL(ToString(u'\uFFFF'), "65535");

        // wchar32
        UNIT_ASSERT_VALUES_EQUAL(ToString(U'\0'), "0");
        UNIT_ASSERT_VALUES_EQUAL(ToString(U'0'), "48");
        UNIT_ASSERT_VALUES_EQUAL(ToString(U'я'), "1103");
        UNIT_ASSERT_VALUES_EQUAL(ToString(U'\U0001F600'), "128512"); // 'GRINNING FACE' (U+1F600)
    }

    Y_UNIT_TEST(TestTIntStringBuf) {
        static_assert(TStringBuf(TIntStringBuf(111)) == TStringBuf("111"));
        static_assert(TStringBuf(TIntStringBuf(-111)) == TStringBuf("-111"));
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf(0)), "0"sv);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf(1111)), "1111"sv);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf(-1)), "-1"sv);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf(-1111)), "-1111"sv);

        constexpr auto v = TIntStringBuf(-1111);
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(v), TStringBuf(ToString(-1111)));
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf<ui16>(65535)), TStringBuf("65535"));
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf<i16>(32767)), TStringBuf("32767"));
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf<i32>(-32768)), TStringBuf("-32768"));

        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf<i8, 2>(127)), TStringBuf("1111111"));
        UNIT_ASSERT_VALUES_EQUAL(TStringBuf(TIntStringBuf<i8, 2>(-128)), TStringBuf("-10000000"));
    }

    Y_UNIT_TEST(TestTrivial) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(ToString(ToString("abc"))), TString("abc"));

        // May cause compilation error:
        // const TString& ref = ToString(TString{"foo"});

        const TString ok = ToString(TString{"foo"});
        UNIT_ASSERT_VALUES_EQUAL(ok, "foo");
        UNIT_ASSERT_VALUES_EQUAL(ToString(ToString(ok)), "foo");
    }
} // Y_UNIT_TEST_SUITE(TCastTest)
