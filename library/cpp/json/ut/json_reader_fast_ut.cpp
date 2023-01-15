#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_prettifier.h>
#include <library/cpp/testing/unittest/registar.h>

#include <library/cpp/string_utils/relaxed_escaper/relaxed_escaper.h>
#include <util/string/cast.h>
#include <util/string/printf.h>

namespace NJson {
    namespace NTest {
        enum ETestEvent {
            E_NO_EVENT = 0,
            E_ERROR = 1,
            E_DICT_OPEN,
            E_DICT_CLOSE,
            E_ARR_OPEN,
            E_ARR_CLOSE,
            E_NULL,
            E_BOOL,
            E_FLT,
            E_INT,
            E_LONG_LONG,
            E_STR,
            E_KEY
        };

        struct TEvent {
            ETestEvent Type = E_NO_EVENT;

            i64 INum = 0;
            double DNum = 0;
            TString Str;

            TEvent(ETestEvent e = E_NO_EVENT)
                : Type(e)
            {
            }

            TEvent(double v, ETestEvent e)
                : Type(e)
                , DNum(v)
            {
            }

            TEvent(i64 v, ETestEvent e)
                : Type(e)
                , INum(v)
            {
            }

            TEvent(TStringBuf t, ETestEvent e)
                : Type(e)
                , Str(NEscJ::EscapeJ<true, false>(t))
            {
            }

            TString ToString() const {
                switch (Type) {
                    default:
                        return "YOUFAILED";
                    case E_ERROR:
                        return Sprintf("error: %s", Str.data());
                    case E_DICT_OPEN:
                        return "{";
                    case E_DICT_CLOSE:
                        return "}";
                    case E_ARR_OPEN:
                        return "[";
                    case E_ARR_CLOSE:
                        return "]";
                    case E_NULL:
                        return "null";
                    case E_BOOL:
                        return INum ? "true" : "false";
                    case E_INT:
                        return ::ToString(INum);
                    case E_FLT:
                        return ::ToString(DNum);
                    case E_STR:
                        return Sprintf("%s", Str.data());
                    case E_KEY:
                        return Sprintf("key: %s", Str.data());
                }
            }
        };

        using TEvents = TVector<TEvent>;

        struct TTestHandler : TJsonCallbacks {
            TEvents Events;

            bool OnOpenMap() override {
                Events.push_back(E_DICT_OPEN);
                return true;
            }

            bool OnCloseMap() override {
                Events.push_back(E_DICT_CLOSE);
                return true;
            }

            bool OnOpenArray() override {
                Events.push_back(E_ARR_OPEN);
                return true;
            }

            bool OnCloseArray() override {
                Events.push_back(E_ARR_CLOSE);
                return true;
            }

            bool OnNull() override {
                Events.push_back(E_NULL);
                return true;
            }

            bool OnBoolean(bool v) override {
                Events.push_back(TEvent((i64)v, E_BOOL));
                return true;
            }

            bool OnInteger(long long v) override {
                Events.push_back(TEvent((i64)v, E_INT));
                return true;
            }

            bool OnUInteger(unsigned long long v) override {
                return OnInteger(v);
            }

            bool OnDouble(double v) override {
                Events.push_back(TEvent(v, E_FLT));
                return true;
            }

            bool OnString(const TStringBuf& v) override {
                Events.push_back(TEvent(v, E_STR));
                return true;
            }

            bool OnMapKey(const TStringBuf& v) override {
                Events.push_back(TEvent(v, E_KEY));
                return true;
            }

            void OnError(size_t, TStringBuf token) override {
                Events.push_back(TEvent(token, E_ERROR));
            }

            void Assert(const TEvents& e, TString str) {
                try {
                    UNIT_ASSERT_VALUES_EQUAL_C(e.size(), Events.size(), str);

                    for (ui32 i = 0, sz = e.size(); i < sz; ++i) {
                        UNIT_ASSERT_VALUES_EQUAL_C((int)e[i].Type, (int)Events[i].Type, Sprintf("'%s' %u", str.data(), i));
                        UNIT_ASSERT_VALUES_EQUAL_C(e[i].INum, Events[i].INum, Sprintf("'%s' %u", str.data(), i));
                        UNIT_ASSERT_VALUES_EQUAL_C(e[i].DNum, Events[i].DNum, Sprintf("'%s' %u", str.data(), i));
                        UNIT_ASSERT_VALUES_EQUAL_C(e[i].Str, Events[i].Str, Sprintf("'%s' %u", str.data(), i));
                    }
                } catch (const yexception&) {
                    Clog << "Exception at '" << str << "'" << Endl;
                    for (const auto& event : Events) {
                        Clog << event.ToString() << Endl;
                    }

                    throw;
                }
            }
        };
    }
}

class TFastJsonTest: public TTestBase {
    UNIT_TEST_SUITE(TFastJsonTest)
    UNIT_TEST(TestParse)
    UNIT_TEST(TestReadJsonFastTree)
    UNIT_TEST(TestNoInlineComment)
    UNIT_TEST_SUITE_END();

public:
    template <bool accept>
    void DoTestParse(TStringBuf json, ui32 amount, ...) {
        using namespace NJson::NTest;
        TEvents evs;
        va_list vl;
        va_start(vl, amount);
        for (ui32 i = 0; i < amount; i++) {
            ETestEvent e = (ETestEvent)va_arg(vl, int);

            switch ((int)e) {
                case E_NO_EVENT:
                case E_DICT_OPEN:
                case E_DICT_CLOSE:
                case E_ARR_OPEN:
                case E_ARR_CLOSE:
                case E_NULL:
                    evs.push_back(e);
                    break;
                case E_BOOL: {
                    bool v = va_arg(vl, int);
                    evs.push_back(TEvent((i64)v, E_BOOL));
                    break;
                }
                case E_INT: {
                    i64 i = va_arg(vl, int);
                    evs.push_back(TEvent(i, E_INT));
                    break;
                }
                case E_LONG_LONG: {
                    i64 i = va_arg(vl, long long);
                    evs.push_back(TEvent(i, E_INT));
                    break;
                }
                case E_FLT: {
                    double f = va_arg(vl, double);
                    evs.push_back(TEvent(f, E_FLT));
                    break;
                }
                case E_STR: {
                    const char* s = va_arg(vl, const char*);
                    evs.push_back(TEvent(TStringBuf(s), E_STR));
                    break;
                }
                case E_KEY:
                case E_ERROR: {
                    const char* s = va_arg(vl, const char*);
                    evs.push_back(TEvent(TStringBuf(s), e));
                    break;
                }
            }
        }
        va_end(vl);

        TTestHandler h;
        const bool res = ReadJsonFast(json, &h);
        UNIT_ASSERT_VALUES_EQUAL_C(res, accept, Sprintf("%s (%s)", ToString(json).data(), h.Events.back().Str.data()));
        h.Assert(evs, ToString(json));
    }

    void TestParse() {
        using namespace NJson::NTest;

        DoTestParse<true>("", 0);
        DoTestParse<true>(" \t \t ", 0);
        DoTestParse<true>("a-b-c@аб_вгд909AБ", 1, E_STR, "a-b-c@аб_вгд909AБ");
        DoTestParse<true>("'я тестовая строка'", 1, E_STR, "я тестовая строка");
        DoTestParse<true>("\"я тестовая строка\"", 1, E_STR, "я тестовая строка");
        DoTestParse<true>("'\\xA\\xA\\xA'", 1, E_STR, "\n\n\n");
        DoTestParse<true>("12.15", 1, E_FLT, 12.15);
        DoTestParse<true>("null", 1, E_NULL);
        DoTestParse<true>("true", 1, E_BOOL, true);
        DoTestParse<true>("false", 1, E_BOOL, false);
        DoTestParse<true>("[]", 2, E_ARR_OPEN, E_ARR_CLOSE);
        DoTestParse<true>("[ a ]", 3, E_ARR_OPEN, E_STR, "a", E_ARR_CLOSE);
        DoTestParse<true>("[ a, b ]", 4, E_ARR_OPEN, E_STR, "a", E_STR, "b", E_ARR_CLOSE);
        DoTestParse<true>("[a,b]", 4, E_ARR_OPEN, E_STR, "a", E_STR, "b", E_ARR_CLOSE);
        DoTestParse<false>("[a,b][a,b]", 5, E_ARR_OPEN, E_STR, "a", E_STR, "b", E_ARR_CLOSE, E_ERROR, "invalid syntax at token: '['");
        DoTestParse<false>("[a,,b]", 3, E_ARR_OPEN, E_STR, "a", E_ERROR, "invalid syntax at token: ','");
        DoTestParse<true>("{ k : v }", 4, E_DICT_OPEN, E_KEY, "k", E_STR, "v", E_DICT_CLOSE);
        DoTestParse<true>("{a:'\\b'/*comment*/, k /*comment*/\n : v }", 6, E_DICT_OPEN, E_KEY, "a", E_STR, "\b", E_KEY, "k", E_STR, "v", E_DICT_CLOSE);
        DoTestParse<true>("{a:.15, k : v }", 6, E_DICT_OPEN, E_KEY, "a", E_FLT, .15, E_KEY, "k", E_STR, "v", E_DICT_CLOSE);
        DoTestParse<true>("[ a, -.1e+5, 1E-7]", 5, E_ARR_OPEN, E_STR, "a", E_FLT, -.1e+5, E_FLT, 1e-7, E_ARR_CLOSE);
        DoTestParse<true>("{}", 2, E_DICT_OPEN, E_DICT_CLOSE);
        DoTestParse<true>("{ a : x, b : [ c, d, ] }", 9, E_DICT_OPEN, E_KEY, "a", E_STR, "x", E_KEY, "b", E_ARR_OPEN, E_STR, "c", E_STR, "d", E_ARR_CLOSE, E_DICT_CLOSE);
        DoTestParse<false>("{ a : x, b : [ c, d,, ] }", 8, E_DICT_OPEN, E_KEY, "a", E_STR, "x", E_KEY, "b", E_ARR_OPEN, E_STR, "c", E_STR, "d", E_ERROR, "invalid syntax at token: ','");
        //        DoTestParse<false>("{ a : x : y }", 4, E_DICT_OPEN
        //                    , E_KEY, "a", E_STR, "x"
        //                    , E_ERROR
        //                    , ":");
        //        DoTestParse<false>("{queries:{ref:[]},{nonref:[]}}", 8, E_DICT_OPEN
        //                           , E_KEY, "queries", E_DICT_OPEN
        //                               , E_KEY, "ref", E_ARR_OPEN, E_ARR_CLOSE
        //                               , E_DICT_CLOSE, E_ERROR, "");
        DoTestParse<true>("'100x00'", 1, E_STR, "100x00");
        DoTestParse<true>("-1", 1, E_INT, -1);
        DoTestParse<true>("-9223372036854775808", 1, E_LONG_LONG, (long long)Min<i64>());
        DoTestParse<false>("100x00", 1, E_ERROR, "invalid syntax at token: '100x'");
        DoTestParse<false>("100 200", 2, E_INT, 100, E_ERROR, "invalid syntax at token: '200'");
        DoTestParse<true>("{g:{x:{a:{b:c,e:f},q:{x:y}},y:fff}}", 22, E_DICT_OPEN, E_KEY, "g", E_DICT_OPEN, E_KEY, "x", E_DICT_OPEN, E_KEY, "a", E_DICT_OPEN, E_KEY, "b", E_STR, "c", E_KEY, "e", E_STR, "f", E_DICT_CLOSE, E_KEY, "q", E_DICT_OPEN, E_KEY, "x", E_STR, "y", E_DICT_CLOSE, E_DICT_CLOSE, E_KEY, "y", E_STR, "fff", E_DICT_CLOSE, E_DICT_CLOSE);
    }

    void TestReadJsonFastTree() {
        const TString json = R"(
            {
                "a": {
                    "b": {}
                 }
            }}
        )";
        NJson::TJsonValue value;
        UNIT_ASSERT(!ReadJsonFastTree(json, &value));
    }

    void TestNoInlineComment() {
        using namespace NJson::NTest;
        DoTestParse<false>("{\"a\":1}//d{\"b\":2}", 5, E_DICT_OPEN, E_KEY, "a", E_INT, 1, E_DICT_CLOSE, E_ERROR, "invalid syntax at token: '/'");
        DoTestParse<false>("{\"a\":1}//d{\"b\":2}\n", 5, E_DICT_OPEN, E_KEY, "a", E_INT, 1, E_DICT_CLOSE, E_ERROR, "invalid syntax at token: '/'");
        DoTestParse<false>("{\"a\":{//d{\"b\":2}\n}}", 4, E_DICT_OPEN, E_KEY, "a", E_DICT_OPEN, E_ERROR, "invalid syntax at token: '/'");
        DoTestParse<false>("{\"a\":{//d{\"b\":2}}}\n", 4, E_DICT_OPEN, E_KEY, "a", E_DICT_OPEN, E_ERROR, "invalid syntax at token: '/'");
        DoTestParse<false>("{\"a\":{//d{\"b\":2}}}", 4, E_DICT_OPEN, E_KEY, "a", E_DICT_OPEN, E_ERROR, "invalid syntax at token: '/'");
    }
};

UNIT_TEST_SUITE_REGISTRATION(TFastJsonTest)
