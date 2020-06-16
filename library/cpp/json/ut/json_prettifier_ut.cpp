#include <library/cpp/json/json_prettifier.h>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(JsonPrettifier) {
    Y_UNIT_TEST(PrettifyJsonShort) {
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson(""), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("null"), "null");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("true"), "true");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("false"), "false");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("1.5"), "1.5");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("test", false, 2, true), "'test'");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[]"), "[ ]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[a]", false, 2), "[\n  \"a\"\n]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[a,b]", false, 2, true), "[\n  'a',\n  'b'\n]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[{},b]", false, 2, true), "[\n  { },\n  'b'\n]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[a,{}]", false, 2, true), "[\n  'a',\n  { }\n]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[{},{}]"), "[\n    { },\n    { }\n]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{}"), "{ }");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{}"), "{ }");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{k:v}", false, 2, true), "{\n  'k' : 'v'\n}");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("Test545", true, 2), "Test545");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'null'", true, 2, true), "'null'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'true'", true, 2, true), "'true'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'false'", true, 2, true), "'false'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'\"'", true, 2, true), "'\"'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'\"'", true, 2, false), "\"\\\"\"");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'\\\''", true, 2, true), "'\\u0027'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'\\\''", true, 2, false), "\"'\"");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'1b'", true, 2, true), "'1b'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("'Test*545'", true, 2, true), "'Test*545'");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{k:v}", true, 2), "{\n  k : v\n}");
    }

    Y_UNIT_TEST(PrettifyJsonLong) {
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[{k:v},{a:b}]", false, 2, true),
                                  "[\n"
                                  "  {\n"
                                  "    'k' : 'v'\n"
                                  "  },\n"
                                  "  {\n"
                                  "    'a' : 'b'\n"
                                  "  }\n"
                                  "]");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{k:v,a:b,x:[1,2,3]}", false, 2, true),
                                  "{\n"
                                  "  'k' : 'v',\n"
                                  "  'a' : 'b',\n"
                                  "  'x' : [\n"
                                  "    1,\n"
                                  "    2,\n"
                                  "    3\n"
                                  "  ]\n"
                                  "}");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{k:v,a:b,x:[1,{f:b},3],m:n}", false, 2, true),
                                  "{\n"
                                  "  'k' : 'v',\n"
                                  "  'a' : 'b',\n"
                                  "  'x' : [\n"
                                  "    1,\n"
                                  "    {\n"
                                  "      'f' : 'b'\n"
                                  "    },\n"
                                  "    3\n"
                                  "  ],\n"
                                  "  'm' : 'n'\n"
                                  "}");

        NJson::TJsonPrettifier prettifierMaxLevel1 = NJson::TJsonPrettifier::Prettifier(false, 2, true);
        prettifierMaxLevel1.MaxPaddingLevel = 1;
        UNIT_ASSERT_STRINGS_EQUAL(prettifierMaxLevel1.Prettify("{k:v,a:b,x:[1,{f:b},3],m:n}"),
                                  "{\n"
                                  "  'k' : 'v',\n"
                                  "  'a' : 'b',\n"
                                  "  'x' : [ 1, { 'f' : 'b' }, 3 ],\n"
                                  "  'm' : 'n'\n"
                                  "}");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{g:{x:{a:{b:c,e:f},q:{x:y}},y:fff}}", true, 2),
                                  "{\n"
                                  "  g : {\n"
                                  "    x : {\n"
                                  "      a : {\n"
                                  "        b : c,\n"
                                  "        e : f\n"
                                  "      },\n"
                                  "      q : {\n"
                                  "        x : y\n"
                                  "      }\n"
                                  "    },\n"
                                  "    y : fff\n"
                                  "  }\n"
                                  "}");

        NJson::TJsonPrettifier prettifierMaxLevel3 = NJson::TJsonPrettifier::Prettifier(true, 2);
        prettifierMaxLevel3.MaxPaddingLevel = 3;
        UNIT_ASSERT_STRINGS_EQUAL(prettifierMaxLevel3.Prettify("{g:{x:{a:{b:c,e:f},q:{x:y}},y:fff}}"),
                                  "{\n"
                                  "  g : {\n"
                                  "    x : {\n"
                                  "      a : { b : c, e : f },\n"
                                  "      q : { x : y }\n"
                                  "    },\n"
                                  "    y : fff\n"
                                  "  }\n"
                                  "}");
    }

    Y_UNIT_TEST(PrettifyJsonInvalid) {
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("}"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("}}"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{}}"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{}}}"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("]"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("]]"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[]]"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[]]]"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("[,,,]"), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::PrettifyJson("{,,,}"), "");
    }

    Y_UNIT_TEST(CompactifyJsonShort) {
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson(""), "");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("null"), "null");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("true"), "true");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("false"), "false");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("1.5"), "1.5");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("test", true), "test");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("test", false), "\"test\"");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("[ ]"), "[]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("[\n  \"a\"\n]", true), "[a]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("[\n  'a',\n  'b'\n]", true), "[a,b]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("[\n  { },\n  'b'\n]", true), "[{},b]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("[\n  'a',\n  { }\n]", true), "[a,{}]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("[\n    { },\n    { }\n]", true), "[{},{}]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("{ }"), "{}");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson("{\n  'k' : 'v'\n}", true), "{k:v}");
    }

    Y_UNIT_TEST(CompactifyJsonLong) {
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson(
                                      "[\n"
                                      "  {\n"
                                      "    'k' : 'v'\n"
                                      "  },\n"
                                      "  {\n"
                                      "    'a' : 'b'\n"
                                      "  }\n"
                                      "]",
                                      true),
                                  "[{k:v},{a:b}]");
        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson(
                                      "{\n"
                                      "  'k' : 'v',\n"
                                      "  'a' : 'b',\n"
                                      "  'x' : [\n"
                                      "    1,\n"
                                      "    2,\n"
                                      "    3\n"
                                      "  ]\n"
                                      "}",
                                      true),
                                  "{k:v,a:b,x:[1,2,3]}");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson(
                                      "{\n"
                                      "  'k' : 'v',\n"
                                      "  'a' : 'b',\n"
                                      "  'x' : [\n"
                                      "    1,\n"
                                      "    {\n"
                                      "      'f' : 'b'\n"
                                      "    },\n"
                                      "    3\n"
                                      "  ],\n"
                                      "  'm' : 'n'\n"
                                      "}",
                                      true),
                                  "{k:v,a:b,x:[1,{f:b},3],m:n}");

        UNIT_ASSERT_STRINGS_EQUAL(NJson::CompactifyJson(
                                      "{\n"
                                      "  g : {\n"
                                      "    x : {\n"
                                      "      a : {\n"
                                      "        b : c,\n"
                                      "        e : f\n"
                                      "      },\n"
                                      "      q : {\n"
                                      "        x : y\n"
                                      "      }\n"
                                      "    },\n"
                                      "    y : fff\n"
                                      "  }\n"
                                      "}",
                                      true),
                                  "{g:{x:{a:{b:c,e:f},q:{x:y}},y:fff}}");
    }
}
