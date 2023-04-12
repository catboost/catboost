#include <library/cpp/testing/unittest/registar.h>
#include <util/system/sanitizers.h>

#include "json.h"
#include <library/cpp/json/json_value.h>

#include <limits>

Y_UNIT_TEST_SUITE(JsonWriter) {
    Y_UNIT_TEST(Struct) {
        NJsonWriter::TBuf w;
        w.BeginList();
        w.BeginObject()
            .WriteKey("key")
            .WriteString("value")
            .UnsafeWritePair("\"xk\":13")
            .WriteKey("key2")
            .BeginList()
            .BeginObject()
            .EndObject()
            .BeginObject()
            .EndObject()
            .EndList()
            .EndObject();
        w.WriteInt(43);
        w.UnsafeWriteValue("\"x\"");
        w.WriteString("...");
        w.EndList();
        const char* exp = "[{\"key\":\"value\",\"xk\":13,\"key2\":[{},{}]},43,\"x\",\"...\"]";
        UNIT_ASSERT_EQUAL(w.Str(), exp);
    }
    Y_UNIT_TEST(EscapedString) {
        NJsonWriter::TBuf w(NJsonWriter::HEM_ESCAPE_HTML);
        w.WriteString(" \n \r \t \007 \b \f ' <tag> &ent; \"txt\" ");
        TString ws = w.Str();
        const char* exp = "\" \\n \\r \\t \\u0007 \\b \\f &#39; &lt;tag&gt; &amp;ent; &quot;txt&quot; \"";
        UNIT_ASSERT_STRINGS_EQUAL(ws.c_str(), exp);
    }
    Y_UNIT_TEST(UnescapedString) {
        NJsonWriter::TBuf w;
        w.WriteString(" \n \r \t \b \f '; -- <tag> &ent; \"txt\"", NJsonWriter::HEM_DONT_ESCAPE_HTML);
        TString ws = w.Str();
        const char* exp = "\" \\n \\r \\t \\b \\f \\u0027; -- \\u003Ctag\\u003E &ent; \\\"txt\\\"\"";
        UNIT_ASSERT_STRINGS_EQUAL(ws.c_str(), exp);
    }
    Y_UNIT_TEST(UnescapedChaining) {
        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML);
        w.UnsafeWriteRawBytes("(", 1);
        w.BeginList().WriteString("<>&'\\").BeginList();
        w.EndList().EndList();
        TString ws = w.Str();
        const char* exp = "([\"\\u003C\\u003E&\\u0027\\\\\",[]]";
        UNIT_ASSERT_STRINGS_EQUAL(ws.c_str(), exp);
    }
    Y_UNIT_TEST(Utf8) {
        TString ws = NJsonWriter::TBuf().WriteString("яЯ σΣ ש א").Str();
        const char* exp = "\"яЯ σΣ ש א\"";
        UNIT_ASSERT_STRINGS_EQUAL(ws.c_str(), exp);
    }
    Y_UNIT_TEST(WrongObject) {
        NJsonWriter::TBuf w;
        w.BeginObject();
        UNIT_ASSERT_EXCEPTION(w.WriteString("hehe"), NJsonWriter::TError);
    }
    Y_UNIT_TEST(WrongList) {
        NJsonWriter::TBuf w;
        w.BeginList();
        UNIT_ASSERT_EXCEPTION(w.WriteKey("hehe"), NJsonWriter::TError);
    }
    Y_UNIT_TEST(Incomplete) {
        NJsonWriter::TBuf w;
        w.BeginList();
        UNIT_ASSERT_EXCEPTION(w.Str(), NJsonWriter::TError);
    }
    Y_UNIT_TEST(BareKey) {
        NJsonWriter::TBuf w;
        w.BeginObject()
            .CompatWriteKeyWithoutQuotes("p")
            .WriteInt(1)
            .CompatWriteKeyWithoutQuotes("n")
            .WriteInt(0)
            .EndObject();
        TString ws = w.Str();
        const char* exp = "{p:1,n:0}";
        UNIT_ASSERT_STRINGS_EQUAL(ws.c_str(), exp);
    }
    Y_UNIT_TEST(UnescapedStringInObject) {
        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML);
        w.BeginObject().WriteKey("key").WriteString("</&>'").EndObject();
        TString ws = w.Str();
        const char* exp = "{\"key\":\"\\u003C\\/&\\u003E\\u0027\"}";
        UNIT_ASSERT_STRINGS_EQUAL(ws.c_str(), exp);
    }
    Y_UNIT_TEST(ForeignStreamStr) {
        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML, &Cerr);
        UNIT_ASSERT_EXCEPTION(w.Str(), NJsonWriter::TError);
    }
    Y_UNIT_TEST(ForeignStreamValue) {
        TStringStream ss;
        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML, &ss);
        w.WriteInt(1543);
        UNIT_ASSERT_STRINGS_EQUAL(ss.Str(), "1543");
    }
    Y_UNIT_TEST(Indentation) {
        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML);
        w.SetIndentSpaces(2);
        w.BeginList()
            .WriteInt(1)
            .WriteString("hello")
            .BeginObject()
            .WriteKey("abc")
            .WriteInt(3)
            .WriteKey("def")
            .WriteInt(4)
            .EndObject()
            .EndList();
        const char* exp = "[\n"
                          "  1,\n"
                          "  \"hello\",\n"
                          "  {\n"
                          "    \"abc\":3,\n"
                          "    \"def\":4\n"
                          "  }\n"
                          "]";
        UNIT_ASSERT_STRINGS_EQUAL(exp, w.Str());
    }
    Y_UNIT_TEST(WriteJsonValue) {
        using namespace NJson;
        TJsonValue val;
        val.AppendValue(1);
        val.AppendValue("2");
        val.AppendValue(3.5);
        TJsonValue obj;
        obj.InsertValue("key", TJsonValue("value"));

        val.AppendValue(obj);
        val.AppendValue(TJsonValue(JSON_NULL));

        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML);
        w.WriteJsonValue(&val);

        const char exp[] = "[1,\"2\",3.5,{\"key\":\"value\"},null]";
        UNIT_ASSERT_STRINGS_EQUAL(exp, w.Str());
    }
    Y_UNIT_TEST(WriteJsonValueSorted) {
        using namespace NJson;
        TJsonValue val;
        val.InsertValue("1", TJsonValue(1));
        val.InsertValue("2", TJsonValue(2));

        TJsonValue obj;
        obj.InsertValue("zero", TJsonValue(0));
        obj.InsertValue("succ", TJsonValue(1));
        val.InsertValue("0", obj);

        NJsonWriter::TBuf w(NJsonWriter::HEM_DONT_ESCAPE_HTML);
        w.WriteJsonValue(&val, true);

        const char exp[] = "{\"0\":{\"succ\":1,\"zero\":0},\"1\":1,\"2\":2}";
        UNIT_ASSERT_STRINGS_EQUAL(exp, w.Str());
    }
    Y_UNIT_TEST(Unescaped) {
        NJsonWriter::TBuf buf(NJsonWriter::HEM_UNSAFE);
        buf.WriteString("</security>'");
        UNIT_ASSERT_STRINGS_EQUAL("\"</security>'\"", buf.Str());
    }
    Y_UNIT_TEST(LittleBobbyJsonp) {
        NJsonWriter::TBuf buf;
        buf.WriteString("hello\xe2\x80\xa8\xe2\x80\xa9stranger");
        UNIT_ASSERT_STRINGS_EQUAL("\"hello\\u2028\\u2029stranger\"", buf.Str());
    }
    Y_UNIT_TEST(LittleBobbyInvalid) {
        NJsonWriter::TBuf buf;
        TStringBuf incomplete("\xe2\x80\xa8", 2);
        buf.WriteString(incomplete);
        // garbage in - garbage out
        UNIT_ASSERT_STRINGS_EQUAL("\"\xe2\x80\"", buf.Str());
    }
    Y_UNIT_TEST(OverlyZealous) {
        NJsonWriter::TBuf buf;
        buf.WriteString("—");
        UNIT_ASSERT_STRINGS_EQUAL("\"—\"", buf.Str());
    }
    Y_UNIT_TEST(RelaxedEscaping) {
        NJsonWriter::TBuf buf(NJsonWriter::HEM_RELAXED);
        buf.WriteString("</>");
        UNIT_ASSERT_STRINGS_EQUAL("\"\\u003C/\\u003E\"", buf.Str());
    }

    Y_UNIT_TEST(FloatFormatting) {
        NJsonWriter::TBuf buf(NJsonWriter::HEM_DONT_ESCAPE_HTML);
        buf.BeginList()
            .WriteFloat(0.12345678987654321f)
            .WriteDouble(0.12345678987654321)
            .WriteFloat(0.315501, PREC_NDIGITS, 3)
            .WriteFloat(244.13854, PREC_NDIGITS, 4)
            .WriteFloat(10385.8324, PREC_POINT_DIGITS, 2)
            .BeginObject()
            .WriteKey("1")
            .WriteDouble(1111.71, PREC_POINT_DIGITS, 0)
            .WriteKey("2")
            .WriteDouble(1111.71, PREC_NDIGITS, 1)
            .EndObject()
            .EndList();
        const char exp[] = "[0.123457,0.1234567899,0.316,244.1,10385.83,{\"1\":1112,\"2\":1e+03}]";
        UNIT_ASSERT_STRINGS_EQUAL(exp, buf.Str());
    }

    Y_UNIT_TEST(NanFormatting) {
        {
            NJsonWriter::TBuf buf;
            buf.BeginObject();
            buf.WriteKey("nanvalue");
            UNIT_ASSERT_EXCEPTION(buf.WriteFloat(std::numeric_limits<double>::quiet_NaN()), yexception);
        }

        {
            NJsonWriter::TBuf buf;
            buf.BeginObject();
            buf.WriteKey("infvalue");
            UNIT_ASSERT_EXCEPTION(buf.WriteFloat(std::numeric_limits<double>::infinity()), yexception);
        }

        {
            NJsonWriter::TBuf buf;
            buf.BeginList();
            UNIT_ASSERT_EXCEPTION(buf.WriteFloat(std::numeric_limits<double>::quiet_NaN()), yexception);
        }

        {
            NJsonWriter::TBuf buf;
            buf.BeginList();
            UNIT_ASSERT_EXCEPTION(buf.WriteFloat(std::numeric_limits<double>::infinity()), yexception);
        }

        {
            NJsonWriter::TBuf buf;
            buf.SetWriteNanAsString();

            buf.BeginObject()
                .WriteKey("nanvalue")
                .WriteFloat(std::numeric_limits<double>::quiet_NaN())
                .WriteKey("infvalue")
                .WriteFloat(std::numeric_limits<double>::infinity())
                .WriteKey("minus_infvalue")
                .WriteFloat(-std::numeric_limits<float>::infinity())
                .WriteKey("l")
                .BeginList()
                .WriteFloat(std::numeric_limits<float>::quiet_NaN())
                .EndList()
                .EndObject();

            UNIT_ASSERT_STRINGS_EQUAL(buf.Str(), R"raw_json({"nanvalue":"nan","infvalue":"inf","minus_infvalue":"-inf","l":["nan"]})raw_json");
        }

        {
            NJsonWriter::TBuf buf;
            buf.BeginObject()
                .WriteKey("<>&")
                .WriteString("Ololo")
                .UnsafeWriteKey("<>&")
                .WriteString("Ololo2")
                .EndObject();

            UNIT_ASSERT_STRINGS_EQUAL(buf.Str(), R"({"\u003C\u003E&":"Ololo","<>&":"Ololo2"})");
        }
    }
}
