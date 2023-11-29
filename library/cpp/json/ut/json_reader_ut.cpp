#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>

#include <library/cpp/testing/unittest/registar.h>
#include <util/stream/str.h>

using namespace NJson;

class TReformatCallbacks: public TJsonCallbacks {
    TJsonWriter& Writer;

public:
    TReformatCallbacks(TJsonWriter& writer)
        : Writer(writer)
    {
    }

    bool OnBoolean(bool val) override {
        Writer.Write(val);
        return true;
    }

    bool OnInteger(long long val) override {
        Writer.Write(val);
        return true;
    }

    bool OnUInteger(unsigned long long val) override {
        Writer.Write(val);
        return true;
    }

    bool OnString(const TStringBuf& val) override {
        Writer.Write(val);
        return true;
    }

    bool OnDouble(double val) override {
        Writer.Write(val);
        return true;
    }

    bool OnOpenArray() override {
        Writer.OpenArray();
        return true;
    }

    bool OnCloseArray() override {
        Writer.CloseArray();
        return true;
    }

    bool OnOpenMap() override {
        Writer.OpenArray();
        return true;
    }

    bool OnCloseMap() override {
        Writer.CloseArray();
        return true;
    }

    bool OnMapKey(const TStringBuf& val) override {
        Writer.Write(val);
        return true;
    }
};

void GenerateDeepJson(TStringStream& stream, ui64 depth) {
    stream << "{\"key\":";
    for (ui32 i = 0; i < depth - 1; ++i) {
        stream << "[";
    }
    for (ui32 i = 0; i < depth - 1; ++i) {
        stream << "]";
    }
    stream << "}";
}

Y_UNIT_TEST_SUITE(TJsonReaderTest) {
    Y_UNIT_TEST(JsonReformatTest) {
        TString data = "{\"null value\": null, \"intkey\": 10, \"double key\": 11.11, \"string key\": \"string\", \"array\": [1,2,3,\"TString\"], \"bool key\": true}";

        TString result1, result2;
        {
            TStringStream in;
            in << data;
            TStringStream out;
            TJsonWriter writer(&out, false);
            TReformatCallbacks cb(writer);
            ReadJson(&in, &cb);
            writer.Flush();
            result1 = out.Str();
        }

        {
            TStringStream in;
            in << result1;
            TStringStream out;
            TJsonWriter writer(&out, false);
            TReformatCallbacks cb(writer);
            ReadJson(&in, &cb);
            writer.Flush();
            result2 = out.Str();
        }

        UNIT_ASSERT_VALUES_EQUAL(result1, result2);
    }

    Y_UNIT_TEST(TJsonEscapedApostrophe) {
        TString jsonString = "{ \"foo\" : \"bar\\'buzz\" }";
        {
            TStringStream in;
            in << jsonString;
            TStringStream out;
            TJsonWriter writer(&out, false);
            TReformatCallbacks cb(writer);
            UNIT_ASSERT(!ReadJson(&in, &cb));
        }

        {
            TStringStream in;
            in << jsonString;
            TStringStream out;
            TJsonWriter writer(&out, false);
            TReformatCallbacks cb(writer);
            UNIT_ASSERT(ReadJson(&in, false, true, &cb));
            writer.Flush();
            UNIT_ASSERT_EQUAL(out.Str(), "[\"foo\",\"bar'buzz\"]");
        }
    }

    Y_UNIT_TEST(TJsonTreeTest) {
        TString data = "{\"intkey\": 10, \"double key\": 11.11, \"null value\":null, \"string key\": \"string\", \"array\": [1,2,3,\"TString\"], \"bool key\": true}";
        TStringStream in;
        in << data;
        TJsonValue value;
        ReadJsonTree(&in, &value);

        UNIT_ASSERT_VALUES_EQUAL(value["intkey"].GetInteger(), 10);
        UNIT_ASSERT_DOUBLES_EQUAL(value["double key"].GetDouble(), 11.11, 0.001);
        UNIT_ASSERT_VALUES_EQUAL(value["bool key"].GetBoolean(), true);
        UNIT_ASSERT_VALUES_EQUAL(value["absent string key"].GetString(), TString(""));
        UNIT_ASSERT_VALUES_EQUAL(value["string key"].GetString(), TString("string"));
        UNIT_ASSERT_VALUES_EQUAL(value["array"][0].GetInteger(), 1);
        UNIT_ASSERT_VALUES_EQUAL(value["array"][1].GetInteger(), 2);
        UNIT_ASSERT_VALUES_EQUAL(value["array"][2].GetInteger(), 3);
        UNIT_ASSERT_VALUES_EQUAL(value["array"][3].GetString(), TString("TString"));
        UNIT_ASSERT(value["null value"].IsNull());

        // AsString
        UNIT_ASSERT_VALUES_EQUAL(value["intkey"].GetStringRobust(), "10");
        UNIT_ASSERT_VALUES_EQUAL(value["double key"].GetStringRobust(), "11.11");
        UNIT_ASSERT_VALUES_EQUAL(value["bool key"].GetStringRobust(), "true");
        UNIT_ASSERT_VALUES_EQUAL(value["string key"].GetStringRobust(), "string");
        UNIT_ASSERT_VALUES_EQUAL(value["array"].GetStringRobust(), "[1,2,3,\"TString\"]");
        UNIT_ASSERT_VALUES_EQUAL(value["null value"].GetStringRobust(), "null");

        const TJsonValue::TArray* array;
        UNIT_ASSERT(GetArrayPointer(value, "array", &array));
        UNIT_ASSERT_VALUES_EQUAL(value["array"].GetArray().size(), array->size());
        UNIT_ASSERT_VALUES_EQUAL(value["array"][0].GetInteger(), (*array)[0].GetInteger());
        UNIT_ASSERT_VALUES_EQUAL(value["array"][1].GetInteger(), (*array)[1].GetInteger());
        UNIT_ASSERT_VALUES_EQUAL(value["array"][2].GetInteger(), (*array)[2].GetInteger());
        UNIT_ASSERT_VALUES_EQUAL(value["array"][3].GetString(), (*array)[3].GetString());
    }

    Y_UNIT_TEST(TJsonRomaTest) {
        TString data = "{\"test\": [ {\"name\": \"A\"} ]}";

        TStringStream in;
        in << data;
        TJsonValue value;
        ReadJsonTree(&in, &value);

        UNIT_ASSERT_VALUES_EQUAL(value["test"][0]["name"].GetString(), TString("A"));
    }

    Y_UNIT_TEST(TJsonReadTreeWithComments) {
        {
            TString leadingCommentData = "{ // \"test\" : 1 \n}";
            {
                // No comments allowed
                TStringStream in;
                in << leadingCommentData;
                TJsonValue value;
                UNIT_ASSERT(!ReadJsonTree(&in, false, &value));
            }

            {
                // Comments allowed
                TStringStream in;
                in << leadingCommentData;
                TJsonValue value;
                UNIT_ASSERT(ReadJsonTree(&in, true, &value));
                UNIT_ASSERT(!value.Has("test"));
            }
        }

        {
            TString trailingCommentData = "{ \"test1\" : 1 // \"test2\" : 2 \n }";
            {
                // No comments allowed
                TStringStream in;
                in << trailingCommentData;
                TJsonValue value;
                UNIT_ASSERT(!ReadJsonTree(&in, false, &value));
            }

            {
                // Comments allowed
                TStringStream in;
                in << trailingCommentData;
                TJsonValue value;
                UNIT_ASSERT(ReadJsonTree(&in, true, &value));
                UNIT_ASSERT(value.Has("test1"));
                UNIT_ASSERT_EQUAL(value["test1"].GetInteger(), 1);
                UNIT_ASSERT(!value.Has("test2"));
            }
        }
    }

    Y_UNIT_TEST(TJsonSignedIntegerTest) {
        {
            TStringStream in;
            in << "{ \"test\" : " << Min<i64>() << " }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsInteger());
            UNIT_ASSERT(!value["test"].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"].GetInteger(), Min<i64>());
            UNIT_ASSERT_EQUAL(value["test"].GetIntegerRobust(), Min<i64>());
        } // Min<i64>()

        {
            TStringStream in;
            in << "{ \"test\" : " << Max<i64>() + 1ull << " }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(!value["test"].IsInteger());
            UNIT_ASSERT(value["test"].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"].GetIntegerRobust(), (i64)(Max<i64>() + 1ull));
        } // Max<i64>() + 1
    }

    Y_UNIT_TEST(TJsonUnsignedIntegerTest) {
        {
            TStringStream in;
            in << "{ \"test\" : 1 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsInteger());
            UNIT_ASSERT(value["test"].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"].GetInteger(), 1);
            UNIT_ASSERT_EQUAL(value["test"].GetIntegerRobust(), 1);
            UNIT_ASSERT_EQUAL(value["test"].GetUInteger(), 1);
            UNIT_ASSERT_EQUAL(value["test"].GetUIntegerRobust(), 1);
        } // 1

        {
            TStringStream in;
            in << "{ \"test\" : -1 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsInteger());
            UNIT_ASSERT(!value["test"].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"].GetInteger(), -1);
            UNIT_ASSERT_EQUAL(value["test"].GetIntegerRobust(), -1);
            UNIT_ASSERT_EQUAL(value["test"].GetUInteger(), 0);
            UNIT_ASSERT_EQUAL(value["test"].GetUIntegerRobust(), static_cast<unsigned long long>(-1));
        } // -1

        {
            TStringStream in;
            in << "{ \"test\" : 18446744073709551615 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(!value["test"].IsInteger());
            UNIT_ASSERT(value["test"].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"].GetInteger(), 0);
            UNIT_ASSERT_EQUAL(value["test"].GetIntegerRobust(), static_cast<long long>(18446744073709551615ull));
            UNIT_ASSERT_EQUAL(value["test"].GetUInteger(), 18446744073709551615ull);
            UNIT_ASSERT_EQUAL(value["test"].GetUIntegerRobust(), 18446744073709551615ull);
        } // 18446744073709551615

        {
            TStringStream in;
            in << "{ \"test\" : 1.1 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(!value["test"].IsInteger());
            UNIT_ASSERT(!value["test"].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"].GetInteger(), 0);
            UNIT_ASSERT_EQUAL(value["test"].GetIntegerRobust(), static_cast<long long>(1.1));
            UNIT_ASSERT_EQUAL(value["test"].GetUInteger(), 0);
            UNIT_ASSERT_EQUAL(value["test"].GetUIntegerRobust(), static_cast<unsigned long long>(1.1));
        } // 1.1

        {
            TStringStream in;
            in << "{ \"test\" : [1, 18446744073709551615] }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsArray());
            UNIT_ASSERT_EQUAL(value["test"].GetArray().size(), 2);
            UNIT_ASSERT(value["test"][0].IsInteger());
            UNIT_ASSERT(value["test"][0].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"][0].GetInteger(), 1);
            UNIT_ASSERT_EQUAL(value["test"][0].GetUInteger(), 1);
            UNIT_ASSERT(!value["test"][1].IsInteger());
            UNIT_ASSERT(value["test"][1].IsUInteger());
            UNIT_ASSERT_EQUAL(value["test"][1].GetUInteger(), 18446744073709551615ull);
        }
    } // TJsonUnsignedIntegerTest

    Y_UNIT_TEST(TJsonDoubleTest) {
        {
            TStringStream in;
            in << "{ \"test\" : 1.0 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsDouble());
            UNIT_ASSERT_EQUAL(value["test"].GetDouble(), 1.0);
            UNIT_ASSERT_EQUAL(value["test"].GetDoubleRobust(), 1.0);
        } // 1.0

        {
            TStringStream in;
            in << "{ \"test\" : 1 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsDouble());
            UNIT_ASSERT_EQUAL(value["test"].GetDouble(), 1.0);
            UNIT_ASSERT_EQUAL(value["test"].GetDoubleRobust(), 1.0);
        } // 1

        {
            TStringStream in;
            in << "{ \"test\" : -1 }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(value["test"].IsDouble());
            UNIT_ASSERT_EQUAL(value["test"].GetDouble(), -1.0);
            UNIT_ASSERT_EQUAL(value["test"].GetDoubleRobust(), -1.0);
        } // -1

        {
            TStringStream in;
            in << "{ \"test\" : " << Max<ui64>() << " }";
            TJsonValue value;
            UNIT_ASSERT(ReadJsonTree(&in, &value));
            UNIT_ASSERT(value.Has("test"));
            UNIT_ASSERT(!value["test"].IsDouble());
            UNIT_ASSERT_EQUAL(value["test"].GetDouble(), 0.0);
            UNIT_ASSERT_EQUAL(value["test"].GetDoubleRobust(), static_cast<double>(Max<ui64>()));
        } // Max<ui64>()
    }     // TJsonDoubleTest

    Y_UNIT_TEST(TJsonInvalidTest) {
        {
            // No exceptions mode.
            TStringStream in;
            in << "{ \"test\" : }";
            TJsonValue value;
            UNIT_ASSERT(!ReadJsonTree(&in, &value));
        }

        {
            // Exception throwing mode.
            TStringStream in;
            in << "{ \"test\" : }";
            TJsonValue value;
            UNIT_ASSERT_EXCEPTION(ReadJsonTree(&in, &value, true), TJsonException);
        }
    }

    Y_UNIT_TEST(TJsonMemoryLeakTest) {
        // after https://clubs.at.yandex-team.ru/stackoverflow/3691
        TString s = ".";
        NJson::TJsonValue json;
        try {
            TStringInput in(s);
            NJson::ReadJsonTree(&in, &json, true);
        } catch (...) {
        }
    } // TJsonMemoryLeakTest

    Y_UNIT_TEST(TJsonDuplicateKeysWithNullValuesTest) {
        const TString json = "{\"\":null,\"\":\"\"}";

        TStringInput in(json);
        NJson::TJsonValue v;
        UNIT_ASSERT(ReadJsonTree(&in, &v));
        UNIT_ASSERT(v.IsMap());
        UNIT_ASSERT_VALUES_EQUAL(1, v.GetMap().size());
        UNIT_ASSERT_VALUES_EQUAL("", v.GetMap().begin()->first);
        UNIT_ASSERT(v.GetMap().begin()->second.IsString());
        UNIT_ASSERT_VALUES_EQUAL("", v.GetMap().begin()->second.GetString());
    }

    // Parsing an extremely deep json tree would result in stack overflow.
    // Not crashing on one is a good indicator of iterative mode.
    Y_UNIT_TEST(TJsonIterativeTest) {
        constexpr ui32 brackets = static_cast<ui32>(1e5);

        TStringStream jsonStream;
        GenerateDeepJson(jsonStream, brackets);

        TJsonReaderConfig config;
        config.UseIterativeParser = true;
        config.MaxDepth = static_cast<ui32>(1e3);

        TJsonValue v;
        UNIT_ASSERT(!ReadJsonTree(&jsonStream, &config, &v));
    }

    Y_UNIT_TEST(TJsonMaxDepthTest) {
        constexpr ui32 depth = static_cast<ui32>(1e3);

        {
            TStringStream jsonStream;
            GenerateDeepJson(jsonStream, depth);
            TJsonReaderConfig config;
            config.MaxDepth = depth;
            TJsonValue v;
            UNIT_ASSERT(ReadJsonTree(&jsonStream, &config, &v));
        }

        {
            TStringStream jsonStream;
            GenerateDeepJson(jsonStream, depth);
            TJsonReaderConfig config;
            config.MaxDepth = depth - 1;
            TJsonValue v;
            UNIT_ASSERT(!ReadJsonTree(&jsonStream, &config, &v));
        }
    }
}


static const TString YANDEX_STREAMING_JSON("{\"a\":1}//d{\"b\":2}");


Y_UNIT_TEST_SUITE(TCompareReadJsonFast) {
    Y_UNIT_TEST(NoEndl) {
        NJson::TJsonValue parsed;

        bool success = NJson::ReadJsonTree(YANDEX_STREAMING_JSON, &parsed, false);
        bool fast_success = NJson::ReadJsonFastTree(YANDEX_STREAMING_JSON, &parsed, false);
        UNIT_ASSERT(success == fast_success);
    }
    Y_UNIT_TEST(WithEndl) {
        NJson::TJsonValue parsed1;
        NJson::TJsonValue parsed2;

        bool success = NJson::ReadJsonTree(YANDEX_STREAMING_JSON + "\n", &parsed1, false);
        bool fast_success = NJson::ReadJsonFastTree(YANDEX_STREAMING_JSON + "\n", &parsed2, false);

        UNIT_ASSERT_VALUES_EQUAL(success, fast_success);
    }
    Y_UNIT_TEST(NoQuotes) {
        TString streamingJson = "{a:1}";
        NJson::TJsonValue parsed;

        bool success = NJson::ReadJsonTree(streamingJson, &parsed, false);
        bool fast_success = NJson::ReadJsonFastTree(streamingJson, &parsed, false);
        UNIT_ASSERT(success != fast_success);
    }
}
