#include "json2yson.h"

#include <library/cpp/yson/parser.h>
#include <library/cpp/yson/json/json_writer.h>
#include <library/cpp/yson/json/yson2json_adapter.h>

namespace NJson2Yson {
    static void WriteJsonValue(const NJson::TJsonValue& jsonValue, NYT::TYson2JsonCallbacksAdapter* adapter) {
        switch (jsonValue.GetType()) {
            default:
            case NJson::JSON_NULL:
                adapter->OnNull();
                break;
            case NJson::JSON_BOOLEAN:
                adapter->OnBoolean(jsonValue.GetBoolean());
                break;
            case NJson::JSON_DOUBLE:
                adapter->OnDouble(jsonValue.GetDouble());
                break;
            case NJson::JSON_INTEGER:
                adapter->OnInteger(jsonValue.GetInteger());
                break;
            case NJson::JSON_UINTEGER:
                adapter->OnUInteger(jsonValue.GetUInteger());
                break;
            case NJson::JSON_STRING:
                adapter->OnString(jsonValue.GetString());
                break;
            case NJson::JSON_ARRAY: {
                adapter->OnOpenArray();
                const NJson::TJsonValue::TArray& arr = jsonValue.GetArray();
                for (const auto& it : arr)
                    WriteJsonValue(it, adapter);
                adapter->OnCloseArray();
                break;
            }
            case NJson::JSON_MAP: {
                adapter->OnOpenMap();
                const NJson::TJsonValue::TMapType& map = jsonValue.GetMap();
                for (const auto& it : map) {
                    adapter->OnMapKey(it.first);
                    WriteJsonValue(it.second, adapter);
                }
                adapter->OnCloseMap();
                break;
            }
        }
    }

    void SerializeJsonValueAsYson(const NJson::TJsonValue& inputValue, NYson::TYsonWriter* ysonWriter) {
        NYT::TYson2JsonCallbacksAdapter adapter(ysonWriter);
        WriteJsonValue(inputValue, &adapter);
    }

    void SerializeJsonValueAsYson(const NJson::TJsonValue& inputValue, IOutputStream* outputStream) {
        NYson::TYsonWriter ysonWriter(outputStream, NYson::EYsonFormat::Binary, ::NYson::EYsonType::Node, false);
        SerializeJsonValueAsYson(inputValue, &ysonWriter);
    }

    void SerializeJsonValueAsYson(const NJson::TJsonValue& inputValue, TString& result) {
        TStringOutput resultStream(result);
        SerializeJsonValueAsYson(inputValue, &resultStream);
    }

    TString SerializeJsonValueAsYson(const NJson::TJsonValue& inputValue) {
        TString result;
        SerializeJsonValueAsYson(inputValue, result);
        return result;
    }

    bool DeserializeYsonAsJsonValue(IInputStream* inputStream, NJson::TJsonValue* outputValue, bool throwOnError) {
        NJson::TParserCallbacks parser(*outputValue);
        NJson2Yson::TJsonBuilder consumer(&parser);
        NYson::TYsonParser ysonParser(&consumer, inputStream, ::NYson::EYsonType::Node);
        try {
            ysonParser.Parse();
        } catch (...) {
            if (throwOnError) {
                throw;
            }
            return false;
        }
        return true;
    }

    bool DeserializeYsonAsJsonValue(TStringBuf str, NJson::TJsonValue* outputValue, bool throwOnError) {
        TMemoryInput inputStream(str);
        return DeserializeYsonAsJsonValue(&inputStream, outputValue, throwOnError);
    }

    void ConvertYson2Json(IInputStream* inputStream, IOutputStream* outputStream) {
        NYT::TJsonWriter writer(outputStream, ::NYson::EYsonType::Node, NYT::JF_TEXT, NYT::JAM_ON_DEMAND, NYT::SBF_BOOLEAN);
        NYson::TYsonParser ysonParser(&writer, inputStream, ::NYson::EYsonType::Node);
        ysonParser.Parse();
    }

    void ConvertYson2Json(TStringBuf yson, IOutputStream* outputStream) {
        TMemoryInput inputStream(yson);
        ConvertYson2Json(&inputStream, outputStream);
    }

    TString ConvertYson2Json(TStringBuf yson) {
        TString json;
        TStringOutput outputStream(json);
        ConvertYson2Json(yson, &outputStream);
        return json;
    }
}
