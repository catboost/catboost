#include "json2yson.h"

#include <library/yson/json_writer.h>
#include <library/yson/parser.h>
#include <library/yson/yson2json_adapter.h>

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

    void SerializeJsonValueAsYson(const NJson::TJsonValue& inputValue, NYT::TYsonWriter* ysonWriter) {
        NYT::TYson2JsonCallbacksAdapter adapter(ysonWriter);
        WriteJsonValue(inputValue, &adapter);
    }

    void SerializeJsonValueAsYson(const NJson::TJsonValue& inputValue, IOutputStream* outputStream) {
        NYT::TYsonWriter ysonWriter(outputStream, NYT::YF_BINARY, NYT::YT_NODE, false);
        SerializeJsonValueAsYson(inputValue, &ysonWriter);
    }

    void DeserializeYsonAsJsonValue(IInputStream* inputStream, NJson::TJsonValue* outputValue) {
        NJson::TParserCallbacks parser(*outputValue, true);
        NJson2Yson::TJsonBuilder consumer(&parser);
        NYT::TYsonParser ysonParser(&consumer, inputStream, NYT::YT_NODE);
        ysonParser.Parse();
    }

    void ConvertYson2Json(IInputStream* inputStream, IOutputStream* outputStream) {
        NYT::TJsonWriter writer(outputStream);
        NYT::TYsonParser ysonParser(&writer, inputStream, NYT::YT_NODE);
        ysonParser.Parse();
    }

    void DeserializeYsonAsJsonValue(TStringBuf str, NJson::TJsonValue* outputValue) {
        TMemoryInput inputStream(str);
        DeserializeYsonAsJsonValue(&inputStream, outputValue);
    }
}
