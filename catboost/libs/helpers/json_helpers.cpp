#include "json_helpers.h"

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/json_writer.h>

#include <util/stream/str.h>


NJson::TJsonValue ReadTJsonValue(const TStringBuf paramsJson) {
    NJson::TJsonValue tree;
    NJson::ReadJsonTree(paramsJson, &tree);
    return tree;
}

TString WriteTJsonValue(const NJson::TJsonValue& jsonValue) {
    TStringStream out;
    {
        NJson::TJsonWriterConfig jsonWriterConfig;
        jsonWriterConfig.FloatToStringMode = EFloatToStringMode::PREC_AUTO;

        NJson::TJsonWriter jsonWriter(&out, jsonWriterConfig);
        jsonWriter.Write(jsonValue);
    }
    return out.Str();
}

void FromJson(const NJson::TJsonValue& value, TString* result) {
    *result = value.GetString();
}

