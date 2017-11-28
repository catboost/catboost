#include "json_helper.h"
#include <library/json/json_reader.h>

NJson::TJsonValue ReadTJsonValue(const TString& paramsJson) {
    TStringInput is(paramsJson);
    NJson::TJsonValue tree;
    NJson::ReadJsonTree(&is, &tree);
    return tree;
}
