#include "json_helper.h"

#include <library/json/json_reader.h>

NJson::TJsonValue ReadTJsonValue(const TStringBuf paramsJson) {
    NJson::TJsonValue tree;
    NJson::ReadJsonTree(paramsJson, &tree);
    return tree;
}
