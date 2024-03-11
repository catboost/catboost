#include "column.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <util/string/cast.h>


TStringBuf ToCanonicalColumnName(TStringBuf columnName) {
    if (columnName == "QueryId") {
        return "GroupId";
    }
    if (columnName == "Target") {
        return "Label";
    }
    if (columnName == "DocId") {
        return "SampleId";
    }
    return columnName;
}

void ParseOutputColumnByIndex(const TString& outputColumn, ui32* columnNumber, TString* name) {
    size_t delimiterPos = outputColumn.find(':');
    TString index = outputColumn.substr(1, delimiterPos - 1);
    *name = (delimiterPos == TString::npos) ? "#" + index : outputColumn.substr(delimiterPos + 1);
    if (!TryFromString(index, *columnNumber)) {
        CB_ENSURE(false, "Wrong index format " << index << " in output column " << outputColumn);
    }
}

TColumn::operator NJson::TJsonValue() const {
    NJson::TJsonValue result(NJson::JSON_MAP);
    result.InsertValue("Type"sv, ToString(Type));
    result.InsertValue("Id"sv, Id);
    if (!SubColumns.empty()) {
        result.InsertValue("SubColumns"sv, VectorToJson(SubColumns));
    }
    return result;
}
