#include "column.h"

#include <catboost/libs/helpers/exception.h>

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
