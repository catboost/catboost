#include "column.h"


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
