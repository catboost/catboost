#include "cd_parser.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/system/fs.h>
#include <util/stream/file.h>
#include <util/string/split.h>
#include <util/generic/set.h>

inline void CheckAllFeaturesPresent(const TVector<TColumn>& columns, const TSet<int>& parsedColumns) {
    for (int i = 0; i < columns.ysize(); ++i) {
        CB_ENSURE(parsedColumns.has(i), "column not present in cd file: " << i);
    }
}

TVector<TColumn> ReadCD(const TString& fileName, const TCdParserDefaults& defaults) {
    CB_ENSURE(NFs::Exists(TString(fileName)), "column description file is not found");
    int columnsCount = defaults.UseDefaultType ? defaults.ColumnCount : 0;

    TVector<TColumn> columns(columnsCount, TColumn{defaults.DefaultColumnType, TString()});
    TSet<int> parsedColumns;

    TString line;
    TIFStream reader(fileName.c_str());
    while (reader.ReadLine(line)) {
        TVector<TString> tokens;
        try {
            Split(line, "\t", tokens);
        } catch (const yexception& e) {
            MATRIXNET_DEBUG_LOG << "Got exception " << e.what() << " while parsing feature descriptions line " << line << Endl;
            break;
        }
        if (tokens.empty()) {
            continue;
        }
        CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3, "Each line should have two or three columns. " << line);
        int index = FromString<int>(tokens[0]);
        CB_ENSURE(index >= 0, "Invalid column index " << index);
        if (defaults.UseDefaultType) {
            CB_ENSURE(index < columnsCount, "Invalid column index " << index);
        }
        CB_ENSURE(!parsedColumns.has(index), "column specified twice in cd file: " << index);
        parsedColumns.insert(index);
        columns.resize(Max(columns.ysize(), index + 1));

        TStringBuf type = tokens[1];
        if (type == "QueryId") {
            type = "GroupId";
        }
        if (type == "Target") {
            type = "Label";
        }
        CB_ENSURE(TryFromString<EColumn>(type, columns[index].Type), "unsupported column type " << type);
        if (tokens.ysize() == 3) {
            columns[index].Id = tokens[2];
        }
    }
    if (!defaults.UseDefaultType) {
        CheckAllFeaturesPresent(columns, parsedColumns);
    }

    return columns;
}
