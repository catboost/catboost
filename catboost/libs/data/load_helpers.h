#pragma once

#include "column.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/system/fs.h>
#include <util/stream/file.h>
#include <util/stream/file.h>
#include <util/string/split.h>


template <class TStr>
yvector<TColumn> ReadCD(const TStr& fileName, const int columnsCount) {
    CB_ENSURE(NFs::Exists(TString(fileName)), "column description file is not found");
    TIFStream reader(fileName.c_str());
    yvector<TColumn> columns(columnsCount, TColumn{EColumn::Num,TString()});
    int targetsCount = 0;
    TString line;
    while (reader.ReadLine(line)) {
        yvector<TString> tokens;
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
        CB_ENSURE(index >= 0 && index < columnsCount, "Invalid column index " << index);
        TStringBuf type = tokens[1];
        CB_ENSURE(TryFromString<EColumn>(type, columns[index].Type), "unsupported column type");
        if (columns[index].Type == EColumn::Target) {
            ++targetsCount;
        }
        if (tokens.ysize() == 3) {
            columns[index].Id = tokens[2];
        }
    }
    CB_ENSURE(targetsCount <= 1, "wrong number of target columns");
    return columns;
}


template <class TStr>
inline int ReadColumnsCount(const TStr& poolFile, char fieldDelimiter = '\t') {
    CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file is not found");
    TIFStream reader(poolFile.c_str());
    TString line;
    CB_ENSURE(reader.ReadLine(line), "pool can't be empty");
    yvector<TStringBuf> words;
    SplitRangeTo<const char, yvector<TStringBuf>>(~line, ~line + line.size(), fieldDelimiter, &words);
    return words.ysize();
}

