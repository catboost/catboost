#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/logging/logging.h>

#include <util/system/fs.h>
#include <util/stream/file.h>
#include <util/string/iterator.h>



template <class TStr>
inline int ReadColumnsCount(const TStr& poolFile, char fieldDelimiter = '\t') {
    CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file is not found");
    TIFStream reader(poolFile.c_str());
    TString line;
    CB_ENSURE(reader.ReadLine(line), "pool can't be empty");
    return StringSplitter(line).Split(fieldDelimiter).Count();
}



