#pragma once

#include "pair.h"

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <util/system/fs.h>
#include <util/stream/file.h>
#include <util/string/split.h>

template <class TStr>
yvector<TPair> ReadPairs(const TStr& fileName, int docCount) {
    CB_ENSURE(NFs::Exists(TString(fileName)), "pairs file is not found");
    TIFStream reader(fileName.c_str());

    yvector<TPair> pairs;
    TString line;
    while (reader.ReadLine(line)) {
        yvector<TString> tokens;
        try {
            Split(line, "\t", tokens);
        }
        catch (const yexception& e) {
            MATRIXNET_DEBUG_LOG << "Got exception " << e.what() << " while parsing pairs line " << line << Endl;
            break;
        }
        if (tokens.empty()) {
            continue;
        }
        CB_ENSURE(tokens.ysize() == 2, "Each line should have two columns. Invalid line number " << line);
        int winnerId = FromString<int>(tokens[0]);
        int loserId = FromString<int>(tokens[1]);
        CB_ENSURE(winnerId >= 0 && winnerId < docCount, "Invalid winner index " << winnerId);
        CB_ENSURE(loserId >= 0 && loserId < docCount, "Invalid loser index " << loserId);
        pairs.push_back(TPair(winnerId, loserId));
    }

    return pairs;
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

template <class TStr>
inline int CountLines(const TStr& poolFile) {
    CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file '" << TString(poolFile) << "' is not found");
    TIFStream reader(poolFile.c_str());
    size_t count = 0;
    TString buffer;
    while (reader.ReadLine(buffer)) {
        ++count;
    }
    return count;
}

