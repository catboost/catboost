#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/logging/logging.h>

#include <util/system/fs.h>
#include <util/stream/file.h>
#include <util/string/iterator.h>

template <class TStr>
TVector<TPair> ReadPairs(const TStr& fileName, int docCount) {
    CB_ENSURE(NFs::Exists(TString(fileName)), "pairs file is not found");
    TIFStream reader(fileName.c_str());

    TVector<TPair> pairs;
    TString line;
    while (reader.ReadLine(line)) {
        TVector<TString> tokens;
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
        CB_ENSURE(tokens.ysize() == 2 || tokens.ysize() == 3, "Each line should have two or three columns. Invalid line number " << line);
        int winnerId = FromString<int>(tokens[0]);
        int loserId = FromString<int>(tokens[1]);
        float weight = 1;
        if (tokens.ysize() == 3) {
            weight = FromString<float>(tokens[2]);
        }
        CB_ENSURE(winnerId >= 0 && winnerId < docCount, "Invalid winner index " << winnerId);
        CB_ENSURE(loserId >= 0 && loserId < docCount, "Invalid loser index " << loserId);
        pairs.emplace_back(winnerId, loserId, weight);
    }

    return pairs;
}

template <class TStr>
inline int ReadColumnsCount(const TStr& poolFile, char fieldDelimiter = '\t') {
    CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file is not found");
    TIFStream reader(poolFile.c_str());
    TString line;
    CB_ENSURE(reader.ReadLine(line), "pool can't be empty");
    return StringSplitter(line).Split(fieldDelimiter).Count();
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

