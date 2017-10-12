#include "test_utils.h"
#include <catboost/cuda/cuda_util/cpu_random.h>

using namespace std;

void GenerateTestPool(TBinarizedPool& pool,
                      const uint binarization,
                      ui32 catFeatures, ui32 seed) {
    TRandom rand(seed);
    const ui32 numSamples = 25527;
    const ui32 samplesPerQuery = 17;
    const ui32 numFeatures = 157;
    pool.CompressedIndex.clear();
    pool.Targets.clear();
    pool.Queries.clear();
    pool.Qids.clear();
    ui32 qid = 100000;
    pool.CatFeatures.resize(catFeatures);
    pool.NumCatFeatures = catFeatures;

    for (ui32 i = 0; i < numSamples; ++i) {
        if (i % samplesPerQuery == 0) {
            qid += 1;
            pool.Queries.resize(i / samplesPerQuery + 1);
        }
        pool.Qids.push_back(qid);
        for (ui32 j = 0; j < catFeatures; ++j) {
            pool.CatFeatures[j].push_back(rand.NextUniformL() % (j % 2 == 0 ? 5 : binarization));
        }
        pool.Targets.push_back((1.0 * (rand.NextUniformL() % 5)) / 4);
        pool.Queries[i / samplesPerQuery].push_back(i);
    }
    pool.NumSamples = numSamples;
    pool.SamplesPerQuery = samplesPerQuery;
    pool.NumFeatures = numFeatures;
    pool.Features.resize(numFeatures);
    for (ui32 f = 0; f < numFeatures; ++f) {
        uint binsCount = ((f % 10 == 1) ? 2 : (ui32)(2 + rand.NextUniformL() % (binarization - 1)));
        if (f % 20 == 0 && (binsCount > 15)) {
            binsCount = 15; //for halfByte tests
        }
        pool.Features[f].resize(numSamples);
        for (ui32 doc = 0; doc < numSamples; ++doc) {
            ui8 bin = (ui8)(rand.NextUniformL() % binsCount);
            pool.Features[f][doc] = bin;
        }
    }
}

void SavePoolToFile(TBinarizedPool& pool, const char* filename) {
    TOFStream output(filename);
    for (ui32 doc = 0; doc < pool.NumSamples; ++doc) {
        output << pool.Qids[doc] << "\t" << pool.Targets[doc] << "\tFakeUrl";
        for (ui32 f = 0; f < pool.NumCatFeatures; ++f) {
            output << "\t" << pool.CatFeatures[f][doc];
        }
        for (ui32 f = 0; f < pool.NumFeatures; ++f) {
            output << "\t" << (float)(pool.Features[f][doc]);
        }
        output << Endl;
    }
}

void GenerateTestPool(TUnitTestPool& pool, ui32 numFeatures) {
    TRandom rand(0);
    const ui32 numSamples = 19371;
    const ui32 samplesPerQuery = 16;

    pool.Features.clear();
    pool.Targets.clear();
    pool.Queries.clear();
    pool.Qids.clear();
    ui32 qid = 1000000;
    for (ui32 i = 0; i < numSamples; ++i) {
        if (i % samplesPerQuery == 0) {
            qid += 10;
            pool.Queries.resize(i / samplesPerQuery + 1);
        }
        pool.Qids.push_back(qid);
        pool.Gids.push_back((rand.NextUniformL() % 20));
        pool.Targets.push_back((1.0 * (rand.NextUniformL() % 5)) / 4);
        pool.Queries[i / samplesPerQuery].push_back(i);
    }
    pool.NumSamples = numSamples;
    pool.SamplesPerQuery = samplesPerQuery;
    pool.NumFeatures = (ui64)numFeatures;
    pool.Features.reserve(numFeatures * numSamples);
    for (ui32 f = 0; f < numFeatures; ++f) {
        bool isBinary = f % 10 == 0;
        for (ui32 doc = 0; doc < numSamples; ++doc) {
            if (!isBinary) {
                pool.Features.push_back((rand.NextUniformL() % 1001) * 1.0 / 1000);
            } else {
                pool.Features.push_back(rand.NextUniformL() % 2);
            }
        }
    }
}

void SavePoolToFile(TUnitTestPool& pool, const char* filename) {
    TOFStream output(filename);
    for (ui32 i = 0; i < pool.NumSamples; ++i) {
        output << pool.Qids[i] << "\t" << pool.Targets[i] << "\tFakeUrl\t" << pool.Gids[i];
        for (ui32 j = 0; j < pool.NumFeatures; ++j) {
            output << "\t" << pool.Features[pool.NumSamples * j + i];
        }
        output << Endl;
    }
}

void SavePoolCDToFile(const char* filename, ui32 catFeatures) {
    TOFStream output(filename);
    output << "0\tQueryId" << Endl;
    output << "1\tTarget" << Endl;
    output << "2\tAuxiliary" << Endl;
    for (ui32 i = 0; i < catFeatures; ++i) {
        output << (3 + i) << "\tCateg" << Endl;
    }
}
