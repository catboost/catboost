#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/ut/lib/for_loader.h>

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/objects_grouping.h>

#include <library/cpp/testing/benchmark/bench.h>
#include <library/cpp/testing/unittest/tests_data.h>

using namespace NCB;
using namespace NDataNewUT;

const size_t PrimersCount = 100;
const size_t FeaturesCount = 100;

TString GetPool() {
    TString pool = "";
    for (size_t primer = 0; primer < PrimersCount; ++primer) {
        pool += ToString(primer);
        for (size_t feature = 0; feature < FeaturesCount; ++feature) {
            pool += "\t" + ToString(feature);
        }
        pool += '\n';
    }
    return pool;
}

TString GetQuotedPool() {
    TString pool = "";
    for (size_t primer = 0; primer < PrimersCount; ++primer) {
        pool += ToString(primer);
        for (size_t feature = 0; feature < FeaturesCount; ++feature) {
            pool += "\t\"\"\"" + ToString(feature) + "\"\"\"";
        }
        pool += '\n';
    }
    return pool;
}

Y_CPU_BENCHMARK(DsvLoaderNumFeatures, iface) {
    TReadDatasetMainParams readDatasetMainParams;
    NPar::TLocalExecutor localExecutor;
    TSrcData srcData;

    TString Cd = "0\tTarget";
    for (size_t feature = 0; feature < FeaturesCount; ++feature) {
        Cd += "\n" + ToString(feature + 1) + "\tNum";
    }
    TString DatasetFileData = GetPool();

    srcData.CdFileData = Cd;
    srcData.DatasetFileData = DatasetFileData;

    TVector<THolder<TTempFile>> srcDataFiles;
    SaveSrcData(srcData, &readDatasetMainParams, &srcDataFiles);

    for (size_t i = 0; i < iface.Iterations(); ++i) {
        auto dataProvider = ReadDataset(
            /*taskType*/Nothing(),
            readDatasetMainParams.PoolPath,
            readDatasetMainParams.PairsFilePath,        // can be uninited
            readDatasetMainParams.GraphFilePath,        // can be uninited
            readDatasetMainParams.GroupWeightsFilePath, // can be uninited
            /*timestampsFilePath*/TPathWithScheme(),
            readDatasetMainParams.BaselineFilePath,     // can be uninited
            /*featureNamesFilePath*/TPathWithScheme(),
            /*poolMetaInfoFilePath*/TPathWithScheme(),
            readDatasetMainParams.ColumnarPoolFormatParams,
            TVector<ui32>{},
            EObjectsOrder::Undefined,
            TDatasetSubset::MakeColumns(),
            /*loadSampleIds*/ false,
            /*forceUnitAutoPairWeights*/ true,
            /*classLabels*/ Nothing(),
            &localExecutor);
        Y_DO_NOT_OPTIMIZE_AWAY(dataProvider);
    }
}

Y_CPU_BENCHMARK(DsvLoaderCatFeatures, iface) {
    TReadDatasetMainParams readDatasetMainParams;
    NPar::TLocalExecutor localExecutor;
    TSrcData srcData;

    TString Cd = "0\tTarget\n";
    for (size_t feature = 0; feature < FeaturesCount; ++feature) {
        Cd += ToString(feature + 1) + "\tCateg\n";
    }
    TString DatasetFileData = GetPool();

    srcData.CdFileData = Cd;
    srcData.DatasetFileData = DatasetFileData;

    TVector<THolder<TTempFile>> srcDataFiles;
    SaveSrcData(srcData, &readDatasetMainParams, &srcDataFiles);

    for (size_t i = 0; i < iface.Iterations(); ++i) {
        auto dataProvider = ReadDataset(
            /*taskType*/Nothing(),
            readDatasetMainParams.PoolPath,
            readDatasetMainParams.PairsFilePath,        // can be uninited
            readDatasetMainParams.GraphFilePath,        // can be uninited
            readDatasetMainParams.GroupWeightsFilePath, // can be uninited
            /*timestampsFilePath*/TPathWithScheme(),
            readDatasetMainParams.BaselineFilePath,     // can be uninited
            /*featureNamesFilePath*/TPathWithScheme(),
            /*poolMetaInfoFilePath*/TPathWithScheme(),
            readDatasetMainParams.ColumnarPoolFormatParams,
            TVector<ui32>{},
            EObjectsOrder::Undefined,
            TDatasetSubset::MakeColumns(),
            /*loadSampleIds*/ false,
            /*forceUnitAutoPairWeights*/ false,
            /*classLabels*/ Nothing(),
            &localExecutor);
        Y_DO_NOT_OPTIMIZE_AWAY(dataProvider);
    }
}

Y_CPU_BENCHMARK(DsvLoaderQuotedCatFeatures, iface) {
    TReadDatasetMainParams readDatasetMainParams;
    NPar::TLocalExecutor localExecutor;
    TSrcData srcData;

    TString Cd = "0\tTarget\n";
    for (size_t feature = 0; feature < FeaturesCount; ++feature) {
        Cd += ToString(feature + 1) + "\tCateg\n";
    }
    TString DatasetFileData = GetQuotedPool();

    srcData.CdFileData = Cd;
    srcData.DatasetFileData = DatasetFileData;

    TVector<THolder<TTempFile>> srcDataFiles;
    SaveSrcData(srcData, &readDatasetMainParams, &srcDataFiles);

    for (size_t i = 0; i < iface.Iterations(); ++i) {
        auto dataProvider = ReadDataset(
            /*taskType*/Nothing(),
            readDatasetMainParams.PoolPath,
            readDatasetMainParams.PairsFilePath,        // can be uninited
            readDatasetMainParams.GraphFilePath,        // can be uninited
            readDatasetMainParams.GroupWeightsFilePath, // can be uninited
            /*timestampsFilePath*/TPathWithScheme(),
            readDatasetMainParams.BaselineFilePath,     // can be uninited
            /*featureNamesFilePath*/TPathWithScheme(),
            /*poolMetaInfoFilePath*/TPathWithScheme(),
            readDatasetMainParams.ColumnarPoolFormatParams,
            TVector<ui32>{},
            EObjectsOrder::Undefined,
            TDatasetSubset::MakeColumns(),
            /*loadSampleIds*/ false,
            /*forceUnitAutoPairWeights*/ false,
            /*classLabels*/ Nothing(),
            &localExecutor);
        Y_DO_NOT_OPTIMIZE_AWAY(dataProvider);
    }
}
