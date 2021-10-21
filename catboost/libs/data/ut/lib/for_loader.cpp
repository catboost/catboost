#include "for_loader.h"

#include <catboost/libs/data/load_data.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/stream/file.h>
#include <util/system/tempfile.h>


namespace NCB {
    namespace NDataNewUT {

    void SaveDataToTempFile(
        TStringBuf srcData,
        TPathWithScheme* dstPath,
        TVector<THolder<TTempFile>>* srcDataFiles
    ) {
        if (srcData) {
            auto tmpFileName = MakeTempName();
            TFileOutput output(tmpFileName);
            output.Write(srcData);
            *dstPath = TPathWithScheme(tmpFileName);
            srcDataFiles->emplace_back(MakeHolder<TTempFile>(tmpFileName));
        }
    }

    void SaveSrcData(
        const TSrcData& srcData,
        TReadDatasetMainParams* readDatasetMainParams,
        TVector<THolder<TTempFile>>* srcDataFiles
    ) {
        SaveDataToTempFile(
            srcData.CdFileData,
            &(readDatasetMainParams->ColumnarPoolFormatParams.CdFilePath),
            srcDataFiles
        );
        SaveDataToTempFile(srcData.DatasetFileData, &(readDatasetMainParams->PoolPath), srcDataFiles);
        readDatasetMainParams->PoolPath.Scheme = srcData.Scheme;
        readDatasetMainParams->ColumnarPoolFormatParams.DsvFormat.HasHeader = srcData.DsvFileHasHeader;
        readDatasetMainParams->ColumnarPoolFormatParams.DsvFormat.NumVectorDelimiter
            = srcData.NumVectorDelimiter;
        SaveDataToTempFile(srcData.PairsFileData, &(readDatasetMainParams->PairsFilePath), srcDataFiles);
        readDatasetMainParams->PairsFilePath.Scheme = srcData.PairsScheme;
        SaveDataToTempFile(
            srcData.GroupWeightsFileData,
            &(readDatasetMainParams->GroupWeightsFilePath),
            srcDataFiles
        );
        SaveDataToTempFile(
            srcData.BaselineFileData,
            &(readDatasetMainParams->BaselineFilePath),
            srcDataFiles
        );
        SaveDataToTempFile(
            srcData.FeatureNamesFileData,
            &(readDatasetMainParams->FeatureNamesFilePath),
            srcDataFiles
        );
    }

    void TestReadDataset(const TReadDatasetTestCase& testCase) {
        TReadDatasetMainParams readDatasetMainParams;

        // TODO(akhropov): temporarily use THolder until TTempFile move semantic are fixed
        TVector<THolder<TTempFile>> srcDataFiles;

        SaveSrcData(testCase.SrcData, &readDatasetMainParams, &srcDataFiles);

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(3);

        auto readDataset = [&] () {
            return ReadDataset(
                /*taskType*/Nothing(),
                readDatasetMainParams.PoolPath,
                readDatasetMainParams.PairsFilePath, // can be uninited
                readDatasetMainParams.GroupWeightsFilePath, // can be uninited
                /*timestampsFilePath*/TPathWithScheme(),
                readDatasetMainParams.BaselineFilePath, // can be uninited
                readDatasetMainParams.FeatureNamesFilePath, // can be uninited
                /*poolMetaInfoFilePath*/TPathWithScheme(),
                readDatasetMainParams.ColumnarPoolFormatParams,
                testCase.SrcData.IgnoredFeatures,
                testCase.SrcData.ObjectsOrder,
                TDatasetSubset::MakeColumns(),
                /*forceUnitAutoPairWeights*/ true,
                /*classLabels*/Nothing(),
                &localExecutor
            );
        };

        if (testCase.ExpectedReadError) {
            UNIT_ASSERT_EXCEPTION(readDataset(), TCatBoostException);
        } else {
            TDataProviderPtr dataProvider = readDataset();
            Compare<TRawObjectsDataProvider>(std::move(dataProvider), testCase.ExpectedData);
        }
    }

    }
}
