#pragma once

#include "for_data_provider.h"

#include <catboost/libs/data/order.h>

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/options/load_options.h>

#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/system/tempfile.h>
#include <util/system/types.h>


namespace NCB {
    namespace NDataNewUT {

    void SaveDataToTempFile(
        TStringBuf srcData,
        TPathWithScheme* dstPath,

        // TODO(akhropov): temporarily use THolder until TTempFile move semantic is fixed
        TVector<THolder<TTempFile>>* srcDataFiles
    );

    struct TSrcData {
        TStringBuf Scheme;
        TStringBuf CdFileData;
        TStringBuf DatasetFileData;
        bool DsvFileHasHeader = false;
        TStringBuf PairsFileData;
        TStringBuf GroupWeightsFileData;
        TStringBuf BaselineFileData;
        TStringBuf FeatureNamesFileData;
        TVector<ui32> IgnoredFeatures;
        EObjectsOrder ObjectsOrder = EObjectsOrder::Undefined;
    };

    struct TReadDatasetMainParams {
        TPathWithScheme PoolPath;
        TPathWithScheme PairsFilePath; // can be uninited
        TPathWithScheme GroupWeightsFilePath; // can be uninited
        TPathWithScheme BaselineFilePath; // can be uninited
        TPathWithScheme FeatureNamesFilePath; // can be uninited
        NCatboostOptions::TColumnarPoolFormatParams ColumnarPoolFormatParams;
    };


    void SaveSrcData(
        const TSrcData& srcData,
        TReadDatasetMainParams* readDatasetMainParams,
        TVector<THolder<TTempFile>>* srcDataFiles
    );

    struct TReadDatasetTestCase {
        TSrcData SrcData;
        TExpectedRawData ExpectedData;
    };

    void TestReadDataset(const TReadDatasetTestCase& testCase);

    }
}
