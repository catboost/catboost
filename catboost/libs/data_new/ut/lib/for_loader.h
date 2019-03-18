#pragma once

#include <catboost/libs/data_new/order.h>

#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/options/load_options.h>

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
        TStringBuf CdFileData;
        TStringBuf DsvFileData;
        bool DsvFileHasHeader = false;
        TStringBuf PairsFileData;
        TStringBuf GroupWeightsFileData;
        TStringBuf BaselineFileData;
        TVector<ui32> IgnoredFeatures;
        EObjectsOrder ObjectsOrder = EObjectsOrder::Undefined;
    };

    struct TReadDatasetMainParams {
        TPathWithScheme PoolPath;
        TPathWithScheme PairsFilePath; // can be uninited
        TPathWithScheme GroupWeightsFilePath; // can be uninited
        TPathWithScheme BaselineFilePath; // can be uninited
        NCatboostOptions::TDsvPoolFormatParams DsvPoolFormatParams;
    };


    void SaveSrcData(
        const TSrcData& srcData,
        TReadDatasetMainParams* readDatasetMainParams,
        TVector<THolder<TTempFile>>* srcDataFiles
    );

    }
}
