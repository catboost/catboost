#include "for_loader.h"

#include <util/stream/file.h>
#include <util/system/mktemp.h>


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
            &(readDatasetMainParams->DsvPoolFormatParams.CdFilePath),
            srcDataFiles
        );
        SaveDataToTempFile(srcData.DsvFileData, &(readDatasetMainParams->PoolPath), srcDataFiles);
        readDatasetMainParams->DsvPoolFormatParams.Format.HasHeader = srcData.DsvFileHasHeader;
        SaveDataToTempFile(srcData.PairsFileData, &(readDatasetMainParams->PairsFilePath), srcDataFiles);
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
    }

    }
}
