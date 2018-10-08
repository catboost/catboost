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

    }
}
