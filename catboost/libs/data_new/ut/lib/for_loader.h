#pragma once

#include <catboost/libs/data_util/path_with_scheme.h>

#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/system/tempfile.h>


namespace NCB {
    namespace NDataNewUT {

    void SaveDataToTempFile(
        TStringBuf srcData,
        TPathWithScheme* dstPath,

        // TODO(akhropov): temporarily use THolder until TTempFile move semantic is fixed
        TVector<THolder<TTempFile>>* srcDataFiles
    );

    }
}
