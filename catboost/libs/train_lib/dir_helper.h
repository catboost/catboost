#pragma once

#include <util/generic/fwd.h>


namespace NCB {

    namespace NPrivate {

        // tmpSubDir is created inside trainDir
        void CreateTrainDirWithTmpDirIfNotExist(const TString& path, TString* tmpSubDirPath);

    }
}
