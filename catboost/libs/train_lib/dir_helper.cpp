#include "dir_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <util/folder/path.h>
#include <util/generic/string.h>


namespace NCB {

    namespace NPrivate {

        void CreateTrainDirWithTmpDirIfNotExist(const TString& path, TString* tmpSubDirPath) {
            TFsPath trainDirPath(path);
            try {
                if (!path.empty()) {
                    trainDirPath.MkDir();
                }
            } catch (...) {
                ythrow TCatBoostException() << "Can't create train working dir: " << path;
            }
            TFsPath tmpDirPath = trainDirPath / "tmp";
            try {
                tmpDirPath.MkDir();
            } catch (...) {
                ythrow TCatBoostException() << "Can't create train tmp dir: " << tmpDirPath.GetName();
            }
            *tmpSubDirPath = tmpDirPath.GetPath();
        }

    }
}
