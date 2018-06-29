
#include "exists_checker.h"

#include <util/system/fs.h>


namespace NCB {

    bool CheckExists(const TPathWithScheme& pathWithScheme) {
        return GetProcessor<IExistsChecker>(pathWithScheme)->Exists(pathWithScheme);
    }


    namespace {

    struct TFSExistsChecker : public IExistsChecker {
        bool Exists(const TPathWithScheme& pathWithScheme) const override {
            return NFs::Exists(pathWithScheme.Path);
        }
    };


    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSExistsCheckerReg("");
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSFileExistsCheckerReg("file");
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSDsvExistsCheckerReg("dsv");
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSQuantizedExistsCheckerReg("quantized");

    }
}
