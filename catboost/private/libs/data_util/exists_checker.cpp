
#include "exists_checker.h"


namespace NCB {

    bool CheckExists(const TPathWithScheme& pathWithScheme) {
        return GetProcessor<IExistsChecker>(pathWithScheme)->Exists(pathWithScheme);
    }

    bool IsSharedFs(const TPathWithScheme& pathWithScheme) {
        return GetProcessor<IExistsChecker>(pathWithScheme)->IsSharedFs();
    }

    namespace {

    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSExistsCheckerReg("");
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSFileExistsCheckerReg("file");
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSDsvExistsCheckerReg("dsv");

    }
}
