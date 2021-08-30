#include "master.h"

#include <catboost/private/libs/app_helpers/mode_fit_helpers.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/options/system_options.h>

#include <util/generic/cast.h>


int ModeFitImpl(const TVector<TString>& args) throw (yexception) {
    const char* argv0 = "spark.native_impl.MasterApp";

    TVector<const char*> argv;
    argv.push_back(argv0);
    for (const auto& arg : args) {
        argv.push_back(arg.data());
    }

    return NCB::ModeFitImpl(SafeIntegerCast<int>(argv.size()), argv.data());
}

void ShutdownWorkers(const TString& hostsFile) throw (yexception) {
    NCatboostOptions::TSystemOptions systemOptions(ETaskType::CPU);
    systemOptions.NodeType = ENodeType::Master;
    systemOptions.FileWithHosts = hostsFile;

    TMasterContext masterContext(systemOptions);
}
