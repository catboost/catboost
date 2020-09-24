#include "master.h"

#include <catboost/private/libs/app_helpers/mode_fit_helpers.h>

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
