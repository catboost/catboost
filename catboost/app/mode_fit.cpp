#include "modes.h"

#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/app_helpers/mode_fit_helpers.h>


using namespace NCB;


int mode_fit(int argc, const char* argv[]) {
    ConfigureMalloc();

    return ModeFitImpl(argc, argv);
}

