#include "modes.h"

#include <catboost/libs/logging/logging.h>

#include <library/svnversion/svnversion.h>
#include <library/getopt/small/modchooser.h>


int main(int argc, const char* argv[]) {
    try {
        DoInitGlobalLog("cout", LOG_MAX_PRIORITY, false, false);
        TMatrixnetLogSettings::GetRef().OutputExtendedInfo = false;
        SetVerboseLogingMode();
        TModChooser modChooser;
        modChooser.AddMode("fit", mode_fit, "train model");
        modChooser.AddMode("calc", mode_calc, "evaluate model predictions");
        modChooser.AddMode("fstr", mode_fstr, "evaluate feature importances");
        modChooser.AddMode("ostr", mode_ostr, "evaluate object importances");
        modChooser.AddMode("eval-metrics", mode_eval_metrics, "evaluate metrics for model");
        modChooser.DisableSvnRevisionOption();
        modChooser.SetVersionHandler(PrintProgramSvnVersion);
        return modChooser.Run(argc, argv);
    } catch (...) {
        Cerr << "AN EXCEPTION OCCURRED. " << CurrentExceptionMessage() << Endl;
        return EXIT_FAILURE;
    }
}
