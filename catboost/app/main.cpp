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
        modChooser.AddMode("calc", mode_calc, "calc model predictions");
        modChooser.AddMode("fstr", mode_fstr, "calc model fstr");
        modChooser.AddMode("plot", mode_plot, "calc metric plots for model");
        modChooser.DisableSvnRevisionOption();
        modChooser.SetVersionHandler(PrintProgramSvnVersion);
        return modChooser.Run(argc, argv);
    } catch (...) {
        Cerr << "AN EXCEPTION OCCURRED. " << CurrentExceptionMessage() << Endl;
        return EXIT_FAILURE;
    }
}
