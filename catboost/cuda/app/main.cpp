#include "modes.h"

#include <catboost/libs/logging/logging.h>
#include <library/svnversion/svnversion.h>
#include <library/getopt/small/modchooser.h>

int main(const int argc, const char** argv) {
    try {
        DoInitGlobalLog("cout", LOG_MAX_PRIORITY, false, false);
        TMatrixnetLogSettings::GetRef().OutputExtendedInfo = false;
        SetVerboseLogingMode();

        TModChooser modChooser;
        modChooser.AddMode("fit", mode_fit, "train catboost on gpu");
        modChooser.DisableSvnRevisionOption();
        modChooser.SetVersionHandler(PrintProgramSvnVersion);
        return modChooser.Run(argc, argv);

    } catch (...) {
        Cerr << CurrentExceptionMessage() << Endl;
        return EXIT_FAILURE;
    }
}
