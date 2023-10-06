#include "modes.h"

#include <catboost/private/libs/init/init_reg.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/getopt/small/modchooser.h>
#include <library/cpp/svnversion/svnversion.h>

#include <util/generic/yexception.h>

#include <util/stream/output.h>

#include <cstdlib>

int main(int argc, const char* argv[]) {
    try {
        NCB::TCmdLineInit::Do(argc, argv);

        TSetLoggingVerbose inThisScope;
        TModChooser modChooser;
        modChooser.AddMode("fit", mode_fit, "train model");
        modChooser.AddMode("calc", mode_calc, "evaluate model predictions");
        modChooser.AddMode("dataset-statistics", mode_dataset_statistics, "calculate dataset statistics");
        modChooser.AddMode("fstr", mode_fstr, "evaluate feature importances");
        modChooser.AddMode("ostr", mode_ostr, "evaluate object importances");
        modChooser.AddMode("eval-metrics", mode_eval_metrics, "evaluate metrics for model");
        modChooser.AddMode("eval-feature", mode_eval_feature, "evaluate features");
        modChooser.AddMode("metadata", mode_metadata, "get/set/dump metainfo fields from model");
        modChooser.AddMode("model-sum", mode_model_sum, "sum model files");
        modChooser.AddMode("run-worker", mode_run_worker, "run worker");
        modChooser.AddMode("roc", mode_roc, "evaluate data for roc curve");
        modChooser.AddMode("model-based-eval", mode_model_based_eval, "model-based eval");
        modChooser.AddMode("normalize-model", mode_normalize_model, "normalize model on a pool");
        modChooser.AddMode("select-features", mode_select_features, "select most important features");
        modChooser.AddMode("dump-options", mode_dump_options, "dump training options");
        modChooser.DisableSvnRevisionOption();
        modChooser.SetVersionHandler(PrintProgramSvnVersion);
        return modChooser.Run(argc, argv);
    } catch (...) {
        Cerr << "AN EXCEPTION OCCURRED. " << CurrentExceptionMessage() << Endl;
        return EXIT_FAILURE;
    }
}
