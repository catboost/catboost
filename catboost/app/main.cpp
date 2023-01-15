#include "modes.h"

#include <catboost/private/libs/app_helpers/mode_calc_helpers.h>
#include <catboost/private/libs/app_helpers/mode_fstr_helpers.h>
#include <catboost/private/libs/app_helpers/mode_normalize_model_helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/init/init_reg.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/getopt/small/modchooser.h>
#include <library/svnversion/svnversion.h>

#include <util/generic/ptr.h>
#include <util/stream/output.h>

#include <cstdlib>

static int mode_calc(int argc, const char** argv) {
    THolder<NCB::IModeCalcImplementation> modeCalcImplementaion;
    if (NCB::TModeCalcImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        modeCalcImplementaion = NCB::TModeCalcImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific);
    } else {
        CB_ENSURE(NCB::TModeCalcImplementationFactory::Has(NCB::EImplementationType::OpenSource),
            "Mode calc implementation factory should have open source implementation");
        modeCalcImplementaion = NCB::TModeCalcImplementationFactory::Construct(NCB::EImplementationType::OpenSource);
    }
    return modeCalcImplementaion->mode_calc(argc, argv);
}

static int mode_fstr(int argc, const char** argv) {
    THolder<NCB::IModeFstrImplementation> modeFstrImplementaion;
    if (NCB::TModeFstrImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        modeFstrImplementaion = NCB::TModeFstrImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific);
    } else {
        CB_ENSURE(NCB::TModeFstrImplementationFactory::Has(NCB::EImplementationType::OpenSource),
            "Mode fstr implementation factory should have open source implementation");
        modeFstrImplementaion = NCB::TModeFstrImplementationFactory::Construct(NCB::EImplementationType::OpenSource);
    }
    return modeFstrImplementaion->mode_fstr(argc, argv);
}

static int mode_normalize_model(int argc, const char** argv) {
    THolder<NCB::IModeNormalizeModelImplementation> impl;
    if (NCB::TModeNormalizeModelImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        impl = NCB::TModeNormalizeModelImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific);
    } else {
        CB_ENSURE(NCB::TModeNormalizeModelImplementationFactory::Has(NCB::EImplementationType::OpenSource),
            "Missing normalize-model implementation");
        impl = NCB::TModeNormalizeModelImplementationFactory::Construct(NCB::EImplementationType::OpenSource);
    }
    return impl->mode_normalize_model(argc, argv);
}

int main(int argc, const char* argv[]) {
    try {
        NCB::TCmdLineInit::Do(argc, argv);

        TSetLoggingVerbose inThisScope;
        TModChooser modChooser;
        modChooser.AddMode("fit", mode_fit, "train model");
        modChooser.AddMode("calc", mode_calc, "evaluate model predictions");
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
        modChooser.DisableSvnRevisionOption();
        modChooser.SetVersionHandler(PrintProgramSvnVersion);
        return modChooser.Run(argc, argv);
    } catch (...) {
        Cerr << "AN EXCEPTION OCCURRED. " << CurrentExceptionMessage() << Endl;
        return EXIT_FAILURE;
    }
}
