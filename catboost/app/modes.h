#pragma once

#include <catboost/private/libs/app_helpers/mode_calc_helpers.h>
#include <catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.h>
#include <catboost/private/libs/app_helpers/mode_fstr_helpers.h>
#include <catboost/private/libs/app_helpers/mode_normalize_model_helpers.h>


int mode_fit(int argc, const char* argv[]);
int mode_ostr(int argc, const char* argv[]);
int mode_eval_metrics(int argc, const char* argv[]);
int mode_eval_feature(int argc, const char* argv[]);
int mode_metadata(int argc, const char* argv[]);
int mode_run_worker(int argc, const char* argv[]);
int mode_roc(int argc, const char* argv[]);
int mode_model_sum(int argc, const char* argv[]);
int mode_model_based_eval(int argc, const char* argv[]);
int mode_select_features(int argc, const char* argv[]);
int mode_dump_options(int argc, const char* argv[]);
int mode_dataset_statistics(int argc, const char* argv[]);

inline int mode_calc(int argc, const char** argv) {
    THolder<NCB::IModeCalcImplementation> modeCalcImplementaion;
    if (NCB::TModeCalcImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        modeCalcImplementaion.Reset(
            NCB::TModeCalcImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific)
        );
    } else {
        CB_ENSURE(NCB::TModeCalcImplementationFactory::Has(NCB::EImplementationType::OpenSource),
                  "Mode calc implementation factory should have open source implementation");
        modeCalcImplementaion.Reset(
            NCB::TModeCalcImplementationFactory::Construct(NCB::EImplementationType::OpenSource)
        );
    }
    return modeCalcImplementaion->mode_calc(argc, argv);
}

inline int mode_dataset_statistics(int argc, const char** argv) {
    THolder<NCB::IModeDatasetStatisticsImplementation> modeDatasetStatisticsImplementaion;
    if (NCB::TModeDatasetStatisticsImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        modeDatasetStatisticsImplementaion.Reset(
            NCB::TModeDatasetStatisticsImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific)
        );
    } else {
        CB_ENSURE(NCB::TModeDatasetStatisticsImplementationFactory::Has(NCB::EImplementationType::OpenSource),
                  "Mode dataset statistics implementation factory should have open source implementation");
        modeDatasetStatisticsImplementaion.Reset(
            NCB::TModeDatasetStatisticsImplementationFactory::Construct(NCB::EImplementationType::OpenSource)
        );
    }
    return modeDatasetStatisticsImplementaion->mode_dataset_statistics(argc, argv);
}

inline int mode_fstr(int argc, const char** argv) {
    THolder<NCB::IModeFstrImplementation> modeFstrImplementaion;
    if (NCB::TModeFstrImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        modeFstrImplementaion.Reset(
            NCB::TModeFstrImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific)
        );
    } else {
        CB_ENSURE(NCB::TModeFstrImplementationFactory::Has(NCB::EImplementationType::OpenSource),
                  "Mode fstr implementation factory should have open source implementation");
        modeFstrImplementaion.Reset(
            NCB::TModeFstrImplementationFactory::Construct(NCB::EImplementationType::OpenSource)
        );
    }
    return modeFstrImplementaion->mode_fstr(argc, argv);
}

inline int mode_normalize_model(int argc, const char** argv) {
    THolder<NCB::IModeNormalizeModelImplementation> impl;
    if (NCB::TModeNormalizeModelImplementationFactory::Has(NCB::EImplementationType::YandexSpecific)) {
        impl.Reset(NCB::TModeNormalizeModelImplementationFactory::Construct(NCB::EImplementationType::YandexSpecific));
    } else {
        CB_ENSURE(NCB::TModeNormalizeModelImplementationFactory::Has(NCB::EImplementationType::OpenSource),
                  "Missing normalize-model implementation");
        impl.Reset(NCB::TModeNormalizeModelImplementationFactory::Construct(NCB::EImplementationType::OpenSource));
    }
    return impl->mode_normalize_model(argc, argv);
}
