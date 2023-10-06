#include <catboost/private/libs/app_helpers/mode_calc_helpers.h>
#include <catboost/private/libs/options/analytical_mode_params.h>
#include <catboost/libs/model/model.h>

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/cpp/getopt/small/last_getopt_parse_result.h>

namespace {
    class TOpenSourceModeCalcImplementation : public NCB::IModeCalcImplementation {
        int mode_calc(int argc, const char** argv) const override {
            NCB::TAnalyticalModeCommonParams params;
            size_t iterationsLimit = 0;
            size_t evalPeriod = 0;
            size_t virtualEnsemblesCount = 10;
            auto parser = NLastGetopt::TOpts();

            NCB::PrepareCalcModeParamsParser(&params, &iterationsLimit, &evalPeriod, &virtualEnsemblesCount, &parser);
            NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

            TFullModel model;
            ReadModelAndUpdateParams(&params, &iterationsLimit, &evalPeriod, &model);

            NCB::CalcModelSingleHost(params, iterationsLimit, evalPeriod, virtualEnsemblesCount, std::move(model));

            return 0;
        }
    };
}

NCB::TModeCalcImplementationFactory::TRegistrator<TOpenSourceModeCalcImplementation> YandexSpecificModeCalcImplementationRegistrator(NCB::EImplementationType::OpenSource);
