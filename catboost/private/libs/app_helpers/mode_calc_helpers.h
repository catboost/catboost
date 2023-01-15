#pragma once

#include "implementation_type_enum.h"

#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/analytical_mode_params.h>

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/object_factory/object_factory.h>


namespace NCB {
    class IModeCalcImplementation {
    public:
        virtual int mode_calc(int argc, const char** argv) const = 0;
        virtual ~IModeCalcImplementation() = default;
    };

    using TModeCalcImplementationFactory = NObjectFactory::TParametrizedObjectFactory<IModeCalcImplementation, EImplementationType>;

    void PrepareCalcModeParamsParser(
        NCB::TAnalyticalModeCommonParams* paramsPtr,
        size_t* iterationsLimitPtr,
        size_t* evalPeriodPtr,
        NLastGetopt::TOpts* parserPtr);

    void ReadModelAndUpdateParams(
        NCB::TAnalyticalModeCommonParams* paramsPtr,
        size_t* iterationsLimitPtr,
        size_t* evalPeriodPtr,
        TFullModel* modelPtr);

    void CalcModelSingleHost(
        const NCB::TAnalyticalModeCommonParams& params,
        size_t iterationsLimit,
        size_t evalPeriod,
        TFullModel&& model);
}
