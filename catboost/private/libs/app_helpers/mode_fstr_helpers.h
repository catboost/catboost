#pragma once

#include "implementation_type_enum.h"

#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/analytical_mode_params.h>

#include <library/cpp/getopt/small/last_getopt_opts.h>
#include <library/cpp/object_factory/object_factory.h>


namespace NCB {
    class IModeFstrImplementation {
    public:
        virtual int mode_fstr(int argc, const char** argv) const = 0;
        virtual ~IModeFstrImplementation() = default;
    };

    using TModeFstrImplementationFactory = NObjectFactory::TParametrizedObjectFactory<IModeFstrImplementation, EImplementationType>;

    void PrepareFstrModeParamsParser(
        NCB::TAnalyticalModeCommonParams* paramsPtr,
        NLastGetopt::TOpts* parserPtr);

    void ModeFstrSingleHost(const NCB::TAnalyticalModeCommonParams& params);
    void ModeFstrSingleHostInner(
        const NCB::TAnalyticalModeCommonParams& params,
        const TFullModel& model);
}
