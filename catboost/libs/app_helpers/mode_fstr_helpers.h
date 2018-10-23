#pragma once

#include "implementation_type_enum.h"

#include <catboost/libs/options/analytical_mode_params.h>

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
}
