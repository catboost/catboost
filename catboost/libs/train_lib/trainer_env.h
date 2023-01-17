#pragma once

#include <catboost/private/libs/options/catboost_options.h>
#include "catboost/private/libs/options/enums.h"

#include <library/cpp/object_factory/object_factory.h>


namespace NCB {

    struct ITrainerEnv {
        virtual ~ITrainerEnv() = default;
    };

    using TTrainerEnvFactory = NObjectFactory::TParametrizedObjectFactory<ITrainerEnv, ETaskType, const NCatboostOptions::TCatBoostOptions&>;

    struct TCpuTrainerEnv : public ITrainerEnv {
        TCpuTrainerEnv(const NCatboostOptions::TCatBoostOptions&) {}
    };

    THolder<ITrainerEnv> CreateTrainerEnv(const NCatboostOptions::TCatBoostOptions& options);

}
