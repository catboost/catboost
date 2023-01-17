
#include "trainer_env.h"


namespace NCB {

    THolder<ITrainerEnv> CreateTrainerEnv(const NCatboostOptions::TCatBoostOptions& options) {
        auto env = TTrainerEnvFactory::Construct(options.GetTaskType(), options);
        CB_ENSURE(env != nullptr, "Environment for task type [" << ToString(options.GetTaskType()) << "] not found");
        return THolder<ITrainerEnv>(env);
    }

    namespace {

    TTrainerEnvFactory::TRegistrator<TCpuTrainerEnv> CpuTrainerInitReg(ETaskType::CPU);

    }
}
