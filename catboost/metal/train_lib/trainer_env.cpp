#include <catboost/libs/train_lib/trainer_env.h>
#include <catboost/metal/native/metal_trainer.h>
#include <catboost/libs/helpers/exception.h>

namespace NCB {
namespace {
    class TMetalTrainerEnv final : public ITrainerEnv {
    public:
        explicit TMetalTrainerEnv(const NCatboostOptions::TCatBoostOptions& options) {
            const auto& system = options.SystemOptions.Get();
            CB_ENSURE(system.Devices == "-1" || system.Devices == "0",
                      "Metal currently supports the default Apple GPU only (devices='0')");
            CB_ENSURE(!system.GpuRamPart.IsSet() && !system.PinnedMemorySize.IsSet(),
                      "Metal uses unified memory; gpu_ram_part and pinned_memory_size are not supported");
            char name[256] = {};
            char error[2048] = {};
            CB_ENSURE(cbm_device_info(name, sizeof(name), error, sizeof(error)) == 0,
                      "Cannot initialize the Metal GPU backend: " << error);
        }
    };

    TTrainerEnvFactory::TRegistrator<TMetalTrainerEnv> MetalTrainerEnvRegistrator(ETaskType::GPU);
}
}
