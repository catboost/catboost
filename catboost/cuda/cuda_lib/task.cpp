#include <catboost/cuda/cuda_lib/serialization/task_factory.h>
#include "task.h"

namespace NCudaLib {
    TSerializedCommand::TSerializedCommand(TBuffer&& data)
        : ICommand(EComandType::SerializedCommand)
    {
        Data.Swap(data);
    }

    THolder<ICommand> TSerializedCommand::Deserialize() {
        return TTaskSerializer::LoadCommand(Data);
    }
#if defined(USE_MPI)
    REGISTER_TASK(0x000007, TResetCommand);
    REGISTER_TASK(0x000008, TStopWorkerCommand);
#endif
}
