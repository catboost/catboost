#pragma once

#include <catboost/cuda/cuda_lib/task.h>
#include <future>

namespace NCudaLib {
    class TRequestStreamCommand: public IGpuCommand {
    private:
        std::promise<ui64> StreamId;

    public:
        TRequestStreamCommand()
            : IGpuCommand(EGpuHostCommandType::RequestStream)
        {
        }

        std::future<ui64> GetStreamId() {
            return StreamId.get_future();
        }

        void SetStreamId(ui64 id) {
            StreamId.set_value(id);
        }
    };

    class TFreeStreamCommand: public IGpuCommand {
    private:
        ui32 Stream;

    public:
        explicit TFreeStreamCommand(ui32 stream)
            : IGpuCommand(EGpuHostCommandType::FreeStream)
            , Stream(stream)
        {
        }

        ui64 GetStream() const {
            return Stream;
        }
    };

}
