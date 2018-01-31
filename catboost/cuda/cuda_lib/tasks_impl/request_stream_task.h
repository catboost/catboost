#pragma once

#include <catboost/cuda/cuda_lib/future/future.h>
#include <catboost/cuda/cuda_lib/task.h>

namespace NCudaLib {
    class IRequestStreamCommand: public ICommand {
    public:
        IRequestStreamCommand()
            : ICommand(EComandType::RequestStream)
        {
        }

        using TFuture = IDeviceFuture<ui32>;
        using TFuturePtr = THolder<TFuture>;

        virtual TFuturePtr GetStreamId() = 0;

        virtual void SetStreamId(ui32 id) = 0;
    };

    template <class TStreamIdPromise>
    class TRequestStreamCommand: public IRequestStreamCommand {
    private:
        TStreamIdPromise StreamId;

    public:
        TRequestStreamCommand(TStreamIdPromise&& streamId)
            : IRequestStreamCommand()
            , StreamId(std::move(streamId))
        {
        }

        TRequestStreamCommand() {
        }

        TFuturePtr GetStreamId() final {
            return StreamId.GetFuture();
        }

        void SetStreamId(ui32 id) final {
            StreamId.SetValue(id);
        }

        Y_SAVELOAD_TASK(StreamId);
    };

    class TFreeStreamCommand: public ICommand {
    private:
        TVector<ui32> Streams;

    public:
        explicit TFreeStreamCommand(TVector<ui32>&& streams)
            : ICommand(EComandType::FreeStream)
            , Streams(std::move(streams))
        {
        }

        TFreeStreamCommand()
            : ICommand(EComandType::FreeStream)
        {
        }

        const TVector<ui32>& GetStreams() const {
            return Streams;
        }

        Y_SAVELOAD_TASK(Streams);
    };

}
