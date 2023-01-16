#pragma once

#include "future.h"
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/libs/helpers/exception.h>
#include <util/generic/noncopyable.h>
#include <util/stream/buffer.h>
#include <util/ysaveload.h>

namespace NCudaLib {
#ifdef USE_MPI

    //this primitives for small messages only.
    //Heavy memp-cpy tasks will be done separatly
    //could be used only on host
    template <class T>
    class TMpiFuture: public IDeviceFuture<T>, public TMoveOnly {
    public:
        bool Has() final {
            return Request->IsComplete();
        }

        void Wait() final {
            Request->WaitComplete();
        }

        const T& Get() final {
            if (!Has()) {
                Wait();
            }

            if (NeedDeserialization) {
                CB_ENSURE(Data.Size() >= Request->ReceivedBytes(), "Error: too big message size. Can't allocate enough memory in advance");
                TBufferInput in(Data);
                ::Load(&in, Result);
                NeedDeserialization = false;
            }
            return Result;
        }

        TMpiFuture() {
        }

        TMpiFuture(TMpiFuture&& other) = default;

    private:
        TMpiFuture(int sourceRank,
                   int tag) {
            auto& manager = GetMpiManager();

            if (std::is_pod<T>::value) {
                Request = manager.ReceivePodAsync(sourceRank, tag, &Result);
                NeedDeserialization = false;
            } else {
                const ui64 maxSize = 128 * 1024; //128KB commands are big enough
                Data.Resize(maxSize);
                Request = manager.ReceiveBufferAsync(sourceRank, tag, &Data);
            }
        }

        template <class TC>
        friend class TMpiPromise;

    private:
        TMpiManager::TMpiRequestPtr Request;
        TBuffer Data;
        bool NeedDeserialization = true;
        T Result;
    };

    template <class T>
    class TMpiPromise {
    public:
        using TFuturePtr = THolder<TMpiFuture<T>>;

        template <class TC>
        void SetValue(TC&& value) {
            CB_ENSURE(DestRank != -1, "Error: Empty promise");
            CB_ENSURE(!IsSet, "Can't set future more, than once");
            auto& manager = GetMpiManager();
            CB_ENSURE(manager.GetHostId() == SourceRank);
            manager.Send(value, DestRank, Tag);
            IsSet = true;
        }

        TMpiPromise() {
        }

        TMpiPromise(TMpiPromise&& other) = default;

        TFuturePtr GetFuture() {
            CB_ENSURE(!IsFutureCreated, "Can't create future more, than once");
            auto& manager = GetMpiManager();
            CB_ENSURE(manager.GetHostId() == DestRank, "Future could be created only on source rank");
            IsFutureCreated = true;
            auto mpiFuture = new TMpiFuture<T>(SourceRank, Tag);
            return THolder<TMpiFuture<T>>(mpiFuture);
        }

        TMpiPromise(int sourceRank,
                    int destRank)
            : SourceRank(sourceRank)
            , DestRank(destRank)
        {
            CB_ENSURE(sourceRank != destRank, "Error: sourceRank and destRank should be different");
            auto& manager = GetMpiManager();
            CB_ENSURE(manager.IsMaster(), "Error: promise could be done only on master");
            Tag = manager.NextCommunicationTag();
        }

        Y_SAVELOAD_DEFINE(SourceRank, DestRank, Tag, IsSet, IsFutureCreated);

    private:
        int SourceRank = -1;
        int DestRank = -1;
        int Tag = -1;
        bool IsSet = false;
        bool IsFutureCreated = false;
    };

    template <class T>
    inline TMpiPromise<T> CreateDevicePromise(int deviceId) {
        return TMpiPromise<T>(deviceId, GetMpiManager().GetMasterId());
    }

    class TRemoteDeviceRequest: public IDeviceRequest {
    public:
        TRemoteDeviceRequest(TMpiRequestPtr&& request) {
            Requests.push_back(std::move(request));
        }

        TRemoteDeviceRequest(TVector<TMpiRequestPtr>&& requests)
            : Requests(std::move(requests))
        {
        }

        bool IsComplete() final {
            return AreRequestsComplete(Requests);
        }

    private:
        TVector<TMpiRequestPtr> Requests;
    };

#endif
}
