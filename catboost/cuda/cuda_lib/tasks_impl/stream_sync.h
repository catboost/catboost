#pragma once

#include "kernel_task.h"
#include <catboost/cuda/cuda_lib/task.h>
#include <catboost/cuda/cuda_lib/single_device.h>
#include <future>

namespace NCudaLib {
    class TTwoDevicesStreamSync: public IGpuStatelessKernelTask {
    private:
        NThreading::TFuture<TCudaEventPtr> SyncEvent;
        NThreading::TPromise<TCudaEventPtr> MyEvent;
        bool IsReady = false;

    protected:
        void SubmitAsyncExecImpl(const TCudaStream& stream) override {
            auto event = CreateCudaEvent();
            event->Record(stream);
            MyEvent.SetValue(std::move(event));
        }

        bool ReadyToSubmitNextImpl(const TCudaStream& stream) override {
            if (IsReady) {
                return true;
            }
            if (SyncEvent.HasValue()) {
                SyncEvent.GetValue(TDuration::Max())->StreamWait(stream);
                IsReady = true;
                return true;
            } else {
                return false;
            }
        }

    public:
        explicit TTwoDevicesStreamSync(ui32 stream = 0)
            : IGpuStatelessKernelTask(stream)
            , MyEvent(NThreading::NewPromise<TCudaEventPtr>())
        {
        }

        NThreading::TFuture<TCudaEventPtr> GetMyEvent() {
            return MyEvent.GetFuture();
        }

        void SetSyncOnEvent(NThreading::TFuture<TCudaEventPtr>&& event) {
            SyncEvent = std::move(event);
        }
    };

    class TSingleHostStreamSync {
    private:
        ui32 StreamId;

        TSet<TCudaSingleDevice*> DevicesToSync;

        void SubmitTwoDevicesSync(TCudaSingleDevice* leftDevice,
                                  TCudaSingleDevice* rightDevice) {
            if (leftDevice != rightDevice) {
                auto leftTask = new TTwoDevicesStreamSync(StreamId);
                auto rightTask = new TTwoDevicesStreamSync(StreamId);
                leftTask->SetSyncOnEvent(rightTask->GetMyEvent());
                rightTask->SetSyncOnEvent(leftTask->GetMyEvent());
                leftDevice->AddTask(THolder<IGpuCommand>(leftTask));
                rightDevice->AddTask(THolder<IGpuCommand>(rightTask));
            } else {
                leftDevice->StreamSynchronize(StreamId);
            }
        }

    public:
        explicit TSingleHostStreamSync(ui32 stream = 0)
            : StreamId(stream)
        {
        }

        void AddDevice(TCudaSingleDevice* device) {
            DevicesToSync.insert(device);
        }

        void AddDevice(ui32 device);

        void operator()() {
            TVector<TCudaSingleDevice*> devices(DevicesToSync.begin(), DevicesToSync.end());
            const ui64 iterations = static_cast<const ui64>(devices.size() ? 1 + MostSignificantBit(devices.size()) : 0);
            for (ui32 iter = 0; iter < iterations; ++iter) {
                ui64 bit = 1 << (iterations - iter - 1);
                for (ui64 dev = 0; dev < bit; ++dev) {
                    Y_ASSERT(dev < devices.size());
                    const ui64 secondDev = dev | bit;
                    if (secondDev < devices.size()) {
                        SubmitTwoDevicesSync(devices[dev], devices[secondDev]);
                    } else {
                        SubmitTwoDevicesSync(devices[dev], devices[dev]);
                    }
                }
            }
        }
    };

}
