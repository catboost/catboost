#pragma once

#include "cuda_base.h"
#include "cuda_events_provider.h"
#include "task.h"
#include "memory_provider_trait.h"
#include "remote_device_future.h"
#include "single_device.h"
#include "helpers.h"
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

    struct TCopyTask {
        ui64 FromOffset = 0;
        ui64 ToOffset = 0;
        ui64 Size = 0;

        SAVELOAD(FromOffset, Size, ToOffset);
    };

    template <EPtrType FromPtrType, EPtrType ToPtrType>
    class TSingleHostAsyncCopyTask: public IGpuStatelessKernelTask {
    private:
        using TToBuffer = THandleBasedMemoryPointer<char, ToPtrType>;
        using TFromBuffer = THandleBasedMemoryPointer<char, FromPtrType>;

        ui64 To;
        ui64 From;

        TCopyTask CopyTask;

    protected:
        void SubmitAsyncExecImpl(const TCudaStream& stream) override {
            ui64 readOffset = CopyTask.FromOffset;
            ui64 readSize = CopyTask.Size;
            ui64 writeOffset = CopyTask.ToOffset;
            if (readSize != 0u) {
                TFromBuffer from(From, readOffset);
                TFromBuffer to(To, writeOffset);

                TMemoryCopier<FromPtrType, ToPtrType>::CopyMemoryAsync(from.Get(), to.Get(), readSize, stream);
            }
        }

    public:
        TSingleHostAsyncCopyTask(const ui64 to,
                                 const ui64 from,
                                 const TCopyTask& copyTask,
                                 ui32 stream = 0)
            : IGpuStatelessKernelTask(stream)
            , To(to)
            , From(from)
            , CopyTask(copyTask)
        {
        }

        SAVELOAD(From, To, CopyTask);
    };

    template <EPtrType DevicePtr>
    class TMasterMemcpy: public IGpuStatelessKernelTask {
    private:
        ui64 DevicePtrHandle;
        char* HostPtr;
        TCopyTask CopyTask;
        bool ReadFromHost;

        NThreading::TPromise<TCudaEventPtr> DoneEventPromise;

    protected:
        void SubmitAsyncExecImpl(const TCudaStream& stream) override {
            TCudaEventPtr eventPtr = CreateCudaEvent();

            THandleBasedMemoryPointer<char, DevicePtr> buffer(DevicePtrHandle);

            char* src = (ReadFromHost ? HostPtr : buffer.Get()) + CopyTask.FromOffset;
            char* dst = (ReadFromHost ? buffer.Get() : HostPtr) + CopyTask.ToOffset;

            if (ReadFromHost) {
                TMemoryCopier<Host, DevicePtr>::template CopyMemoryAsync<char>(src, dst, CopyTask.Size, stream);
            } else {
                TMemoryCopier<DevicePtr, Host>::template CopyMemoryAsync<char>(src, dst, CopyTask.Size, stream);
            }
            eventPtr->Record(stream);

            DoneEventPromise.SetValue(std::move(eventPtr));
        }

    public:
        TMasterMemcpy(const ui64 handle,
                      char* hostPtr,
                      const TCopyTask& copyTask,
                      bool isRead,
                      ui32 stream = 0)
            : IGpuStatelessKernelTask(stream)
            , DevicePtrHandle(handle)
            , HostPtr(hostPtr)
            , CopyTask(copyTask)
            , ReadFromHost(isRead)
            , DoneEventPromise(NThreading::NewPromise<TCudaEventPtr>())
        {
        }

        TDeviceEvent DoneEvent() {
            return TDeviceEvent(DoneEventPromise.GetFuture());
        }
    };

    class TSingleHostStreamSync {
    private:
        ui32 StreamId;

        yset<TCudaSingleDevice*> DevicesToSync;

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

    class TDataCopier {
    private:
        ui32 StreamId = 0;
        bool Submitted = false;
        TSingleHostStreamSync StreamSync;

        struct TTask {
            TCudaSingleDevice* Device;
            THolder<IGpuCommand> Cmd;
        };
        TVector<TTask> MemoryCopyTasks;

    public:
        explicit TDataCopier(ui32 streamId = 0)
            : StreamId(streamId)
            , Submitted(true)
            , StreamSync(StreamId)
        {
        }

        ~TDataCopier() throw (yexception) {
            CB_ENSURE(Submitted, "Copy task wasn't submitted");
        }

        template <typename T, class TC, EPtrType FromType, EPtrType ToType>
        TDataCopier& AddAsyncMemoryCopyTask(const TCudaSingleDevice::TSingleBuffer<T, FromType>& from, ui64 readOffset,
                                            TCudaSingleDevice::TSingleBuffer<TC, ToType>& to, ui64 writeOffset, ui64 writeSize) {
            Submitted = false;
            static_assert(sizeof(T) == sizeof(TC), "Error types should have equal size");

            TCopyTask copyTask;
            copyTask.Size = writeSize * sizeof(T);
            copyTask.FromOffset = (from.GetOffset() + readOffset) * sizeof(T);
            copyTask.ToOffset = (to.GetOffset() + writeOffset) * sizeof(T);

            auto copyTaskCmd = new TSingleHostAsyncCopyTask<FromType, ToType>(to.MemoryHandle(),
                                                                              from.MemoryHandle(),
                                                                              copyTask,
                                                                              StreamId);
            MemoryCopyTasks.push_back({to.Owner, THolder<IGpuCommand>(copyTaskCmd)});
            StreamSync.AddDevice(from.Owner);
            StreamSync.AddDevice(to.Owner);
            return *this;
        }

        void SubmitCopy() {
            if (MemoryCopyTasks.size()) {
                StreamSync();
                //now handles are 100% good
                for (auto& task : MemoryCopyTasks) {
                    task.Device->AddTask(std::move(task.Cmd));
                }
                //ensure that any implicit synchronized tasks will see memcpy results (aka defragmentation, objects destruction, cpu functions)
                StreamSync();
                Submitted = true;
            }
        }

        template <class T>
        static inline char* ToCopyPointerType(const T* ptr) {
            return reinterpret_cast<char*>(const_cast<typename std::remove_const<T>::type*>(ptr));
        }

        template <typename T, EPtrType DevicePtr>
        static TDeviceEvent AsyncWrite(const T* from, TCudaSingleDevice::TSingleBuffer<T, DevicePtr>& buffer, ui32 stream, ui64 writeOffset, ui64 writeSize) {
            TCopyTask copyTask;
            copyTask.Size = writeSize * sizeof(T);
            copyTask.ToOffset = (buffer.GetOffset() + writeOffset) * sizeof(T);
            copyTask.FromOffset = 0;

            auto task = new TMasterMemcpy<DevicePtr>(buffer.MemoryHandle(), ToCopyPointerType<T>(from), copyTask, true, stream);
            auto isDone = task->DoneEvent();
            buffer.Owner->AddTask(THolder<IGpuCommand>(task));
            return isDone;
        };

        template <typename T, EPtrType DevicePtr>
        static TDeviceEvent AsyncRead(const TCudaSingleDevice::TSingleBuffer<T, DevicePtr>& from, ui32 stream, ui64 fromOffset, typename std::remove_const<T>::type* to, ui64 writeSize) {
            TCopyTask copyTask;
            copyTask.Size = writeSize * sizeof(T);
            copyTask.ToOffset = 0;
            copyTask.FromOffset = (from.GetOffset() + fromOffset) * sizeof(T);

            auto task = new TMasterMemcpy<DevicePtr>(from.MemoryHandle(), ToCopyPointerType<T>(to), copyTask, false, stream);
            auto isDone = task->DoneEvent();
            from.Owner->AddTask(THolder<IGpuCommand>(task));
            return isDone;
        };
    };
}
