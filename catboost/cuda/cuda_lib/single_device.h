#pragma once

#include "cuda_base.h"
#include "remote_objects.h"
#include "gpu_single_worker.h"
#include "cuda_events_provider.h"
#include "kernel.h"

#include <catboost/cuda/cuda_lib/tasks_impl/host_tasks.h>
#include <catboost/cuda/cuda_lib/tasks_impl/memory_allocation.h>
#include <catboost/cuda/cuda_lib/future/future.h>
#include <catboost/cuda/cuda_lib/tasks_impl/cpu_func.h>
#include <catboost/cuda/cuda_lib/tasks_queue/mpi_task_queue.h>

namespace NCudaLib {
    class TCudaSingleDevice {
    private:
        template <EPtrType PtrType>
        using TPtr = typename TMemoryProviderImplTrait<PtrType>::TRawFreeMemory;
        const ui32 OBJECT_HANDLE_REQUEST_SIZE = 1024;

        using TLocalQueue = TSingleHostTaskQueue;
        template <class TFunc>
        using TLocalFunc = TCpuFunc<TFunc, false>;

#if defined(USE_MPI)
        using TRemoteQueue = TRemoteHostTasksForwarder;
        ;
        template <class TFunc>
        using TRemoteFunc = TCpuFunc<TFunc, true>;
#endif

        TAtomic ExceptionsCount;
        friend class TSetDeviceExceptionCallback;

        void* TaskQueue;

        TDeviceId DeviceId;
        TCudaDeviceProperties DeviceProperties;
        bool IsStoppedFlag = true;

        TVector<ui64> FreeHandles;
        ui64 TotalHandles = 0;
        TVector<ui32> UserFreeStreams;

        template <class T>
        friend class THandleBasedObject;

    private:
        void RequestHandlesImpl();

        ui64 GetFreeHandle();

        friend class TDevicesProvider;

    public:
        template <class T>
        class THandleBasedObject: private TNonCopyable, public TThrRefBase {
        private:
            static const ui64 EMPTY_HANDLE = 0;
            TCudaSingleDevice* Device;
            ui64 Handle = EMPTY_HANDLE;

        public:
            THandleBasedObject(TCudaSingleDevice* device,
                               ui64 handle)
                : Device(device)
                , Handle(handle)
            {
            }

            ~THandleBasedObject() {
                if (Handle != EMPTY_HANDLE) {
                    Device->ResetPointer<T>(Handle);
                    Device->FreeHandles.push_back(Handle);
                }
            }

            friend class TCudaSingleDevice;
        };

        template <class T, EPtrType Type>
        class TSingleBuffer: public TMoveOnly {
        private:
            using TRawPtr = typename TMemoryProviderImplTrait<Type>::TRawFreeMemory;
            TIntrusivePtr<THandleBasedObject<TRawPtr>> Memory;
            ui64 AllocatedSize;
            TCudaSingleDevice* Owner;
            ui64 Offset;
            friend class TDataCopier;

        public:
            TSingleBuffer(TIntrusivePtr<THandleBasedObject<TRawPtr>> memory,
                          ui64 allocatedSize,
                          TCudaSingleDevice* device,
                          ui64 offset = 0)
                : Memory(std::move(memory))
                , AllocatedSize(allocatedSize)
                , Owner(device)
                , Offset(offset)
            {
            }

            TSingleBuffer()
                : Memory(nullptr)
                , AllocatedSize(0)
                , Owner(nullptr)
                , Offset(0)
            {
            }

            ui64 GetOffset() const {
                return Offset;
            }

            bool IsEmpty() const {
                return Size() == 0;
            }

            bool NotEmpty() const {
                return !IsEmpty();
            }

            ui64 Size() const {
                Y_ASSERT(Offset <= AllocatedSize);
                CB_ENSURE(Offset <= AllocatedSize, TStringBuilder() << "Offset " << Offset << " > size " << AllocatedSize);
                return AllocatedSize - Offset;
            }

            TSingleBuffer ShiftedBuffer(ui64 offset) const {
                return TSingleBuffer(Memory, AllocatedSize, Owner, Offset + offset);
            }

            TSingleBuffer<const T, Type> ShiftedConstBuffer(ui64 offset) const {
                return TSingleBuffer<const T, Type>(Memory, AllocatedSize, Owner, Offset + offset);
            }

            template <class U>
            TSingleBuffer<U, Type> ReinterpretCast() {
                static_assert(sizeof(U) == sizeof(T), "Error: support to reinterpret cast of equal element size only");
                return TSingleBuffer<U, Type>(Memory, AllocatedSize, Owner, Offset);
            }

            TSingleBuffer<std::remove_const_t<T>, Type> ConstCast() {
                return TSingleBuffer<std::remove_const_t<T>, Type>(Memory, AllocatedSize, Owner, Offset);
            }

            template <class U>
            TSingleBuffer<const U, Type> ReinterpretCast() const {
                static_assert(sizeof(U) == sizeof(T), "Error: support to reinterpret cast of equal element size only");
                return TSingleBuffer<const U, Type>(Memory, AllocatedSize, Owner, Offset);
            }

            ui64 MemoryHandle() const {
                return Memory != nullptr ? Memory->Handle : 0;
            }

            THandleBasedMemoryPointer<T, Type> GetPointer() {
                return THandleBasedMemoryPointer<T, Type>(MemoryHandle(),
                                                          Offset);
            }

            THandleBasedMemoryPointer<const T, Type> GetPointer() const {
                return THandleBasedMemoryPointer<const T, Type>(MemoryHandle(),
                                                                Offset);
            }
        };

        template <class TTask>
        void AddTask(THolder<TTask>&& cmd) {
            CB_ENSURE(TaskQueue, "Error: uninitialized device " << DeviceId.HostId << " " << DeviceId.DeviceId);

            if (IsLocalDevice()) {
                reinterpret_cast<TLocalQueue*>(TaskQueue)->AddTask(std::move(cmd));
            } else {
#if defined(USE_MPI)
                reinterpret_cast<TRemoteQueue*>(TaskQueue)->AddTask(std::move(cmd));
#else
                CB_ENSURE(false, "Remote device support is not enabled");
#endif
            }
        }

        template <class TTask,
                  class... Args>
        void EmplaceTask(Args... args) const {
            CB_ENSURE(TaskQueue, "Error: uninitialized device " << DeviceId.HostId << " " << DeviceId.DeviceId);

            const auto isLocalDevice = IsLocalDevice();
            if (isLocalDevice) {
                reinterpret_cast<TLocalQueue*>(TaskQueue)->EmplaceTask<TTask>(std::forward<Args>(args)...);
            } else {
#if defined(USE_MPI)
                reinterpret_cast<TRemoteQueue*>(TaskQueue)->EmplaceTask<TTask>(std::forward<Args>(args)...);
#else
                CB_ENSURE(false, "Remote device support is not enabled");
#endif
            }
        }

        template <bool IsRemote>
        THolder<IDeviceFuture<ui32>> RequestStreamImpl() {
            using TStreamIdPromise = typename TPromiseFactory<IsRemote>::template TPromise<ui32>;
            using TCmd = TRequestStreamCommand<TStreamIdPromise>;
            auto cmd = MakeHolder<TCmd>(TPromiseFactory<IsRemote>::template CreateDevicePromise<ui32>(DeviceId));
            auto streamFuture = cmd->GetStreamId();
            AddTask(std::move(cmd));
            return streamFuture;
        }

    public:
        TCudaSingleDevice(void* taskQueue,
                          TDeviceId deviceId,
                          const TCudaDeviceProperties& deviceProps)
            : TaskQueue(taskQueue)
            , DeviceId(deviceId)
            , DeviceProperties(deviceProps)
            , IsStoppedFlag(true)
        {
        }

        ~TCudaSingleDevice() {
            if (!IsStoppedFlag) {
                Stop();
            }
            EmplaceTask<TStopWorkerCommand>();
        }

        void Start(double gpuMemoryPart,
                   ui64 pinnedMemoryToUse) {
            CB_ENSURE(IsStoppedFlag, "Error: can't start device more than once");
            EmplaceTask<TResetCommand>(gpuMemoryPart, pinnedMemoryToUse);
            RequestHandlesImpl();
            IsStoppedFlag = false;
        }

        void Stop() {
            CB_ENSURE(!IsStoppedFlag, "Error: can't stop device more than once");
            EmplaceTask<TFreeStreamCommand>(UserFreeStreams);
            UserFreeStreams.clear();
            CB_ENSURE(TotalHandles == FreeHandles.size());
            {
                TVector<ui64> handlesToFree;
                handlesToFree.swap(FreeHandles);
                TotalHandles = 0;
                LaunchFunc(TFreeHandlesTask(std::move(handlesToFree)))->Wait();
            }
            EmplaceTask<TResetCommand>(0.0, (ui64)0);
            WaitComplete()->Wait();
            IsStoppedFlag = true;
        }

        template <class T, EPtrType Type>
        TSingleBuffer<T, Type> CreateSingleBuffer(ui64 size) {
            using TRawPtr = typename TMemoryProviderImplTrait<Type>::TRawFreeMemory;
            auto ptr = CreateRemoteObject<TRawPtr>(size * sizeof(T));
            return TSingleBuffer<T, Type>(ptr.Release(), size, this);
        };

        template <class T, class... Args>
        THolder<THandleBasedObject<T>> CreateRemoteObject(Args&&... args) {
            auto handle = GetFreeHandle();
            auto cmd = TCreateObjectCommandTrait<T>::Create(handle, std::forward<Args>(args)...);
            AddTask(std::move(cmd));
            return MakeHolder<THandleBasedObject<T>>(this, handle);
        }

        template <class TKernel>
        void LaunchKernel(TKernel&& kernel,
                          ui32 stream) const {
            using TKernelTask = TGpuKernelTask<TKernel>;
            EmplaceTask<TKernelTask>(std::forward<TKernel>(kernel), stream);
        }

        template <class T>
        void ResetPointer(ui64 handle) {
            using TTask = TResetPointerCommand<T>;
            EmplaceTask<TTask>(handle);
        }

        bool IsLocalDevice() const {
            return DeviceId.HostId == 0;
        }

        bool IsRemoteDevice() const {
            return !IsLocalDevice();
        }

        int GetHostId() const {
            return DeviceId.HostId;
        }

        bool IsStopped() const {
            return IsStoppedFlag;
        }

        int GetDeviceId() const {
            return DeviceId.DeviceId;
        }

        TDeviceId GetDevice() const {
            return DeviceId;
        }

        template <class TFunc>
        auto LaunchFunc(TFunc&& func) -> TDeviceFuturePtr<typename TFuncReturnType<TFunc>::TOutput> {
            CB_ENSURE(TaskQueue, "Error: uninitialized device " << DeviceId.HostId << " " << DeviceId.DeviceId);
            using TOutput = typename TFuncReturnType<TFunc>::TOutput;
            if (IsLocalDevice()) {
                using TTask = TLocalFunc<TFunc>;
                auto task = MakeHolder<TTask>(TPromiseFactory<false>::template CreateDevicePromise<TOutput>(DeviceId),
                                              std::forward<TFunc>(func));
                auto futureResult = task->GetResult();
                AddTask(std::move(task));
                return std::move(futureResult);
            } else {
#if defined(USE_MPI)
                using TTask = TRemoteFunc<TFunc>;
                auto task = MakeHolder<TTask>(TPromiseFactory<true>::template CreateDevicePromise<TOutput>(DeviceId),
                                              std::forward<TFunc>(func));
                auto futureResult = task->GetResult();
                AddTask(std::move(task));
                return std::move(futureResult);
#else
                CB_ENSURE(false, "Remote device support is not enabled");
                return nullptr;
#endif
            }
        }

        void FreeStream(ui32 streamId) {
            CB_ENSURE(streamId != 0);
            UserFreeStreams.push_back(streamId);
        }

        ui32 DefaultStream() const {
            return 0;
        }

        void StreamSynchronize(ui32 streamHandle) {
            LaunchKernel(TSyncStreamKernel(), streamHandle);
            //ensure all jobs to stream we submitted to GPU (so streamSync was executed and blocked until all stream jobs are done)
            EmplaceTask<TWaitSubmitCommand>();
        }

        THolder<IDeviceFuture<ui64>> WaitComplete() {
            return LaunchFunc<TBlockingSyncDevice>(TBlockingSyncDevice());
        }

        void DeviceSynchronize() {
            StreamSynchronize(0u);
        }

        const TCudaDeviceProperties& GetDeviceProperties() const {
            return DeviceProperties;
        }

        TMemoryState GetMemoryState();

        ui32 RequestStream();
    };

    template <EPtrType Type>
    using TRawFreeMemory = typename TMemoryProviderImplTrait<Type>::TRawFreeMemory;

    template <EPtrType Type>
    struct TAllocateMemoryCommand {
        using TTask = TCudaMallocTask<Type>;
        using TTaskPtr = THolder<TTask>;

        static TTaskPtr Create(ui64 handle, ui64 size) {
            CB_ENSURE(size < 80ull * 1024 * 1024 * 1024, "Allocation of size " << size);
            return MakeHolder<TTask>(handle, size);
        }
    };
    template <>
    struct TCreateObjectCommandTrait<TRawFreeMemory<EPtrType::CudaDevice>>: public TAllocateMemoryCommand<EPtrType::CudaDevice> {
    };

    template <>
    struct TCreateObjectCommandTrait<TRawFreeMemory<EPtrType::Host>>: public TAllocateMemoryCommand<EPtrType::Host> {
    };

    template <>
    struct TCreateObjectCommandTrait<TRawFreeMemory<EPtrType::CudaHost>>: public TAllocateMemoryCommand<EPtrType::CudaHost> {
    };

}
