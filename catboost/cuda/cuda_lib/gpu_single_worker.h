#pragma once

#include "cuda_base.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "task.h"
#include "helpers.h"

#include <catboost/cuda/cuda_lib/tasks_impl/kernel_task.h>
#include <catboost/cuda/cuda_lib/tasks_impl/memory_allocation.h>
#include <catboost/cuda/cuda_lib/tasks_impl/memory_state_func.h>
#include <catboost/cuda/cuda_lib/tasks_impl/request_stream_task.h>
#include <catboost/cuda/cuda_lib/tasks_queue/single_host_task_queue.h>

#include <library/cpp/threading/name_guard/name_guard.h>

#include <util/generic/map.h>
#include <util/generic/queue.h>
#include <util/generic/set.h>
#include <util/string/builder.h>
#include <util/system/event.h>
#include <util/system/yield.h>

namespace NCudaLib {
    class IExceptionCallback: public TThrRefBase {
    public:
        virtual void Call(const TString& errorMessage) = 0;
    };

    using TExceptionCallbackPtr = TIntrusivePtr<IExceptionCallback>;

    class TGpuOneDeviceWorker: public IWorkerStateProvider {
    private:
        class TComputationStream {
        private:
            struct TKernelTask {
                THolder<IGpuKernelTask> Task = nullptr;
                THolder<NKernel::IKernelContext> KernelContext = nullptr;
                TCudaStream* Stream;

                TKernelTask() = default;

                TKernelTask(THolder<IGpuKernelTask>&& task,
                            THolder<NKernel::IKernelContext>&& kernelContext,
                            TCudaStream* stream)
                    : Task(std::move(task))
                    , KernelContext(std::move(kernelContext))
                    , Stream(stream)
                {
                }

                bool IsRunning() const {
                    return !IsEmpty() && !Task->ReadyToSubmitNext(*Stream, KernelContext.Get());
                }

                bool IsReadyToSubmitNext() {
                    return !IsEmpty() && Task->ReadyToSubmitNext(*Stream, KernelContext.Get());
                }

                void Free() {
                    Task.Reset(nullptr);
                    KernelContext.Reset(nullptr);
                }

                bool IsEmpty() const {
                    return Task == nullptr;
                }
            };

        private:
            TCudaStream Stream;
            TQueue<TKernelTask> WaitingTasks;
            TKernelTask RunningTask;
            mutable bool IsActiveFlag;

        public:
            TComputationStream()
                : Stream(GetStreamsProvider().RequestStream())
                , IsActiveFlag(false)
            {
            }

            ~TComputationStream() noexcept(false) {
                CB_ENSURE(RunningTask.IsEmpty(), "Some tasks are not completed");
                CB_ENSURE(WaitingTasks.size() == 0, "Some tasks are waiting for processing");
            }

            void AddTask(THolder<IGpuKernelTask>&& task,
                         THolder<NKernel::IKernelContext>&& taskData) {
                IsActiveFlag = true;
                WaitingTasks.push(TKernelTask(std::move(task), std::move(taskData), &Stream));
                TryProceedTask();
            }

            bool HasTasks() const {
                return !RunningTask.IsEmpty() || WaitingTasks.size();
            }

            TCudaStream& GetStream() {
                return Stream;
            }

            bool IsActive() const {
                return IsActiveFlag;
            }

            void TryProceedTask() {
                if (RunningTask.IsRunning()) {
                    return;
                } else if (RunningTask.IsReadyToSubmitNext()) {
                    RunningTask.Free();
                }
                if (WaitingTasks.size()) {
                    using std::swap;
                    swap(RunningTask, WaitingTasks.front());
                    WaitingTasks.pop();
                    RunningTask.Task->SubmitAsyncExec(Stream, RunningTask.KernelContext.Get());
                }
            }

            void Synchronize() const {
                Stream.Synchronize();
                IsActiveFlag = false;
                CB_ENSURE(RunningTask.IsEmpty());
            }
        };

        class TTempMemoryManager: public NKernelHost::IMemoryManager {
        private:
            TGpuOneDeviceWorker* Owner;

        protected:
            TTempMemoryManager(TGpuOneDeviceWorker* owner)
                : Owner(owner)
            {
            }

            ui64 AllocateImpl(EPtrType ptrType, ui64 size) final {
                auto handle = GetHandleStorage().GetHandle(1)[0];
                Owner->AllocateTempMemory(handle, ptrType, size);
                return handle;
            }

        public:
            friend class TGpuOneDeviceWorker;
        };

    private:
        //kernel tasks are requests from master
        using TGpuTaskPtr = THolder<ICommand>;
        using TTaskQueue = TSingleHostTaskQueue;
        using TComputationStreamPtr = THolder<TComputationStream>;

        TTaskQueue InputTaskQueue;
        int LocalDeviceId;
        TCudaDeviceProperties DeviceProperties;
        TTempMemoryManager TempMemoryManager;

        //objects will be deleted lazily.
        TVector<THolder<IFreeMemoryTask>> ObjectsToFree;
        TVector<THolder<IFreeMemoryTask>> TempMemoryAllocatedObjects;

        //Streams
        TVector<TComputationStreamPtr> Streams;
        TVector<ui32> FreeStreams;

        using THostMemoryProvider = TMemoryProviderImplTrait<EPtrType::CudaHost>::TMemoryProvider;
        using THostMemoryProviderPtr = THolder<THostMemoryProvider>;

        using TDeviceMemoryProvider = TMemoryProviderImplTrait<EPtrType::CudaDevice>::TMemoryProvider;
        using TDeviceMemoryProviderPtr = THolder<TDeviceMemoryProvider>;

        THostMemoryProviderPtr HostMemoryProvider;
        TDeviceMemoryProviderPtr DeviceMemoryProvider;

        TVector<TExceptionCallbackPtr> ErrorCallbacks;
        TAdaptiveLock CallbackLock;

        std::unique_ptr<std::thread> WorkingThread;
        TAtomic Stopped = 1;

    private:
        void WaitAllTaskToSubmit() {
            ui32 i = 0;
            while (CheckRunningTasks()) {
                ++i;
                if (i % 1000 == 0) {
                    SchedYield();
                }
            }
        }

        void SyncActiveStreams(bool skipDefault = false) {
            for (ui32 i = (ui64)skipDefault; i < Streams.size(); ++i) {
                if (Streams[i]->IsActive()) {
                    SyncStream(i);
                }
            }
        }

        bool CheckRunningTasks() const {
            bool hasRunning = false;
            for (ui32 i = 0; i < Streams.size(); ++i) {
                if (Streams[i]->IsActive()) {
                    if (Streams[i]->HasTasks()) {
                        hasRunning = true;
                        Streams[i]->TryProceedTask();
                    }
                }
            }
            return hasRunning;
        }

        void DeleteObjects() {
            for (auto& freeTask : ObjectsToFree) {
                freeTask->Exec();
            }
            ObjectsToFree.resize(0);
        }

        void DefragmentMemory() {
            DeviceMemoryProvider->TryDefragment();
            HostMemoryProvider->TryDefragment();
        }

        void SyncStream(ui32 id) {
            Streams[id]->Synchronize();
        }

        void AllocateTempMemory(ui64 handle, EPtrType ptrType, ui64 size);

        void RunErrorCallbacks(const TString& errorMessage) const {
            TGuard<TAdaptiveLock> guard(CallbackLock);
            for (auto& callback : ErrorCallbacks) {
                callback->Call(errorMessage);
            }
        }

    public:
        bool NeedSyncForMalloc(const EPtrType ptrType, ui64 size) {
            switch (ptrType) {
                case EPtrType::CudaHost: {
                    return HostMemoryProvider->NeedSyncForAllocation<char>(size);
                }
                case EPtrType::CudaDevice: {
                    return DeviceMemoryProvider->NeedSyncForAllocation<char>(size);
                }
                case EPtrType::Host: {
                    return false;
                }
                default: {
                    CB_ENSURE(false, "Unknown pointer type");
                }
            }
        }

        void AllocateMemory(const IAllocateMemoryTask& mallocTask) {
            const auto handle = mallocTask.GetHandle();
            Y_ASSERT(!TObjectByHandleStorage::IsNullptr(handle));

            //ensure, that no one will use memory is we need some ptr-rebuild.
            if (NeedSyncForMalloc(mallocTask.GetPtrType(), mallocTask.GetSize())) {
                WaitSubmitAndSync();
            }

            auto& storage = GetHandleStorage();
            switch (mallocTask.GetPtrType()) {
                case EPtrType::CudaHost: {
                    storage.SetObjectPtrByHandle(handle, HostMemoryProvider->Create(mallocTask.GetSize()));
                    break;
                }
                case EPtrType::CudaDevice: {
                    storage.SetObjectPtrByHandle(handle, DeviceMemoryProvider->Create(mallocTask.GetSize()));
                    break;
                }
                case EPtrType::Host: {
                    storage.SetObjectPtrByHandle(handle, new char[mallocTask.GetSize()]);
                    break;
                }
            }
        }

    private:
        inline void WaitSubmitAndSync(bool skipDefault = false) {
            WaitAllTaskToSubmit();
            SyncActiveStreams(skipDefault);
            DeleteObjects();
        }

        void CreateNewComputationStream() {
            Streams.push_back(MakeHolder<TComputationStream>());
        }

        inline ui32 RequestStreamImpl() {
            if (FreeStreams.size() == 0) {
                FreeStreams.push_back(Streams.size());
                CreateNewComputationStream();
            }
            ui32 id = FreeStreams.back();
            FreeStreams.pop_back();
            return id;
        }

        void Reset(const TResetCommand& initTask) {
            ui64 gpuMemorySize = 0;

            DeviceMemoryProvider.Reset();
            HostMemoryProvider.Reset();

            if (initTask.GpuMemoryPart) {
                ui64 free = 0;
                ui64 total = 0;
                CUDA_SAFE_CALL(cudaMemGetInfo(&free, &total));
                if (free * 1.0 / DeviceProperties.GetDeviceMemory() < 0.75) {
                    CATBOOST_WARNING_LOG << "Warning: less than 75% GPU memory available for training. Free: " << free * 1.0 / 1024 / 1024 << " Total: " << total * 1.0 / 1024 / 1024 << Endl;
                }
                gpuMemorySize = (ui64)(free * initTask.GpuMemoryPart);
            }

            DeviceMemoryProvider = gpuMemorySize ? MakeHolder<TDeviceMemoryProvider>(gpuMemorySize) : nullptr;
            HostMemoryProvider = initTask.PinnedMemorySize ? MakeHolder<THostMemoryProvider>(initTask.PinnedMemorySize) : nullptr;
        }

        bool RunIteration();

    public:
        TGpuOneDeviceWorker(int gpuDevId,
                            TExceptionCallbackPtr callback = nullptr)
            : LocalDeviceId(gpuDevId)
            , DeviceProperties(NCudaHelpers::GetDeviceProps(gpuDevId))
            , TempMemoryManager(this)
        {
            if (callback) {
                RegisterErrorCallback(callback);
            }
            WorkingThread.reset(new std::thread([this]() -> void {
                Y_THREAD_NAME_GUARD(TStringBuilder() << "GpuWorker" << LocalDeviceId);
                this->Run();
            }));
        }

        ~TGpuOneDeviceWorker() noexcept(false) {
            CB_ENSURE(AtomicGet(Stopped), "Worker is not stopped");
        }

        TTaskQueue& GetTaskQueue() {
            return InputTaskQueue;
        }

        void Run();

        bool IsRunning() const {
            return !AtomicGet(Stopped);
        }

        TMemoryState GetMemoryState() const final {
            CB_ENSURE(!AtomicGet(Stopped));
            CB_ENSURE(HostMemoryProvider);
            CB_ENSURE(DeviceMemoryProvider);
            TMemoryState result;
            result.RequestedPinnedRam = HostMemoryProvider->GetRequestedRamSize();
            result.FreePinnedRam = HostMemoryProvider->GetFreeMemorySize();

            result.RequestedGpuRam = DeviceMemoryProvider->GetRequestedRamSize();
            result.FreeGpuRam = DeviceMemoryProvider->GetFreeMemorySize();
            return result;
        }

        void RegisterErrorCallback(TExceptionCallbackPtr callback) {
            TGuard<TAdaptiveLock> guard(CallbackLock);
            ErrorCallbacks.push_back(callback);
        }

        void Join() {
            if (WorkingThread && WorkingThread->joinable()) {
                WorkingThread->join();
            }
        }
    };
}
