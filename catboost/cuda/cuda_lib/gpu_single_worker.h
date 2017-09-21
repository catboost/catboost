#pragma once

#include "cuda_base.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "task.h"
#include "helpers.h"

#include <library/threading/chunk_queue/queue.h>
#include <util/ysafeptr.h>
#include <util/generic/map.h>
#include <util/generic/queue.h>
#include <util/system/yield.h>
#include <util/generic/set.h>
#include <util/system/event.h>

namespace NCudaLib {
    class IExceptionCallback: public TThrRefBase {
    public:
        virtual void Call(const TString& errorMessage) = 0;
    };

    using TExceptionCallbackPtr = TIntrusivePtr<IExceptionCallback>;

    class TGpuOneDeviceWorker {
    private:
        class TComputationStream {
        private:
            struct TKernelTask {
                THolder<IGpuKernelTask> Task = nullptr;
                THolder<NKernel::IKernelContext> KernelContext = nullptr;
                TCudaStream* Stream;

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
            yqueue<TKernelTask> WaitingTasks;
            TKernelTask RunningTask;
            mutable bool IsActiveFlag;

        public:
            ~TComputationStream() {
                CB_ENSURE(RunningTask.IsEmpty());
                CB_ENSURE(WaitingTasks.size() == 0);
            }
            void AddTask(THolder<IGpuKernelTask>&& task,
                         THolder<NKernel::IKernelContext>&& taskData) {
                IsActiveFlag = true;
                WaitingTasks.push({std::move(task), std::move(taskData), &Stream});
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

            template <EPtrType PtrType>
            class TRemoveObjectCommand: public IFreeMemoryTask {
            private:
                using TPtr = THolder<typename TMemoryProviderImplTrait<PtrType>::TRawFreeMemory>;
                TPtr MemoryPtr;

            public:
                TRemoveObjectCommand(TPtr&& ptr)
                    : MemoryPtr(std::move(ptr))
                {
                }

                void* GetPtr() {
                    return MemoryPtr->Get();
                }

                void Exec() override {
                    MemoryPtr.Reset(nullptr);
                }
            };

        protected:
            TTempMemoryManager(TGpuOneDeviceWorker* owner)
                : Owner(owner)
            {
            }

            void* AllocateImpl(EPtrType ptrType, ui64 size) override {
                switch (ptrType) {
                    case CudaHost: {
                        auto rawPtr = Owner->HostMemoryProvider->Create(size);
                        using TType = typename std::remove_pointer<decltype(rawPtr)>::type;
                        auto cmd = new TRemoveObjectCommand<CudaHost>(THolder<TType>(rawPtr));
                        void* ptr = cmd->GetPtr();
                        Owner->AddFreeMemoryTask(cmd);
                        return ptr;
                    }
                    case CudaDevice: {
                        auto rawPtr = Owner->DeviceMemoryProvider->Create(size);
                        using TType = typename std::remove_pointer<decltype(rawPtr)>::type;
                        auto cmd = new TRemoveObjectCommand<CudaDevice>(THolder<TType>(rawPtr));
                        void* ptr = cmd->GetPtr();
                        Owner->AddFreeMemoryTask(cmd);
                        return ptr;
                    }
                    case Host:
                    default: {
                        ythrow yexception() << "Unsupported operation type";
                    }
                }
                CB_ENSURE("Not here");
            }

        public:
            friend class TGpuOneDeviceWorker;
        };

    private:
        //kernel tasks are requests from master
        using TGpuTaskPtr = THolder<IGpuCommand>;
        using TTaskQueue = typename NThreading::TOneOneQueue<TGpuTaskPtr>;
        using TComputationStreamPtr = THolder<TComputationStream>;

        int LocalDeviceId;
        TCudaDeviceProperties DeviceProperties;
        TTempMemoryManager TempMemoryManager;

        TTaskQueue InputTaskQueue;

        //objects will be deleted lazily.
        yvector<THolder<IFreeMemoryTask>> ObjectsToFree;

        //Streams
        yvector<TComputationStreamPtr> Streams;
        yvector<ui64> FreeStreams;

        using THostMemoryProvider = TMemoryProviderImplTrait<CudaHost>::TMemoryProvider;
        using THostMemoryProviderPtr = THolder<THostMemoryProvider>;
        using TDeviceMemoryProvider = TMemoryProviderImplTrait<CudaDevice>::TMemoryProvider;
        using TDeviceMemoryProviderPtr = THolder<TDeviceMemoryProvider>;
        THostMemoryProviderPtr HostMemoryProvider;
        TDeviceMemoryProviderPtr DeviceMemoryProvider;

        yvector<TExceptionCallbackPtr> ErrorCallbacks;
        TAdaptiveLock CallbackLock;

        std::unique_ptr<std::thread> WorkingThread;
        std::promise<void> StopPromise;
        std::future<void> Stop;

        TManualEvent JobsEvent;

        void WaitAllTaskToSubmit() {
            while (CheckRunningTasks()) {
                SchedYield();
            }
        }

        void SyncActiveStreams(bool skipDefault = false) {
            for (ui64 i = (ui64)skipDefault; i < Streams.size(); ++i) {
                if (Streams[i]->IsActive()) {
                    SyncStream(i);
                }
            }
        }

        bool CheckRunningTasks() const {
            bool hasRunning = false;
            for (ui64 i = 0; i < Streams.size(); ++i) {
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
            for (auto& task : ObjectsToFree) {
                task->Exec();
            }
            ObjectsToFree.resize(0);
        }

        void SyncStream(ui64 id) {
            Streams[id]->Synchronize();
        }

        void AddFreeMemoryTask(THolder<IFreeMemoryTask>&& freeMemoryTask) {
            ObjectsToFree.push_back(std::move(freeMemoryTask));
        }

        void RunErrorCallbacks(const TString& errorMessage) const {
            TGuard<TAdaptiveLock> guard(CallbackLock);
            for (auto& callback : ErrorCallbacks) {
                callback->Call(errorMessage);
            }
        }

    public:
        bool NeedSyncForMalloc(const EPtrType ptrType, ui64 size) {
            switch (ptrType) {
                case CudaHost: {
                    return HostMemoryProvider->NeedSyncForAllocation<char>(size);
                }
                case CudaDevice: {
                    return DeviceMemoryProvider->NeedSyncForAllocation<char>(size);
                }
                case Host: {
                    return false;
                }
            }
            CB_ENSURE(false);
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
                case CudaHost: {
                    storage.SetObjectPtrByHandle(handle, HostMemoryProvider->Create(mallocTask.GetSize()));
                    break;
                }
                case CudaDevice: {
                    storage.SetObjectPtrByHandle(handle, DeviceMemoryProvider->Create(mallocTask.GetSize()));
                    break;
                }
                case Host: {
                    storage.SetObjectPtrByHandle(handle, new char[mallocTask.GetSize()]);
                }
            }
        }

    private:
        inline void WaitSubmitAndSync() {
            WaitAllTaskToSubmit();
            SyncActiveStreams();
        }

        void CreateNewComputationStream() {
            Streams.push_back(new TComputationStream());
        }

        inline ui64 RequestStreamImpl() {
            if (FreeStreams.size() == 0) {
                FreeStreams.push_back(Streams.size());
                CreateNewComputationStream();
            }
            ui64 id = FreeStreams.back();
            FreeStreams.pop_back();
            return id;
        }

        void RunIteration() {
            try {
                const bool hasRunning = CheckRunningTasks();
                const bool isEmpty = InputTaskQueue.IsEmpty();
                if (!hasRunning && isEmpty) {
                    JobsEvent.Reset();

                    const ui32 waitIters = 100000;
                    for (ui32 iter = 0; iter < waitIters; ++iter) {
                        SchedYield();
                        if (!InputTaskQueue.IsEmpty()) {
                            break;
                        }
                    }
                    if (InputTaskQueue.IsEmpty()) {
                        JobsEvent.WaitD(TInstant::Seconds(1));
                    }
                } else if (!isEmpty) {
                    THolder<IGpuCommand> task;
                    bool done = InputTaskQueue.Dequeue(task);
                    CB_ENSURE(done, "Error: dequeue failed");

                    switch (task->GetCommandType()) {
                        //could be run async
                        case EGpuHostCommandType::StreamKernel: {
                            IGpuKernelTask* kernelTask = dynamic_cast<IGpuKernelTask*>(task.Get());
                            const ui32 streamId = kernelTask->GetStreamId();
                            if (streamId == 0) {
                                WaitAllTaskToSubmit();
                                SyncActiveStreams(true);
                            }
                            auto& stream = *Streams[streamId];
                            auto data = kernelTask->PrepareExec(TempMemoryManager);
                            task.Release();
                            stream.AddTask(THolder<IGpuKernelTask>(kernelTask), std::move(data));
                            break;
                        }
                        //synchronized on memory defragmentation
                        case EGpuHostCommandType::MemoryAllocation: {
                            IAllocateMemoryTask* memoryTask = dynamic_cast<IAllocateMemoryTask*>(task.Get());
                            AllocateMemory(*memoryTask);
                            break;
                        }
                        case EGpuHostCommandType::WaitSubmit: {
                            WaitAllTaskToSubmit();
                            break;
                        }
                        //synchronized always
                        case EGpuHostCommandType::HostTask: {
                            WaitSubmitAndSync();
                            DeleteObjects();
                            auto taskPtr = dynamic_cast<IHostTask*>(task.Get());
                            taskPtr->Exec();
                            break;
                        }
                        case EGpuHostCommandType::MemoryDealocation: {
                            WaitSubmitAndSync();
                            IFreeMemoryTask* freeMemoryTask = dynamic_cast<IFreeMemoryTask*>(task.Get());
                            task.Release();
                            ObjectsToFree.push_back(THolder<IFreeMemoryTask>(freeMemoryTask));
                            DeleteObjects();
                            break;
                        }
                        case EGpuHostCommandType::RequestStream: {
                            const ui32 streamId = RequestStreamImpl();
                            dynamic_cast<TRequestStreamCommand*>(task.Get())->SetStreamId(streamId);
                            break;
                        }
                        case EGpuHostCommandType::FreeStream: {
                            FreeStreams.push_back(dynamic_cast<TFreeStreamCommand*>(task.Get())->GetStream());
                            break;
                        }
                        default: {
                            ythrow yexception() << "Unknown command type";
                        }
                    }
                }
                CheckLastError();
            } catch (...) {
                RunErrorCallbacks(CurrentExceptionMessage());
            }
        }

    public:
        TGpuOneDeviceWorker(int gpuDevId,
                            ui64 gpuMemoryLimit,
                            ui64 pinnedMemoryLimit)
            : LocalDeviceId(gpuDevId)
            , DeviceProperties(NCudaHelpers::GetDeviceProps(gpuDevId))
            , TempMemoryManager(this)
            , Stop(StopPromise.get_future())
        {
            WorkingThread.reset(new std::thread([=]() -> void {
                this->Run(gpuMemoryLimit, pinnedMemoryLimit);
            }));
        }

        ~TGpuOneDeviceWorker() throw (yexception) {
            StopPromise.set_value();
            WorkingThread->join();
            CB_ENSURE(InputTaskQueue.IsEmpty());
        }

        void Run(ui64 gpuMemoryLimit, ui64 pinnedMemoryLimit) {
            SetDevice(LocalDeviceId);
            DeviceMemoryProvider = MakeHolder<TDeviceMemoryProvider>(gpuMemoryLimit);
            HostMemoryProvider = MakeHolder<THostMemoryProvider>(pinnedMemoryLimit);
            CreateNewComputationStream();
            SetDefaultStream(Streams[0]->GetStream());

            ui64 Iteration = 0;
            while (true) {
                ++Iteration;
                RunIteration();
                if (::NHelpers::IsFutureReady(Stop)) {
                    break;
                }
            }
        }

        void FreeStream(ui32 stream) {
            SyncStream(stream);
            AddTask(THolder<IGpuCommand>(new TFreeStreamCommand(stream)));
        }

        ui64 RequestStream() {
            auto cmd = new TRequestStreamCommand();
            auto result = cmd->GetStreamId();
            AddTask(THolder<IGpuCommand>(cmd));
            result.wait();
            return result.get();
        }

        void RegisterErrorCallback(TExceptionCallbackPtr callback) {
            TGuard<TAdaptiveLock> guard(CallbackLock);
            ErrorCallbacks.push_back(callback);
        }

        yvector<ui64> RequestHandles(ui64 count) {
            return GetHandleStorage().GetHandle(count);
        }

        int GetDeviceId() const {
            return LocalDeviceId;
        }

        ui64 GetGpuRamSize() const {
            return DeviceMemoryProvider ? DeviceMemoryProvider->GpuRamSize() : 0;
        }

        ui64 GetFreeMemory() const {
            return DeviceMemoryProvider ? DeviceMemoryProvider->GetFreeMemorySize() : 0;
        }

        TCudaDeviceProperties GetDeviceProperties() const {
            return DeviceProperties;
        }

        void AddTask(THolder<IGpuCommand>&& task) {
            InputTaskQueue.Enqueue(std::move(task));
            JobsEvent.Signal();
        }
    };
}
