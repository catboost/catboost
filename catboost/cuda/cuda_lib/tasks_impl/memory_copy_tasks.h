#pragma once

#include "kernel_task.h"
#include "memory_copy_staged_operation.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/cuda/cuda_lib/stream_section_tasks_launcher.h>
#include <catboost/cuda/cuda_lib/single_device.h>

namespace NCudaLib {
    struct TLocalMasterMemcpyTask {
        NCudaLib::THandleRawPtr RawPtr;
        ui64 Size = 0;

        Y_SAVELOAD_DEFINE(RawPtr, Size);
    };

    enum EMemcpyTaskType {
        Read,
        Write
    };

    //local tasks are called on read side only.
    struct TIntraHostCopyTask {
        NCudaLib::THandleRawPtr SourcePtr;
        NCudaLib::THandleRawPtr DestPtr;
        ui64 Size = 0;

        Y_SAVELOAD_DEFINE(SourcePtr, DestPtr, Size);
    };

    class TMasterIntraHostMemcpy: public IGpuStatelessKernelTask<TMasterIntraHostMemcpy> {
    private:
        NCudaLib::THandleRawPtr DevicePtr;
        char* HostPtr;
        ui64 Size = 0;
        EMemcpyTaskType MemcpyType;
        NThreading::TPromise<TCudaEventPtr> DoneEventPromise;

    public:
        TMasterIntraHostMemcpy(NCudaLib::THandleRawPtr ptr,
                               char* hostPtr,
                               ui64 copySize,
                               EMemcpyTaskType type,
                               ui32 stream = 0)
            : IGpuStatelessKernelTask(stream)
            , DevicePtr(ptr)
            , HostPtr(hostPtr)
            , Size(copySize)
            , MemcpyType(type)
            , DoneEventPromise(NThreading::NewPromise<TCudaEventPtr>())
        {
        }

        THolder<IDeviceRequest> DoneEvent() {
            return MakeHolder<TLocalDeviceRequest>(DoneEventPromise.GetFuture());
        }

        void SubmitAsyncExecImpl(const TCudaStream& stream) {
            TCudaEventPtr eventPtr = CreateCudaEvent();
            const bool readFromHost = MemcpyType == EMemcpyTaskType::Write;
            char* src = (readFromHost ? HostPtr : DevicePtr.GetRawPtr());
            char* dst = (readFromHost ? DevicePtr.GetRawPtr() : HostPtr);
            CopyMemoryAsync(src, dst, Size, stream);
            eventPtr->Record(stream);
            DoneEventPromise.SetValue(std::move(eventPtr));
        }

        bool ReadyToSubmitNextImpl(const TCudaStream&) {
            return true;
        }
    };

#if defined(USE_MPI)

    struct TInterHostCopyTask {
        NCudaLib::TDeviceId RemoteDevice;
        NCudaLib::THandleRawPtr DataPtr;
        EMemcpyTaskType Type;
        ui64 Size = 0;
        int Tag = 0;

        Y_SAVELOAD_DEFINE(RemoteDevice, DataPtr, Type, Size, Tag);
    };

    inline constexpr ui64 GetDeviceToDeviceBlockSize() {
#if defined(WITHOUT_CUDA_AWARE_MPI)
        //512KB
        return 512 * 1024;
#else
        return 128 * 1024 * 1024;
#endif
    }

    inline constexpr ui64 GetMasterToDeviceBlockSize(EPtrType deviceType) {
        if (deviceType == EPtrType::CudaHost) {
            return 64 * 1024 * 1024;
        } else {
#if defined(WITHOUT_CUDA_AWARE_MPI)
            //512KB
            return 512 * 1024;
#else
            return 128 * 1024 * 1024;
#endif
        }
    }

    class TMasterInterHostMemcpy: public IGpuKernelTask {
    private:
        enum class EStage {
            WaitingDevice,
            Transfering,
            Finished
        };

        struct TContext: public NKernel::IKernelContext {
            TCudaEventPtr DeviceReadyEvent;
            THolder<IStagedTask> Task;
            TMpiManager* Manager;
            EStage Stage;
        };

    public:
        TMasterInterHostMemcpy(NCudaLib::THandleRawPtr destPtr,
                               ui64 size,
                               int tag,
                               EMemcpyTaskType type,
                               ui32 stream = 0)
            : IGpuKernelTask(stream)
            , DevicePtr(destPtr)
            , Size(size)
            , Tag(tag)
            , MemcpyType(type)
        {
        }

        TMasterInterHostMemcpy() {
        }

        THolder<NKernel::IKernelContext> PrepareExec(NKernelHost::IMemoryManager& memoryManager) const final {
            auto ctx = MakeHolder<TContext>();
            ctx->Manager = &GetMpiManager();
            ctx->DeviceReadyEvent = CreateCudaEvent();
            bool readFromHost = MemcpyType == EMemcpyTaskType::Write;
            const ui64 blockSize = GetMasterToDeviceBlockSize(DevicePtr.Type);

            const int masterRank = ctx->Manager->GetMasterId();

            if (readFromHost) {
                ctx->Task = BlockedRecvTask(DevicePtr.GetRawPtr(), Size, blockSize, masterRank, Tag, memoryManager, false);
            } else {
                ctx->Task = BlockedSendTask(DevicePtr.GetRawPtr(), Size, blockSize, masterRank, Tag, memoryManager, false);
            }
            return ctx;
        }

        void SubmitAsyncExec(const TCudaStream& stream,
                             NKernel::IKernelContext* ctx) final {
            auto& context = *reinterpret_cast<TContext*>(ctx);
            context.Stage = EStage::WaitingDevice;
            context.DeviceReadyEvent->Record(stream);
        }

        bool ReadyToSubmitNext(const TCudaStream& stream,
                               NKernel::IKernelContext* ctx) final {
            auto& context = *reinterpret_cast<TContext*>(ctx);
            switch (context.Stage) {
                case EStage::WaitingDevice: {
                    if (context.DeviceReadyEvent->IsComplete()) {
                        context.Stage = EStage::Transfering;
                    } else {
                        return false;
                    }
                }
                case EStage::Transfering: {
                    if (context.Task->Exec(stream)) {
                        context.Stage = EStage::Finished;
                    } else {
                        return false;
                    }
                }
                case EStage::Finished: {
                    return true;
                }
                default: {
                    CB_ENSURE(false, "Unknown stage");
                }
            }
            Y_UNREACHABLE();
        }

        Y_SAVELOAD_IMPL(DevicePtr, Size, Tag, MemcpyType);

    private:
        NCudaLib::THandleRawPtr DevicePtr;
        ui64 Size = 0;
        int Tag;
        EMemcpyTaskType MemcpyType;
    };

#endif

    struct TMemoryCopyContext {
        bool AreSubmitted = false;
#if defined(USE_MPI)
        TVector<IStagedTaskPtr> Tasks;
#endif
    };

    class TMemoryCopyTasks: public NKernelHost::TKernelWithContext<TMemoryCopyContext> {
    private:
#if defined(USE_MPI)
        TVector<TInterHostCopyTask> InterHostTasks; //mpi
        bool Compress = false;
#endif
        TVector<TIntraHostCopyTask> IntraHostTasks; //local

        friend class TDataCopier;

    protected:
        void SubmitTasks(const TCudaStream& stream,
                         TMemoryCopyContext&) {
            for (auto& task : IntraHostTasks) {
                const char* src = task.SourcePtr.GetRawPtr();
                char* dest = task.DestPtr.GetRawPtr();
                CopyMemoryAsync(src, dest, task.Size, stream);
            }
        }

    public:
        TMemoryCopyTasks() {
        }

        //for temp memory allocation.
        // All memory deallocation will be done before next sync command. Trade of memory for speed.
        THolder<TMemoryCopyContext> PrepareContext(NKernelHost::IMemoryManager& memoryManager) const {
            Y_UNUSED(memoryManager);
            auto ctx = MakeHolder<TMemoryCopyContext>();
#if defined(USE_MPI)
            //inter-host task
            for (auto& task : InterHostTasks) {
                Y_ASSERT(GetMpiManager().GetHostId() != task.RemoteDevice.HostId);
                const ui64 blockSize = GetDeviceToDeviceBlockSize();
                if (task.Type == EMemcpyTaskType::Read) {
                    char* dst = task.DataPtr.GetRawPtr();
                    ctx->Tasks.push_back(BlockedRecvTask(dst, task.Size, blockSize, task.RemoteDevice.HostId, task.Tag, memoryManager, Compress));
                } else {
                    Y_ASSERT(task.Type == EMemcpyTaskType::Write);
                    char* src = task.DataPtr.GetRawPtr();
                    ctx->Tasks.push_back(BlockedSendTask(src, task.Size, blockSize, task.RemoteDevice.HostId, task.Tag, memoryManager, Compress));
                }
            }
#endif
            return ctx;
        }

        bool Exec(const TCudaStream& stream,
                  TMemoryCopyContext* context) {
            if (!context->AreSubmitted) {
                SubmitTasks(stream, *context);
                context->AreSubmitted = true;
            }

#if defined(USE_MPI)
            ExecStagedTasks(stream, &context->Tasks);
            return context->Tasks.size() == 0;
#else
            return true;
#endif
        }

#if defined(USE_MPI)
        Y_SAVELOAD_DEFINE(InterHostTasks, IntraHostTasks, Compress); //mpi
#else
        Y_SAVELOAD_DEFINE(IntraHostTasks);
#endif
    };

    class TDataCopier {
    private:
        ui32 StreamId = 0;
#if defined(USE_MPI)
        bool CompressFlag = false;
#endif
        bool Submitted = false;
        TStreamSectionTaskLauncher SectionTaskLauncher;

        struct TTask {
            TCudaSingleDevice* SourceDevice;
            TCudaSingleDevice* DestDevice;
            THandleRawPtr SourcePtr;
            THandleRawPtr DestPtr;
            ui64 Size = 0;
        };

        TMap<TCudaSingleDevice*, TMemoryCopyTasks> Tasks;

    private:
        TMemoryCopyTasks& GetTasksRef(TCudaSingleDevice* device) {
            return Tasks[device];
        }

        //Streams are from TCudaSingleDevice
        static THolder<IDeviceRequest> AsyncReadLocal(TCudaSingleDevice* device, THandleRawPtr ptr, ui64 readSize, char* dst, ui32 stream) {
            auto task = MakeHolder<TMasterIntraHostMemcpy>(ptr, dst, readSize, EMemcpyTaskType::Read, stream);
            auto isDone = task->DoneEvent();
            device->AddTask(std::move(task));
            return isDone;
        }

        static THolder<IDeviceRequest> AsyncWriteLocal(TCudaSingleDevice* device, const char* src, THandleRawPtr ptr, ui64 writeSize, ui32 stream) {
            auto task = MakeHolder<TMasterIntraHostMemcpy>(ptr, const_cast<char*>(src), writeSize, EMemcpyTaskType::Write, stream);
            auto isDone = task->DoneEvent();
            device->AddTask(std::move(task));
            return isDone;
        }

        static THolder<IDeviceRequest> AsyncReadRemote(TCudaSingleDevice* device, THandleRawPtr ptr, ui64 readSize, char* dst, ui32 stream) {
#if defined(USE_MPI)
            auto& manager = GetMpiManager();
            int tag = manager.NextCommunicationTag();
            auto task = MakeHolder<TMasterInterHostMemcpy>(ptr, readSize, tag, EMemcpyTaskType::Read, stream);
            TVector<TMpiRequestPtr> requests;
            manager.ReadAsync(dst, readSize, GetMasterToDeviceBlockSize(ptr.Type), device->GetHostId(), tag, &requests);
            device->AddTask(std::move(task));
            return MakeHolder<TRemoteDeviceRequest>(std::move(requests));
#else
            Y_UNUSED(device);
            Y_UNUSED(ptr);
            Y_UNUSED(readSize);
            Y_UNUSED(dst);
            Y_UNUSED(stream);
            CB_ENSURE(false, "Error: Remote device support is unimplemented");
            return nullptr;
#endif
        }

        static THolder<IDeviceRequest> AsyncWriteRemote(TCudaSingleDevice* device, const char* src, THandleRawPtr ptr, ui64 writeSize, ui32 stream) {
#if defined(USE_MPI)
            auto& manager = GetMpiManager();
            int tag = manager.NextCommunicationTag();
            auto task = MakeHolder<TMasterInterHostMemcpy>(ptr, writeSize, tag, EMemcpyTaskType::Write, stream);
            TVector<TMpiRequestPtr> requests;
            manager.WriteAsync(src, writeSize, GetMasterToDeviceBlockSize(ptr.Type), device->GetHostId(), tag, &requests);
            device->AddTask(std::move(task));
            return MakeHolder<TRemoteDeviceRequest>(std::move(requests));
#else
            Y_UNUSED(device);
            Y_UNUSED(ptr);
            Y_UNUSED(writeSize);
            Y_UNUSED(src);
            Y_UNUSED(stream);
            CB_ENSURE(false, "Error: Remote device support is unimplemented");
            return nullptr;
#endif
        }

        void AddTask(const TTask& task) {
            auto& sourceDeviceTasks = GetTasksRef(task.SourceDevice);
            auto& destDeviceTasks = GetTasksRef(task.DestDevice);

            if (task.DestDevice->GetHostId() == task.SourceDevice->GetHostId()) {
                TIntraHostCopyTask copyTask;
                copyTask.SourcePtr = task.SourcePtr;
                copyTask.DestPtr = task.DestPtr;
                copyTask.Size = task.Size;

                destDeviceTasks.IntraHostTasks.push_back(std::move(copyTask));
            } else {
#if defined(USE_MPI)
                const int tag = GetMpiManager().NextCommunicationTag();
                {
                    TInterHostCopyTask copyTask;
                    copyTask.Size = task.Size;
                    copyTask.DataPtr = task.DestPtr;
                    copyTask.RemoteDevice = task.SourceDevice->GetDevice();
                    copyTask.Type = EMemcpyTaskType::Read;
                    copyTask.Tag = tag;
                    destDeviceTasks.InterHostTasks.push_back(std::move(copyTask));
                }
                {
                    TInterHostCopyTask copyTask;
                    copyTask.Size = task.Size;
                    copyTask.DataPtr = task.SourcePtr;
                    copyTask.RemoteDevice = task.DestDevice->GetDevice();
                    copyTask.Type = EMemcpyTaskType::Write;
                    copyTask.Tag = tag;
                    sourceDeviceTasks.InterHostTasks.push_back(std::move(copyTask));
                }
#else
                Y_UNUSED(sourceDeviceTasks);
                CB_ENSURE(false, "Remote device support was not enabled");
#endif
            }
        }

        //Streams are from CudaManager
        template <typename T, EPtrType DevicePtr>
        static THolder<IDeviceRequest> AsyncRead(const TCudaSingleDevice::TSingleBuffer<T, DevicePtr>& from, ui32 stream, ui64 fromOffset, std::remove_const_t<T>* to, ui64 readSize) {
            Y_ASSERT(readSize);
            const auto sizeInBytes = readSize * sizeof(T);
            THandleRawPtr readPtr(DevicePtr, from.MemoryHandle(), (from.GetOffset() + fromOffset) * sizeof(T));
            char* dst = ToCopyPointerType(to);
            TCudaSingleDevice* owner = from.Owner;
            ui32 deviceStream = GetCudaManager().StreamAt(stream, owner);

            if (owner->IsRemoteDevice()) {
                return AsyncReadRemote(owner, readPtr, sizeInBytes, dst, deviceStream);
            } else {
                return AsyncReadLocal(owner, readPtr, sizeInBytes, dst, deviceStream);
            }
        };

        template <typename T,
                  EPtrType DevicePtr>
        static THolder<IDeviceRequest> AsyncWrite(const T* from, TCudaSingleDevice::TSingleBuffer<T, DevicePtr>& buffer, ui32 stream, ui64 writeOffset, ui64 writeSize) {
            Y_ASSERT(writeSize);
            const ui64 sizeInBytes = writeSize * sizeof(T);
            THandleRawPtr writePtr(DevicePtr, buffer.MemoryHandle(), (buffer.GetOffset() + writeOffset) * sizeof(T));
            TCudaSingleDevice* device = buffer.Owner;
            ui32 deviceStream = GetCudaManager().StreamAt(stream, device);

            if (device->IsRemoteDevice()) {
                return AsyncWriteRemote(device, ToCopyPointerType(from), writePtr, sizeInBytes, deviceStream);
            } else {
                return AsyncWriteLocal(device, ToCopyPointerType(from), writePtr, sizeInBytes, deviceStream);
            }
        };

        template <class TCudaBuffer>
        friend class TCudaBufferReader;

        template <class TCudaBuffer>
        friend class TCudaBufferWriter;

    public:
        explicit TDataCopier(ui32 streamId = 0)
            : StreamId(streamId)
            , Submitted(true)
        {
        }

        ~TDataCopier() noexcept(false) {
            CB_ENSURE(Submitted, "Copy task wasn't submitted");
        }

        TDataCopier& SetCompressFlag(bool flag) {
#if defined(USE_MPI)
            CompressFlag = flag;
#else
            Y_UNUSED(flag);
#endif
            return *this;
        }

        template <typename T, class TC,
                  EPtrType FromType, EPtrType ToType>
        TDataCopier& AddAsyncMemoryCopyTask(const TCudaSingleDevice::TSingleBuffer<T, FromType>& from, ui64 readOffset,
                                            TCudaSingleDevice::TSingleBuffer<TC, ToType>& to, ui64 writeOffset, ui64 writeSize) {
            if (writeSize) {
                Submitted = false;
                static_assert(sizeof(T) == sizeof(TC), "Error types should have equal size");
                TTask task;
                task.SourceDevice = from.Owner;
                task.DestDevice = to.Owner;

                SectionTaskLauncher.Add(task.SourceDevice);
                SectionTaskLauncher.Add(task.DestDevice);

                //we need stream section only for memcpy withing one host, otherwise mpi communicatios guarantee consistency of pointers
                SectionTaskLauncher.Group(task.SourceDevice, task.DestDevice);

                task.SourcePtr = THandleRawPtr(FromType, from.MemoryHandle(),
                                               (from.GetOffset() + readOffset) * sizeof(T));

                task.DestPtr = THandleRawPtr(ToType, to.MemoryHandle(), (to.GetOffset() + writeOffset) * sizeof(T));
                task.Size = writeSize * sizeof(T);

                AddTask(task);
            }
            return *this;
        }

        void SubmitCopy() {
            if (Tasks.size()) {
                SectionTaskLauncher.LaunchTaskByDevicePtr([&](TCudaSingleDevice* device) {
#if defined(USE_MPI)
                    Tasks[device].Compress = CompressFlag;
#endif
                    return std::move(Tasks[device]);
                },
                                                          StreamId);
                Submitted = true;
            }
        }

        template <class T>
        static inline char* ToCopyPointerType(const T* ptr) {
            return reinterpret_cast<char*>(const_cast<std::remove_const_t<T>*>(ptr));
        }
    };
}

Y_DECLARE_PODTYPE(NCudaLib::TIntraHostCopyTask);
