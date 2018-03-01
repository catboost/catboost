#pragma once

#include <util/system/types.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/cuda/cuda_lib/remote_objects.h>
#include <catboost/cuda/cuda_lib/cuda_events_provider.h>

namespace NCudaLib {
#if defined(USE_MPI)

    class IStagedTask {
    public:
        virtual ~IStagedTask() {
        }

        virtual bool Exec(const TCudaStream& stream) = 0;
    };

    inline constexpr double GetCompressBlockReserveFactor() {
        return 1.5;
    }

    using IStagedTaskPtr = THolder<IStagedTask>;
    template <class T,
              class TOperator>
    class TThroughHostStagedRecvTask: public IStagedTask {
    private:
        T* Dst = nullptr;
        ui64 Size = 0;

        int RemoteHost = -1;
        int Tag = -1;
        TOperator Operator;
        bool UseCompression = false;

        struct {
            NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost> ReadBuffer;
            NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost> OperatorBuffer;
            NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost> DecompressBuffer;

            NCudaLib::TMpiManager* Manager = nullptr;

            NCudaLib::TCudaEventPtr OperatorDoneEvent;
            NCudaLib::TMpiRequestPtr ReadDoneEvent;

            bool IsRemoteCopyComplete = true;
            bool IsOperatorComplete = true;

            ui64 BlockSize = 0;
            ui64 ReservedSize = 0;
            ui64 WorkingBufferSize = 0;

            ui64 Offset = 0;
            bool IsFirst = true;
        } State;

    public:
        TThroughHostStagedRecvTask(T* dst,
                                   ui64 size,
                                   TOperator op,
                                   int remoteHost,
                                   int tag,
                                   NKernelHost::IMemoryManager& memoryManager,
                                   bool useCompression)
            : Dst(dst)
            , Size(size)
            , RemoteHost(remoteHost)
            , Tag(tag)
            , Operator(op)
            , UseCompression(useCompression)
        {
            State.OperatorDoneEvent = NCudaLib::CudaEventProvider().Create();
            State.Manager = &GetMpiManager();
            const ui64 blockSize = op.GetBlockSize(size);
            Y_ASSERT(blockSize);
            State.BlockSize = blockSize;

            const bool needCompression = UseCompression && (sizeof(T) * Size > State.Manager->GetMinCompressSize()) && (sizeof(T) * blockSize > State.Manager->GetMinCompressSize());

            State.ReservedSize = (ui64)(needCompression
                                            ? blockSize * GetCompressBlockReserveFactor()
                                            : blockSize);
            State.ReadBuffer = memoryManager.Allocate<T, EPtrType::CudaHost>(State.ReservedSize);
            State.OperatorBuffer = memoryManager.Allocate<T, EPtrType::CudaHost>(State.ReservedSize);
            if (needCompression) {
                State.DecompressBuffer = memoryManager.Allocate<T, EPtrType::CudaHost>(State.ReservedSize);
            }
        }

        bool Exec(const TCudaStream& stream) final {
            if (State.Offset >= Size) {
                return true;
            }

            if (State.IsFirst) {
                State.IsOperatorComplete = true;
                State.IsRemoteCopyComplete = false;
                const ui64 size = Min<ui64>(Size, State.BlockSize);
                const ui32 receiveSize = UseCompression ? sizeof(T) * State.ReservedSize : sizeof(T) * size;

                State.ReadDoneEvent = State.Manager->ReadAsync((char*)State.ReadBuffer.Get(),
                                                               receiveSize,
                                                               RemoteHost,
                                                               Tag);
                State.WorkingBufferSize = size;
                State.IsFirst = false;
            }

            if (!State.IsRemoteCopyComplete) {
                State.IsRemoteCopyComplete = State.ReadDoneEvent->IsComplete();
            }
            if (!State.IsOperatorComplete) {
                State.IsOperatorComplete = State.OperatorDoneEvent->IsComplete();
            }

            if (State.IsRemoteCopyComplete && State.IsOperatorComplete) {
                if (UseCompression && (sizeof(T) * State.WorkingBufferSize > State.Manager->GetMinCompressSize())) {
                    auto codec = State.Manager->GetCodec();
                    TStringBuf data((char*)State.ReadBuffer.Get(), State.ReadDoneEvent->ReceivedBytes());
                    const ui64 size = codec->Decompress(data, (void*)State.DecompressBuffer.Get());
                    CB_ENSURE(size == sizeof(T) * State.WorkingBufferSize, "error: decompress size should equal to buffer size");
                    using std::swap;
                    swap(State.ReadBuffer, State.DecompressBuffer);
                }
                {
                    using std::swap;
                    swap(State.ReadBuffer, State.OperatorBuffer);
                }

                const ui64 readOffset = State.Offset + State.BlockSize;

                if (readOffset < Size) {
                    const ui64 size = Min<ui64>(Size - readOffset, State.BlockSize);
                    const ui32 receiveSize = UseCompression ? sizeof(T) * State.ReservedSize : sizeof(T) * size;
                    State.ReadDoneEvent = State.Manager->ReadAsync((char*)State.ReadBuffer.Get(),
                                                                   receiveSize,
                                                                   RemoteHost,
                                                                   Tag);
                    State.WorkingBufferSize = size;
                }

                {
                    const ui64 size = Min<ui64>(Size - State.Offset, State.BlockSize);
                    Operator(Dst + State.Offset, State.OperatorBuffer.Get(), size, stream);
                }

                State.Offset += State.BlockSize;

                State.IsRemoteCopyComplete = false;
                State.IsOperatorComplete = false;

                if (State.Offset < Size) {
                    State.OperatorDoneEvent->Record(stream);
                    return false;
                } else {
                    return true;
                }
            }
            return State.Offset >= Size;
        }
    };

    template <class T,
              class TOperator>
    class TThroughHostStagedSendTask: public IStagedTask {
    private:
        const T* Src = nullptr;
        ui64 Size = 0;
        int RemoteHost = -1;
        int Tag = -1;
        TOperator Operator;
        bool UseCompression = false;

        struct {
            NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost> BufferToSend;
            NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost> OperatorBuffer;
            NCudaLib::THandleBasedMemoryPointer<T, EPtrType::CudaHost> CompressBuffer;

            NCudaLib::TMpiManager* Manager = nullptr;

            NCudaLib::TCudaEventPtr OperatorDoneEvent;
            ui64 OperatorWorkingBufferSize = 0;
            NCudaLib::TMpiRequestPtr WriteDoneEvent;

            bool IsRemoteCopyComplete = true;
            bool IsOperatorComplete = true;

            ui64 Offset = 0;
            bool IsFirst = true;
            ui64 BlockSize = 0;
        } State;

    public:
        TThroughHostStagedSendTask(const T* src, ui64 size,
                                   TOperator op,
                                   int remoteHost, int tag,
                                   NKernelHost::IMemoryManager& memoryManager,
                                   bool useCompression)
            : Src(src)
            , Size(size)
            , RemoteHost(remoteHost)
            , Tag(tag)
            , Operator(op)
            , UseCompression(useCompression)
        {
            State.OperatorDoneEvent = NCudaLib::CudaEventProvider().Create();
            State.Manager = &GetMpiManager();

            const ui64 blockSize = op.GetBlockSize(size);
            State.BlockSize = blockSize;
            Y_ASSERT(blockSize);
            Y_ASSERT(size);
            const bool needCompression = UseCompression && (sizeof(T) * Size > State.Manager->GetMinCompressSize()) && (sizeof(T) * blockSize > State.Manager->GetMinCompressSize());

            const ui64 reserveSize = (ui64)(needCompression
                                                ? blockSize * GetCompressBlockReserveFactor()
                                                : blockSize);
            State.BufferToSend = memoryManager.Allocate<T, EPtrType::CudaHost>(reserveSize);
            State.OperatorBuffer = memoryManager.Allocate<T, EPtrType::CudaHost>(reserveSize);
            if (needCompression) {
                State.CompressBuffer = memoryManager.Allocate<T, EPtrType::CudaHost>(reserveSize);
            }
        }

        bool Exec(const TCudaStream& stream) final {
            if (State.Offset >= Size) {
                Y_ASSERT(State.WriteDoneEvent);
                return State.WriteDoneEvent->IsComplete();
            }

            if (State.IsFirst) {
                State.IsOperatorComplete = false;
                State.IsRemoteCopyComplete = true;
                const ui64 size = Min<ui64>(Size, State.BlockSize);
                Operator(State.OperatorBuffer.Get(), Src, size, stream);
                State.OperatorWorkingBufferSize = size;
                State.OperatorDoneEvent->Record(stream);
                State.IsFirst = false;
            }

            if (!State.IsRemoteCopyComplete) {
                State.IsRemoteCopyComplete = State.WriteDoneEvent->IsComplete();
            }
            if (!State.IsOperatorComplete) {
                State.IsOperatorComplete = State.OperatorDoneEvent->IsComplete();
            }

            if (State.IsRemoteCopyComplete && State.IsOperatorComplete) {
                ui64 writeSize = Min<ui64>(Size - State.Offset, State.BlockSize) * sizeof(T);
                Y_ASSERT(writeSize);

                if (UseCompression && (sizeof(T) * State.OperatorWorkingBufferSize > State.Manager->GetMinCompressSize())) {
                    TStringBuf data((char*)State.OperatorBuffer.Get(),
                                    sizeof(T) * State.OperatorWorkingBufferSize);
                    writeSize = State.Manager->GetCodec()->Compress(data, (void*)State.CompressBuffer.Get());
                    CB_ENSURE(writeSize < sizeof(T) * State.BlockSize * GetCompressBlockReserveFactor());
                    using std::swap;
                    swap(State.OperatorBuffer, State.CompressBuffer);
                }
                {
                    using std::swap;
                    swap(State.BufferToSend, State.OperatorBuffer);
                }
                const ui64 deviceReadOffset = State.Offset + State.BlockSize;

                if (deviceReadOffset < Size) {
                    const ui64 size = Min<ui64>(Size - deviceReadOffset, State.BlockSize);
                    Operator(State.OperatorBuffer.Get(), Src + deviceReadOffset, size, stream);
                    State.OperatorDoneEvent->Record(stream);
                    State.IsOperatorComplete = false;
                    State.OperatorWorkingBufferSize = size;
                }

                {
                    State.WriteDoneEvent = State.Manager->WriteAsync((const char*)State.BufferToSend.Get(), writeSize, RemoteHost, Tag);
                    State.IsRemoteCopyComplete = false;
                }
                State.Offset += State.BlockSize;
            }
            if (State.Offset >= Size) {
                return State.WriteDoneEvent->IsComplete();
            } else {
                return false;
            }
        }
    };

    template <class T>
    struct TMemcpyOperator {
        ui64 BlockSize = 0;

        void operator()(T* dst, const T* src, ui64 size, const TCudaStream& stream) {
            NCudaLib::CopyMemoryAsync(src, dst, size, stream);
        }

        ui64 GetBlockSize(ui64 size) {
            if (BlockSize == 0) {
                return Min<ui64>(size, 8 * 1024 * 1024 / sizeof(T));
            } else {
                return BlockSize;
            }
        }
    };

    //if we have cuda-aware support, than we will delegate everything
    template <class T, bool IsRecv>
    class TMpiDelegatingStageTask: public IStagedTask {
    private:
        using TOp = TMemcpyOperator<T>;
        T* Buffer = nullptr;
        ui64 Size = 0;
        int RemoteHost = -1;
        int Tag = -1;

        struct {
            TVector<TMpiRequestPtr> Requests;
            ui64 BlockSize = 0;
            bool AreRequestsCreated = false;
        } State;

    public:
        TMpiDelegatingStageTask(T* data, ui64 size,
                                TMemcpyOperator<std::remove_const_t<T>> op,
                                int remoteHost, int tag,
                                NKernelHost::IMemoryManager&)
            : Buffer(data)
            , Size(size)
            , RemoteHost(remoteHost)
            , Tag(tag)
        {
            State.BlockSize = op.GetBlockSize(Size);
        }

        //assumes stream already synchronized
        bool Exec(const TCudaStream&) final {
            if (!State.AreRequestsCreated) {
                auto& manager = GetMpiManager();
                const auto maxBlockSize = ((ui64)1) << 31;
                CB_ENSURE(State.BlockSize < maxBlockSize);

                for (ui64 offset = 0; offset < Size; offset += State.BlockSize) {
                    const ui64 size = Min<ui64>(Size - offset, State.BlockSize);
#if defined(WITHOUT_CUDA_AWARE_MPI)
                    CB_ENSURE(NCudaLib::GetPointerType(Buffer) == EPtrType::CudaHost);
#endif

                    if (IsRecv) {
                        State.Requests.push_back(manager.ReadAsync((char*)(Buffer + offset), sizeof(T) * size, RemoteHost, Tag));
                    } else {
                        State.Requests.push_back(manager.WriteAsync((const char*)(Buffer + offset), sizeof(T) * size, RemoteHost, Tag));
                    }
                }
                State.AreRequestsCreated = true;
            }

            return AreRequestsComplete(State.Requests);
        }
    };

    template <class T>
    inline bool DelegatePtrToMpi(const T* ptr) {
#if defined(WITHOUT_CUDA_AWARE_MPI)
        return IsHostPtr(NCudaLib::GetPointerType(ptr));
#else
        Y_UNUSED(ptr);
        return true;
#endif
    }

    template <class T>
    THolder<IStagedTask> BlockedSendTask(const T* source, ui64 size, ui64 blockSize,
                                         int host, int tag, NKernelHost::IMemoryManager& memoryManager,
                                         bool compress) {
        TMemcpyOperator<T> op;
        op.BlockSize = blockSize;

        if (DelegatePtrToMpi(source) && !compress) {
            using TTask = TMpiDelegatingStageTask<const T, false>;
            return MakeHolder<TTask>(source, size, op, host, tag, memoryManager);
        } else {
            using TTask = TThroughHostStagedSendTask<T, TMemcpyOperator<T>>;
            return MakeHolder<TTask>(source, size, op, host, tag, memoryManager, compress);
        }
    }

    template <class T>
    THolder<IStagedTask> BlockedRecvTask(T* dest, ui64 size, ui64 blockSize,
                                         int host, int tag, NKernelHost::IMemoryManager& memoryManager,
                                         bool compress) {
        TMemcpyOperator<T> op;
        op.BlockSize = blockSize;

        if (DelegatePtrToMpi(dest) && !compress) {
            using TTask = TMpiDelegatingStageTask<T, true>;
            return MakeHolder<TTask>(dest, size, op, host, tag, memoryManager);
        } else {
            using TTask = TThroughHostStagedRecvTask<T, TMemcpyOperator<T>>;
            return MakeHolder<TTask>(dest, size, op, host, tag, memoryManager, compress);
        }
    }

    inline void ExecStagedTasks(const TCudaStream& stream, TVector<IStagedTaskPtr>* tasks) {
        TVector<THolder<IStagedTask>> stillRunning;

        for (ui32 i = 0; i < tasks->size(); ++i) {
            if (!(*tasks)[i]->Exec(stream)) {
                stillRunning.push_back(std::move((*tasks)[i]));
            }
        }
        tasks->swap(stillRunning);
    }

#endif
}
