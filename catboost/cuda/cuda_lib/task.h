#pragma once

#include "cuda_base.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "cuda_events_provider.h"
#include "remote_device_future.h"
#include "kernel.h"

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <util/ysafeptr.h>
#include <memory>
#include <util/system/event.h>
#include <future>
#include <library/threading/future/future.h>

namespace NCudaLib {
    enum class EGpuHostCommandType {
        StreamKernel,      //async tasks, will be launch in stream
        HostTask,          //sync task, ensure every task in stream was completed
        MemoryAllocation,  // usually async, but could sync or memory defragmentation
        MemoryDealocation, // sync everything and free memory
        RequestStream,
        FreeStream,
        WaitSubmit
    };

    class IGpuCommand {
    private:
        EGpuHostCommandType Type;

    public:
        explicit IGpuCommand(EGpuHostCommandType type)
            : Type(type)
        {
        }

        virtual ~IGpuCommand() {
        }

        EGpuHostCommandType GetCommandType() const {
            return Type;
        }
    };

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

    //for async tasks like kernels; async-memcpy; etc
    class IGpuKernelTask: public IGpuCommand {
    private:
        ui32 Stream;

    public:
        explicit IGpuKernelTask(ui32 stream = 0)
            : IGpuCommand(EGpuHostCommandType::StreamKernel)
            , Stream(stream)
        {
        }

        //for temp memory allocation.
        // All memory deallocation will be done before next sync command. Trade of memory for speed.
        virtual THolder<NKernel::IKernelContext> PrepareExec(NKernelHost::IMemoryManager& memoryManager) const {
            Y_UNUSED(memoryManager);
            return nullptr;
        }

        //BTW, some kernels could block this
        virtual void SubmitAsyncExec(const TCudaStream& stream,
                                     NKernel::IKernelContext* context) = 0;

        //Warning: could be called several times even after return true.
        virtual bool ReadyToSubmitNext(const TCudaStream& stream,
                                       NKernel::IKernelContext* context) {
            Y_UNUSED(stream);
            Y_UNUSED(context);
            return true;
        }

        ui32 GetStreamId() const {
            return Stream;
        }

        SAVELOAD(Stream);
    };

    //for async tasks like kernels; async-memcpy; etc
    class IGpuStatelessKernelTask: public IGpuKernelTask {
    protected:
        virtual void SubmitAsyncExecImpl(const TCudaStream& stream) = 0;

        virtual bool ReadyToSubmitNextImpl(const TCudaStream& stream) {
            Y_UNUSED(stream);
            return true;
        }

    public:
        explicit IGpuStatelessKernelTask(ui32 stream = 0)
            : IGpuKernelTask(stream)
        {
        }

        //for temp memory allocation.
        // All memory deallocation will be done before next sync command. Trade of memory for speed.
        THolder<NKernel::IKernelContext> PrepareExec(NKernelHost::IMemoryManager& memoryManager) const override {
            Y_UNUSED(memoryManager);
            return nullptr;
        }

        //BTW, some kernels could block this
        void SubmitAsyncExec(const TCudaStream& stream,
                             NKernel::IKernelContext* context) override final {
            Y_UNUSED(context);
            SubmitAsyncExecImpl(stream);
        }

        bool ReadyToSubmitNext(const TCudaStream& stream,
                               NKernel::IKernelContext* context) override final {
            Y_UNUSED(context);
            return ReadyToSubmitNextImpl(stream);
        }
    };

    namespace NHelpers {
        template <class TKernel, class TTempData>
        struct TKernelPrepareHelper {
            const TKernel& Kernel;

            explicit TKernelPrepareHelper(const TKernel& kernel)
                : Kernel(kernel)
            {
            }

            THolder<NKernel::IKernelContext> PrepareContext(NKernelHost::IMemoryManager& memoryManager) const {
                return Kernel.PrepareContext(memoryManager);
            }
        };

        template <class TKernel>
        struct TKernelPrepareHelper<TKernel, void> {
            const TKernel& Kernel;

            explicit TKernelPrepareHelper(const TKernel& kernel)
                : Kernel(kernel)
            {
            }

            THolder<NKernel::IKernelContext> PrepareContext(NKernelHost::IMemoryManager& memoryManager) const {
                Y_UNUSED(memoryManager);
                return nullptr;
            }
        };

        template <class TKernel, class TTempData>
        struct TKernelRunHelper {
            TKernel& Kernel;

            explicit TKernelRunHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            void Run(const TCudaStream& stream, NKernel::IKernelContext* data) {
                CB_ENSURE(data != nullptr);
                auto* tempData = reinterpret_cast<TTempData*>(data);
                Kernel.Run(stream, *tempData);
            }
        };

        template <class TKernel>
        struct TKernelRunHelper<TKernel, void> {
            TKernel& Kernel;

            explicit TKernelRunHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            void Run(const TCudaStream& stream, NKernel::IKernelContext* data) {
                CB_ENSURE(data == nullptr);
                Kernel.Run(stream);
            }
        };

        template <class TKernel, class TTempData, bool Flag>
        struct TKernelPostprocessHelper {
        };

        template <class TKernel>
        struct TKernelPostprocessHelper<TKernel, void, true> {
            TKernel& Kernel;

            explicit TKernelPostprocessHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream, NKernel::IKernelContext* context) {
                CB_ENSURE(context == nullptr);
                Kernel.Postprocess(stream);
            }
        };

        template <class TKernel, class TTempData>
        struct TKernelPostprocessHelper<TKernel, TTempData, true> {
            TKernel& Kernel;

            explicit TKernelPostprocessHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream, NKernel::IKernelContext* context) {
                CB_ENSURE(context != nullptr);
                auto* tempContext = reinterpret_cast<TTempData*>(context);
                Kernel.Postprocess(stream, *tempContext);
            }
        };

        template <class TKernel, class TTempData>
        struct TKernelPostprocessHelper<TKernel, TTempData, false> {
            TKernel& Kernel;

            explicit TKernelPostprocessHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream, NKernel::IKernelContext* data) {
                Y_UNUSED(stream);
                Y_UNUSED(data);
            }
        };

        template <class TKernel>
        struct TKernelContextTrait {
            using TKernelContext = typename std::remove_pointer<decltype(TKernel::EmptyContext())>::type;
        };

        template <class TKernel>
        struct TKernelPostprocessTrait {
            static constexpr bool Value = TKernel::NeedPostProcess();
        };
    }

    template <class TKernel>
    class TGpuKernelTask: public IGpuKernelTask {
    private:
        TKernel Kernel;
        TCudaEventPtr CudaEvent;
        bool PostProcessDone = false;

    public:
        explicit TGpuKernelTask(TKernel&& kernel, ui32 stream = 0)
            : IGpuKernelTask(stream)
            , Kernel(std::move(kernel))
        {
        }

        static constexpr bool NeedPostProcess = NHelpers::TKernelPostprocessTrait<TKernel>::Value;

        using TKernelContext = typename NHelpers::TKernelContextTrait<TKernel>::TKernelContext;

        THolder<NKernel::IKernelContext> PrepareExec(NKernelHost::IMemoryManager& memoryManager) const override {
            return NHelpers::TKernelPrepareHelper<TKernel, TKernelContext>(Kernel).PrepareContext(memoryManager);
        }

        void SubmitAsyncExec(const TCudaStream& stream,
                             NKernel::IKernelContext* data) override {
            NHelpers::TKernelRunHelper<TKernel, TKernelContext>(Kernel).Run(stream, data);
            if (NeedPostProcess) {
                CudaEvent = CreateCudaEvent(true);
                CudaEvent->Record(stream);
            }
        }

        bool ReadyToSubmitNext(const TCudaStream& stream,
                               NKernel::IKernelContext* data) override {
            if (PostProcessDone) {
                return true;
            }

            if (NeedPostProcess) {
                const bool isComplete = CudaEvent->IsComplete();
                if (!isComplete) {
                    return false;
                }
            }

            NHelpers::TKernelPostprocessHelper<TKernel, TKernelContext, NeedPostProcess>(Kernel).Run(stream, data);
            PostProcessDone = true;
            return true;
        }

        SAVELOAD(Kernel);
    };

    class IHostTask: public IGpuCommand {
    public:
        IHostTask()
            : IGpuCommand(EGpuHostCommandType::HostTask)
        {
        }

        virtual void Exec() = 0;
    };

    class IAllocateMemoryTask: public IGpuCommand {
    public:
        IAllocateMemoryTask()
            : IGpuCommand(EGpuHostCommandType::MemoryAllocation)
        {
        }

        virtual ui64 GetHandle() const = 0;

        virtual ui64 GetSize() const = 0;

        virtual EPtrType GetPtrType() const = 0;
    };

    template <EPtrType PtrType>
    class TCudaMallocTask: public IAllocateMemoryTask {
    private:
        ui64 Handle;
        ui64 Size;

    public:
        TCudaMallocTask(ui64 handle,
                        ui64 size)
            : Handle(handle)
            , Size(size)
        {
        }

        ui64 GetHandle() const override {
            return Handle;
        }

        ui64 GetSize() const override {
            return Size;
        }

        EPtrType GetPtrType() const {
            return PtrType;
        }

        SAVELOAD(Handle, Size);
    };

    class IFreeMemoryTask: public IGpuCommand {
    public:
        IFreeMemoryTask()
            : IGpuCommand(EGpuHostCommandType::MemoryDealocation)
        {
        }

        virtual void Exec() = 0;
    };

    template <class T>
    class TResetRemotePointerCommand: public IFreeMemoryTask {
    private:
        THandleBasedPointer<T> Ptr;

    public:
        explicit TResetRemotePointerCommand(THandleBasedPointer<T> ptr)
            : Ptr(ptr)
        {
        }

        void Exec() override {
            Ptr.Reset();
        }
    };

    template <class TTask>
    class THostTask: public IHostTask {
    public:
        using TOutput = typename TTask::TOutput;

        TDeviceFuture<TOutput> GetResult() {
            Promise = NThreading::NewPromise<TOutput>();
            return TDeviceFuture<TOutput>(Promise.GetFuture());
        }

        virtual void Exec() override {
            Promise.SetValue(Task());
        }

        THostTask(TTask&& task)
            : Task(std::move(task))
        {
        }

    private:
        TTask Task;
        NThreading::TPromise<TOutput> Promise;
    };

    struct TSyncStreamKernel: public NKernelHost::TStatelessKernel {
        void Run(const TCudaStream& stream) {
            stream.Synchronize();
        }
    };

    using TSyncTask = TGpuKernelTask<TSyncStreamKernel>;

    class TWaitSubmitCommand: public IGpuCommand {
    public:
        TWaitSubmitCommand()
            : IGpuCommand(EGpuHostCommandType::WaitSubmit)
        {
        }
    };

    struct TBlockingSyncDevice {
        using TOutput = ui64;

        ui64 operator()() {
            return 0;
        }
    };

    using TBlockingDeviceSynchronize = THostTask<TBlockingSyncDevice>;

    template <class T>
    struct TCreateObjectCommandTrait {
        class TWithoutConstructCommand: public IHostTask {
        private:
            ui64 Handle;

        public:
            explicit TWithoutConstructCommand(ui64 handle)
                : Handle(handle)
            {
            }

            void Exec() override {
                THandleBasedPointer<T> ptr(Handle);
                ptr.Reset(new T);
            }

            SAVELOAD(Handle);
        };

        static THolder<IHostTask> Create(ui64 handle) {
            return new TWithoutConstructCommand(handle);
        }
    };
}
