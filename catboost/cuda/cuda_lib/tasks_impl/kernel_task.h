#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/cuda/cuda_lib/cuda_events_provider.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/task.h>
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>
#include <util/ysaveload.h>
#include <memory>

namespace NCudaLib {
    //for async tasks like kernels; async-memcpy; etc
    class IGpuKernelTask: public ICommand {
    public:
        explicit IGpuKernelTask(ui32 stream = 0)
            : ICommand(EComandType::StreamKernel)
            , Stream(stream)
        {
        }

        //for temp memory allocation.
        // All memory deallocation will be done before next sync command. Trade of memory for speed.

        //our build system doesn't allow to use arcadia stl code in cu-files :( so need use dirty hacks in kernel files. First create all ptrs, then get raw ptr. They'll be consisten until kernel will be executed
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

        void Load(IInputStream* input) final {
            ::Load(input, Stream);
            LoadImpl(input);
        }

        void Save(IOutputStream* output) const final {
            ::Save(output, Stream);
            SaveImpl(output);
        }

    protected:
        virtual void LoadImpl(IInputStream*) {
            CB_ENSURE(false, "Unimplemented");
        }

        virtual void SaveImpl(IOutputStream*) const {
            CB_ENSURE(false, "Unimplemented");
        }

    private:
        ui32 Stream;
    };

    template <class TImpl>
    class IGpuStatelessKernelTask: public IGpuKernelTask {
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
                             NKernel::IKernelContext* context) final {
            Y_UNUSED(context);
            static_cast<TImpl*>(this)->SubmitAsyncExecImpl(stream);
        }

        bool ReadyToSubmitNext(const TCudaStream& stream,
                               NKernel::IKernelContext* context) final {
            Y_UNUSED(context);
            return static_cast<TImpl*>(this)->ReadyToSubmitNextImpl(stream);
        }
    };

    namespace NHelpers {
        template <class TKernel,
                  class TKernelContext>
        struct TKernelPrepareHelper {
            const TKernel& Kernel;

            explicit TKernelPrepareHelper(const TKernel& kernel)
                : Kernel(kernel)
            {
            }

            THolder<TKernelContext> PrepareContext(NKernelHost::IMemoryManager& memoryManager) const {
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

            THolder<void> PrepareContext(NKernelHost::IMemoryManager& memoryManager) const {
                Y_UNUSED(memoryManager);
                return nullptr;
            }
        };

        template <class TKernel,
                  class TKernelContext>
        struct TKernelRunHelper;

        template <class TKernel,
                  class TKernelContext>
        struct TKernelRunHelper {
            TKernel& Kernel;

            explicit TKernelRunHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            void Run(const TCudaStream& stream,
                     TKernelContext* data) {
                CB_ENSURE(data != nullptr);
                Kernel.Run(stream, *data);
            }
        };

        template <class TKernel>
        struct TKernelRunHelper<TKernel, void> {
            TKernel& Kernel;

            explicit TKernelRunHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream,
                            void* data) {
                CB_ENSURE(data == nullptr);
                Kernel.Run(stream);
            }
        };

        template <class TKernel,
                  class TKernelContext,
                  bool Flag>
        struct TKernelPostprocessHelper {
        };

        template <class TKernel>
        struct TKernelPostprocessHelper<TKernel, void, true> {
            TKernel& Kernel;

            explicit TKernelPostprocessHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream,
                            void* context) {
                CB_ENSURE(context == nullptr);
                Kernel.Postprocess(stream);
            }
        };

        template <class TKernel,
                  class TKernelContext>
        struct TKernelPostprocessHelper<TKernel, TKernelContext, true> {
            TKernel& Kernel;

            explicit TKernelPostprocessHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream,
                            TKernelContext* context) {
                Kernel.Postprocess(stream, *context);
            }
        };

        template <class TKernel,
                  class TKernelContext>
        struct TKernelPostprocessHelper<TKernel, TKernelContext, false> {
            TKernel& Kernel;

            explicit TKernelPostprocessHelper(TKernel& kernel)
                : Kernel(kernel)
            {
            }

            inline void Run(const TCudaStream& stream,
                            TKernelContext* data) {
                Y_UNUSED(stream);
                Y_UNUSED(data);
            }
        };

        template <class TKernel>
        struct TKernelContextTrait {
            using TKernelContext = std::remove_pointer_t<decltype(TKernel::EmptyContext())>;
        };

        template <class TKernel>
        struct TKernelPostprocessTrait {
            static constexpr bool Value = TKernel::NeedPostProcess();
        };
    }

    template <class TKernel>
    class TGpuKernelTask: public IGpuKernelTask {
    public:
        static constexpr bool NeedPostProcess = NHelpers::TKernelPostprocessTrait<TKernel>::Value;
        using TKernelContext = typename NHelpers::TKernelContextTrait<TKernel>::TKernelContext;

    private:
        TKernel Kernel;

    private:
        struct TGpuKernelTaskContext: public NKernel::IKernelContext {
            THolder<TKernelContext> KernelContext;
            TCudaEventPtr CudaEvent;
            bool PostProcessDone = false;
        };

        friend class TTaskSerializer;

    public:
        TGpuKernelTask() {
        }

        explicit TGpuKernelTask(TKernel&& kernel,
                                ui32 stream = 0)
            : IGpuKernelTask(stream)
            , Kernel(std::move(kernel))
        {
        }

        THolder<NKernel::IKernelContext> PrepareExec(NKernelHost::IMemoryManager& memoryManager) const final {
            THolder<TGpuKernelTaskContext> context = MakeHolder<TGpuKernelTaskContext>();
            context->KernelContext = NHelpers::TKernelPrepareHelper<TKernel, TKernelContext>(Kernel).PrepareContext(memoryManager);
            return std::move(context);
        }

        void SubmitAsyncExec(const TCudaStream& stream,
                             NKernel::IKernelContext* ctx) override {
            auto context = reinterpret_cast<TGpuKernelTaskContext*>(ctx);
            NHelpers::TKernelRunHelper<TKernel, TKernelContext>(Kernel).Run(stream,
                                                                            context->KernelContext.Get());
            if (NeedPostProcess) {
                context->CudaEvent = CreateCudaEvent(true);
                context->CudaEvent->Record(stream);
            }
        }

        bool ReadyToSubmitNext(const TCudaStream& stream,
                               NKernel::IKernelContext* ctx) final {
            auto* context = reinterpret_cast<TGpuKernelTaskContext*>(ctx);
            if (context->PostProcessDone) {
                return true;
            }

            if (NeedPostProcess) {
                const bool isComplete = context->CudaEvent->IsComplete();
                if (!isComplete) {
                    return false;
                }
            }

            NHelpers::TKernelPostprocessHelper<TKernel, TKernelContext, NeedPostProcess>(Kernel).Run(stream,
                                                                                                     context->KernelContext.Get());
            context->PostProcessDone = true;
            return true;
        }

        const TKernel& GetKernel() const {
            return Kernel;
        }

    protected:
        void LoadImpl(IInputStream* input) final {
            ::Load(input, Kernel);
        }

        void SaveImpl(IOutputStream* output) const final {
            ::Save(output, Kernel);
        }
    };

    struct TSyncStreamKernel: public NKernelHost::TStatelessKernel {
        void Run(const TCudaStream& stream) {
            stream.Synchronize();
        }

        Y_SAVELOAD_EMPTY();
    };

    using TSyncTask = TGpuKernelTask<TSyncStreamKernel>;

    template <class TKernel>
    class TKernelRegistrator {
    public:
        TKernelRegistrator(ui64 id) {
            using TTask = TGpuKernelTask<TKernel>;
            TTaskRegistrator<TTask> registrator(id);
        }
    };

#define REGISTER_KERNEL(id, className) \
    static TKernelRegistrator<className> taskRegistrator##id(id);

#define REGISTER_KERNEL_TEMPLATE(id, className, T) \
    static TKernelRegistrator<className<T>> taskRegistrator##id(id);

#define REGISTER_KERNEL_TEMPLATE_2(id, className, T1, T2) \
    static TKernelRegistrator<className<T1, T2>> taskRegistrator##id(id);

}
