#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/task.h>
#include <catboost/cuda/cuda_lib/tasks_impl/remote_device_future.h>
#include <library/threading/future/future.h>
#include <util/ysafeptr.h>
#include <util/system/event.h>
#include <util/ysaveload.h>
#include <memory>

namespace NCudaLib {
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

        Y_SAVELOAD_DEFINE(Stream);
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
            using TKernelContext = std::remove_pointer_t<decltype(TKernel::EmptyContext())>;
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

        Y_SAVELOAD_DEFINE(Kernel);
    };

    struct TSyncStreamKernel: public NKernelHost::TStatelessKernel {
        void Run(const TCudaStream& stream) {
            stream.Synchronize();
        }
    };

    using TSyncTask = TGpuKernelTask<TSyncStreamKernel>;

}
