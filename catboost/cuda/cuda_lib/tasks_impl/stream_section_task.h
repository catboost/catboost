#pragma once

#include "kernel_task.h"
#include <catboost/cuda/cuda_lib/inter_device_stream_section.h>

namespace NCudaLib {
    /*
     * This tasks guarantees handle consistency for kernel and ensures all previous tasks are completed
     *  so we could access peered devices as well as send/read dat
     */
    template <class TKernel>
    class TStreamSectionKernelTask: public IGpuKernelTask {
    public:
        using TKernelContext = typename NHelpers::TKernelContextTrait<TKernel>::TKernelContext;

    private:
        enum ESectionState {
            Entering,
            Entered,
            Leaving,
            Left
        };

        struct TGpuKernelTaskContext: public NKernel::IKernelContext {
            THolder<TKernelContext> KernelContext;
            THolder<TStreamSection> Section;
            ESectionState SectionState = ESectionState::Entering;
        };

        friend class TTaskSerializer;

    public:
        TStreamSectionKernelTask() {
        }

        explicit TStreamSectionKernelTask(TKernel&& kernel,
                                          const TStreamSectionConfig& section,
                                          ui32 stream = 0)
            : IGpuKernelTask(stream)
            , Kernel(std::move(kernel))
            , Section(section)
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
            if (Section.StreamSectionSize <= 1 && Section.LocalOnly) {
                //shortcut
                context->SectionState = ESectionState::Entered;
            } else {
                context->Section = GetStreamSectionProvider().Create(Section, stream);
            }
            ReadyToSubmitNext(stream, ctx);
        }

        bool ReadyToSubmitNext(const TCudaStream& stream,
                               NKernel::IKernelContext* ctx) final {
            auto* context = reinterpret_cast<TGpuKernelTaskContext*>(ctx);
            switch (context->SectionState) {
                case ESectionState::Entering: {
                    Y_ASSERT(context->Section);
                    if (context->Section->TryEnter()) {
                        context->SectionState = ESectionState::Entered;
                    } else {
                        return false;
                    }
                    [[fallthrough]];
                }
                case ESectionState::Entered: {
                    const bool isDone = Kernel.Exec(stream,
                                                    context->KernelContext.Get());
                    if (isDone) {
                        context->SectionState = ESectionState::Leaving;
                    } else {
                        return false;
                    }
                    [[fallthrough]];
                }
                case ESectionState::Leaving: {
                    if (context->Section) {
                        if (context->Section->TryLeave()) {
                            context->SectionState = ESectionState::Left;
                        } else {
                            return false;
                        }
                    } else {
                        context->SectionState = ESectionState::Left;
                    }
                    [[fallthrough]];
                }
                case ESectionState::Left: {
                    return true;
                }
            }
            CB_ENSURE(false, "Unexpected stream section state");
        }

    protected:
        void LoadImpl(IInputStream* input) final {
            ::Load(input, Kernel);
            ::Load(input, Section);
        }

        void SaveImpl(IOutputStream* output) const final {
            ::Save(output, Kernel);
            ::Save(output, Section);
        }

    private:
        TKernel Kernel;
        TStreamSectionConfig Section;
    };

    template <class TKernel>
    class TStreamSectionTaskRegistrator {
    public:
        TStreamSectionTaskRegistrator(ui64 id) {
            using TTask = TStreamSectionKernelTask<TKernel>;
            TTaskRegistrator<TTask> registrator(id);
        }
    };

#define REGISTER_STREAM_SECTION_TASK(id, className) \
    static TStreamSectionTaskRegistrator<className> kernelRegistrator##className_##id(id);

#define REGISTER_STREAM_SECTION_TASK_TEMPLATE(id, className, T) \
    static TStreamSectionTaskRegistrator<className<T>> kernelRegistrator_##className_##T_##id(id);

}
