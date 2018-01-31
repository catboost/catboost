#pragma once

#include <catboost/cuda/cuda_lib/task.h>
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>

#include <catboost/cuda/cuda_lib/future/local_promise_future.h>
#include <catboost/cuda/cuda_lib/future/mpi_promise_future.h>
#include <catboost/cuda/cuda_lib/future/promise_factory.h>

namespace NCudaLib {
    template <class TTask>
    struct THostFuncTrait {
        static constexpr bool NeedWorkerState() {
            return false;
        }
    };

    template <class TFunc,
              bool NeedWorkerState = THostFuncTrait<TFunc>::NeedWorkerState()>
    struct TFuncRunner;

    template <class TFunc>
    struct TFuncRunner<TFunc, false> {
        static auto Run(TFunc* func,
                        const IWorkerStateProvider* workerState) -> decltype((*func)()) {
            Y_UNUSED(workerState);
            return (*func)();
        }
    };

    template <class TFunc>
    struct TFuncRunner<TFunc, true> {
        static auto Run(TFunc* func,
                        const IWorkerStateProvider* workerState) -> decltype((*func)(*workerState)) {
            return (*func)(*workerState);
        }
    };

    template <class TFunc>
    struct TFuncReturnType {
        using TOutput = decltype(TFuncRunner<std::remove_reference_t<TFunc>>::Run(nullptr, nullptr));
    };

    template <class TTask,
              bool IsRemote = false>
    class TCpuFunc: public IHostTask {
    public:
        using TOutput = typename TFuncReturnType<TTask>::TOutput;
        using TPromise = typename TPromiseFactory<IsRemote>::template TPromise<TOutput>;
        using TFuturePtr = THolder<IDeviceFuture<TOutput>>;

        TFuturePtr GetResult() {
            return Promise.GetFuture();
        }

        void Exec(const IWorkerStateProvider& workerStateProvider) final {
            auto result = TFuncRunner<TTask>::Run(&Task, &workerStateProvider);
            Promise.SetValue(result);
        }

        ECpuFuncType GetHostTaskType() final {
            return TTask::FuncType();
        }

        TCpuFunc() {
        }

        TCpuFunc(TPromise&& promise,
                 TTask&& task)
            : Task(std::forward<TTask>(task))
            , Promise(std::move(promise))
        {
        }

        template <class... TArgs>
        TCpuFunc(TPromise&& promise,
                 TArgs... args)
            : Promise(std::move(promise))
            , Task(std::forward<TArgs>(args)...)
        {
        }

        void Load(IInputStream* input) final {
            ::Load(input, Promise);
            ::Load(input, Task);
        }

        void Save(IOutputStream* output) const final {
            ::Save(output, Promise);
            ::Save(output, Task);
        }

    private:
        TTask Task;
        TPromise Promise;
    };

    template <class TFunc>
    class TFuncRegistrator {
    public:
        explicit TFuncRegistrator(ui64 id) {
            using TTask = TCpuFunc<TFunc, true>;
            GetTaskUniqueIdsProvider().RegisterUniqueId<TTask>(id);
            TTaskFactory::TRegistrator<TTask> registrator(id);
        }
    };

#define REGISTER_CPU_FUNC(id, className) \
    static TFuncRegistrator<className> taskRegistrator##id(id);

}
