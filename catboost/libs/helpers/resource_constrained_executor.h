#pragma once

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/map.h>
#include <util/system/types.h>

#include <functional>
#include <utility>


namespace NCB {

    /*
      Executor class that runs tasks that consume some resource with total usage limit (e.g. RAM)
        concurrently (using LocalExecutor) but obeying this total usage limit.
        Tasks are run in order of decreasing resource usage within current limit
        (taking into account already running tasks):
         task with the maximum resource usage that can be run under current limit is chosen first.

        If LenientMode is enabled allow to execute tasks with more resource usage than the quota

        Destructor is equivalent to ExecTasks

        Exceptions are propagated.

      TODO(akhropov): maybe add task priorities
    */
    class TResourceConstrainedExecutor {
    public:
        /* unit to measure resurce consumption in
         * ui64 for now, maybe will be a template parameter if need arises
         */
        using TResourceUnit = ui64;

        using TFunctionWithResourceUsage = std::pair<const TResourceUnit, std::function<void()>>;

    private:
        using TQueue = TMultiMap<TResourceUnit, std::function<void()>, std::greater<TResourceUnit>>;

    public:
        TResourceConstrainedExecutor(
            const TString& resourceName,
            TResourceUnit resourceQuota,
            bool lenientMode,
            NPar::ILocalExecutor* localExecutor
        );

        // waits until all tasks are finished
        ~TResourceConstrainedExecutor() noexcept(false);

        void Add(TFunctionWithResourceUsage&& functionWithResourceUsage);

        /* Executes all tasks added so far until completion.
         * It is possible to add tasks after that and call this method again.
         */
        void ExecTasks();

        NPar::ILocalExecutor* GetExecutorPtr() {
            return &LocalExecutor;
        }

    private:
        NPar::ILocalExecutor& LocalExecutor;
        TString ResourceName;
        const TResourceUnit ResourceQuota;
        TQueue Queue;
        bool LenientMode;
    };
}
