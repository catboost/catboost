#include "resource_constrained_executor.h"

#include "exception.h"
#include "parallel_tasks.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/vector.h>
#include <util/stream/str.h>
#include <util/system/guard.h>

#include <exception>


namespace NCB {

    TResourceConstrainedExecutor::TResourceConstrainedExecutor(
        const TString& resourceName,
        TResourceUnit resourceQuota,
        bool lenientMode,
        NPar::ILocalExecutor* localExecutor
    )
        : LocalExecutor(*localExecutor)
        , ResourceName(resourceName)
        , ResourceQuota(resourceQuota)
        , LenientMode(lenientMode)
    {}

    TResourceConstrainedExecutor::~TResourceConstrainedExecutor() noexcept(false) {
        ExecTasks();
    }

    void TResourceConstrainedExecutor::Add(TFunctionWithResourceUsage&& functionWithResourceUsage) {
        if (functionWithResourceUsage.first > ResourceQuota) {
            TStringStream message;
            message << "Resource " << ResourceName
                << ": functionWithResourceUsage.ResourceUsage(" << functionWithResourceUsage.first
                << ") > ResourceQuota(" << ResourceQuota << ')';
            if (LenientMode) {
                CATBOOST_WARNING_LOG << message.Str() << Endl;
            } else {
                ythrow TCatBoostException() << message.Str();
            }
        }

        Queue.insert(std::move(functionWithResourceUsage));
    }

    void TResourceConstrainedExecutor::ExecTasks() {
        while (!Queue.empty()) {
            TVector<std::function<void()>> tasks;

            TResourceUnit freeResource = ResourceQuota;

            while (true) {
                auto it = Queue.lower_bound(freeResource);
                if (it == Queue.end()) {
                    break;
                }

                freeResource -= it->first;
                tasks.push_back(std::move(it->second));
                Queue.erase(it);
            };

            if (LenientMode && tasks.empty()) {
                // execute at least one task even if it requests more than ResourceQuota
                auto it = Queue.begin();
                tasks.push_back(std::move(it->second));
                Queue.erase(it);
            } else {
                Y_ASSERT(!tasks.empty());
            }

            ExecuteTasksInParallel(&tasks, &LocalExecutor);
        }
    }
}
