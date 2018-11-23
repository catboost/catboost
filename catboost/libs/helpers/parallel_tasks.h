#pragma once

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

#include <functional>


namespace NCB {

    // tasks is not const because elements of tasks are cleared after execution
    void ExecuteTasksInParallel(TVector<std::function<void()>>* tasks, NPar::TLocalExecutor* localExecutor);

}
