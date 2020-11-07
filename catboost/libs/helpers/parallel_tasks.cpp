#include "parallel_tasks.h"

#include <util/generic/cast.h>


void NCB::ExecuteTasksInParallel(TVector<std::function<void()>>* tasks, NPar::ILocalExecutor* localExecutor) {
    localExecutor->ExecRangeWithThrow(
        [&tasks](int id) {
            (*tasks)[id]();
            (*tasks)[id] = nullptr; // destroy early, do not wait for all tasks to finish
        },
        0,
        SafeIntegerCast<int>(tasks->size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}
