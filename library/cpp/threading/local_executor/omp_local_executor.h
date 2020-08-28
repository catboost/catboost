#pragma once

#include "local_executor.h"

namespace NPar {
    class OMPTLocalExecutor: public TLocalExecutor {
    public:
        // Creates executor without threads. You'll need to explicitly call `RunAdditionalThreads`
        // to add threads to underlying thread pool.
        //
        OMPTLocalExecutor() = default;
        ~OMPTLocalExecutor() = default;
        // `Exec` and `ExecRange` versions that accept functions.
        //
        void Exec(TLocallyExecutableFunction exec, int id, int flags) override;
        void ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) override;

        // Version of `ExecRange` that throws exception from task with minimal id if at least one of
        // task threw an exception.
        //
        void ExecRangeWithThrow(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) override;
    };
}
