#include "omp_local_executor.h"

#include <library/cpp/threading/future/future.h>

#include <util/generic/utility.h>
#include <util/system/atomic.h>
#include <util/system/event.h>
#include <util/system/thread.h>
#include <util/system/tls.h>
#include <util/system/yield.h>
#include <util/thread/lfqueue.h>

#include <utility>

#if defined(USE_OMP)
#include <contrib/libs/cxxsupp/openmp/omp.h>
#endif

void NPar::OMPTLocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags) {
#if defined(USE_OMP)
    flags = 0;
    exec(id);
#else
    Exec(new TFunctionWrapper(std::move(exec)), id, flags);
#endif
}

void NPar::OMPTLocalExecutor::ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
#if defined(USE_OMP)
    flags = 0;
    if (lastId - firstId == 1) {
        exec(firstId);
    } else {
        #pragma omp parallel for schedule(static)
          for (int id = firstId; id < lastId; ++id) {
              exec(id);
          }
    }
#else
    ExecRange(new TFunctionWrapper(exec), firstId, lastId, flags);
#endif
}

void NPar::OMPTLocalExecutor::ExecRangeWithThrow(TLocallyExecutableFunction exec, int firstId, int lastId, int flags) {
#if defined(USE_OMP)
    flags = 0;
    if (lastId - firstId == 1) {
        exec(firstId);
    } else {
        #pragma omp parallel for schedule(static)
          for (int id = firstId; id < lastId; ++id) {
              exec(id);
          }
    }
#else
    Y_VERIFY((flags & WAIT_COMPLETE) != 0, "ExecRangeWithThrow() requires WAIT_COMPLETE to wait if exceptions arise.");
    TVector<NThreading::TFuture<void>> currentRun = ExecRangeWithFutures(exec, firstId, lastId, flags);
    for (auto& result : currentRun) {
        result.GetValueSync(); // Exception will be rethrown if exists. If several exception - only the one with minimal id is rethrown.
    }
#endif
}
