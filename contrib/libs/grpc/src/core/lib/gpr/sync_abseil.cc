//
//
// Copyright 2020 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//

#include <grpc/support/port_platform.h>

#if defined(GPR_ABSEIL_SYNC) && !defined(GPR_CUSTOM_SYNC)

#include <errno.h>
#include <time.h>

#include "y_absl/base/call_once.h"
#include "y_absl/synchronization/mutex.h"
#include "y_absl/time/clock.h"
#include "y_absl/time/time.h"

#include <grpc/support/alloc.h>
#include <grpc/support/log.h>
#include <grpc/support/sync.h>
#include <grpc/support/time.h>

#include "src/core/lib/gprpp/crash.h"
#include "src/core/lib/gprpp/time_util.h"

void gpr_mu_init(gpr_mu* mu) {
  static_assert(sizeof(gpr_mu) == sizeof(y_absl::Mutex),
                "gpr_mu and Mutex must be the same size");
  new (mu) y_absl::Mutex;
}

void gpr_mu_destroy(gpr_mu* mu) {
  reinterpret_cast<y_absl::Mutex*>(mu)->~Mutex();
}

void gpr_mu_lock(gpr_mu* mu) Y_ABSL_NO_THREAD_SAFETY_ANALYSIS {
  reinterpret_cast<y_absl::Mutex*>(mu)->Lock();
}

void gpr_mu_unlock(gpr_mu* mu) Y_ABSL_NO_THREAD_SAFETY_ANALYSIS {
  reinterpret_cast<y_absl::Mutex*>(mu)->Unlock();
}

int gpr_mu_trylock(gpr_mu* mu) {
  return reinterpret_cast<y_absl::Mutex*>(mu)->TryLock();
}

//----------------------------------------

void gpr_cv_init(gpr_cv* cv) {
  static_assert(sizeof(gpr_cv) == sizeof(y_absl::CondVar),
                "gpr_cv and CondVar must be the same size");
  new (cv) y_absl::CondVar;
}

void gpr_cv_destroy(gpr_cv* cv) {
  reinterpret_cast<y_absl::CondVar*>(cv)->~CondVar();
}

int gpr_cv_wait(gpr_cv* cv, gpr_mu* mu, gpr_timespec abs_deadline) {
  y_absl::CondVar* const c = reinterpret_cast<y_absl::CondVar*>(cv);
  y_absl::Mutex* const m = reinterpret_cast<y_absl::Mutex*>(mu);
  if (gpr_time_cmp(abs_deadline, gpr_inf_future(abs_deadline.clock_type)) ==
      0) {
    c->Wait(m);
    return 0;
  }
  // Use WaitWithTimeout if possible instead of WaitWithDeadline hoping that
  // it's going to use a monotonic clock.
  if (abs_deadline.clock_type == GPR_TIMESPAN) {
    return c->WaitWithTimeout(m, grpc_core::ToAbslDuration(abs_deadline));
  } else if (abs_deadline.clock_type == GPR_CLOCK_MONOTONIC) {
    y_absl::Duration duration = grpc_core::ToAbslDuration(
        gpr_time_sub(abs_deadline, gpr_now(GPR_CLOCK_MONOTONIC)));
    return c->WaitWithTimeout(m, duration);
  } else {
    return c->WaitWithDeadline(m, grpc_core::ToAbslTime(abs_deadline));
  }
}

void gpr_cv_signal(gpr_cv* cv) {
  reinterpret_cast<y_absl::CondVar*>(cv)->Signal();
}

void gpr_cv_broadcast(gpr_cv* cv) {
  reinterpret_cast<y_absl::CondVar*>(cv)->SignalAll();
}

//----------------------------------------

void gpr_once_init(gpr_once* once, void (*init_function)(void)) {
  static_assert(sizeof(gpr_once) == sizeof(y_absl::once_flag),
                "gpr_once and y_absl::once_flag must be the same size");
  y_absl::call_once(*reinterpret_cast<y_absl::once_flag*>(once), init_function);
}

#endif  // defined(GPR_ABSEIL_SYNC) && !defined(GPR_CUSTOM_SYNC)
