// Copyright 2023 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef Y_ABSL_SYNCHRONIZATION_INTERNAL_SEM_WAITER_H_
#define Y_ABSL_SYNCHRONIZATION_INTERNAL_SEM_WAITER_H_

#include "y_absl/base/config.h"

#ifdef Y_ABSL_HAVE_SEMAPHORE_H
#include <semaphore.h>

#include <atomic>
#include <cstdint>

#include "y_absl/base/internal/thread_identity.h"
#include "y_absl/synchronization/internal/futex.h"
#include "y_absl/synchronization/internal/kernel_timeout.h"
#include "y_absl/synchronization/internal/waiter_base.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace synchronization_internal {

#define Y_ABSL_INTERNAL_HAVE_SEM_WAITER 1

class SemWaiter : public WaiterCrtp<SemWaiter> {
 public:
  SemWaiter();

  bool Wait(KernelTimeout t);
  void Post();
  void Poke();

  static constexpr char kName[] = "SemWaiter";

 private:
  int TimedWait(KernelTimeout t);

  sem_t sem_;

  // This seems superfluous, but for Poke() we need to cause spurious
  // wakeups on the semaphore. Hence we can't actually use the
  // semaphore's count.
  std::atomic<int> wakeups_;
};

}  // namespace synchronization_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_HAVE_SEMAPHORE_H

#endif  // Y_ABSL_SYNCHRONIZATION_INTERNAL_SEM_WAITER_H_
