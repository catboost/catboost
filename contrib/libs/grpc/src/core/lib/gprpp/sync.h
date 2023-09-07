//
//
// Copyright 2019 gRPC authors.
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

#ifndef GRPC_SRC_CORE_LIB_GPRPP_SYNC_H
#define GRPC_SRC_CORE_LIB_GPRPP_SYNC_H

#include <grpc/support/port_platform.h>

#include "y_absl/base/thread_annotations.h"
#include "y_absl/synchronization/mutex.h"

#include <grpc/support/log.h>
#include <grpc/support/sync.h>

#ifndef GPR_ABSEIL_SYNC
#include "src/core/lib/gprpp/time_util.h"
#endif

// The core library is not accessible in C++ codegen headers, and vice versa.
// Thus, we need to have duplicate headers with similar functionality.
// Make sure any change to this file is also reflected in
// include/grpcpp/impl/sync.h.
//
// Whenever possible, prefer using this file over <grpcpp/impl/sync.h>
// since this file doesn't rely on g_core_codegen_interface and hence does not
// pay the costs of virtual function calls.

namespace grpc_core {

#ifdef GPR_ABSEIL_SYNC

using Mutex = y_absl::Mutex;
using MutexLock = y_absl::MutexLock;
using ReleasableMutexLock = y_absl::ReleasableMutexLock;
using CondVar = y_absl::CondVar;

// Returns the underlying gpr_mu from Mutex. This should be used only when
// it has to like passing the C++ mutex to C-core API.
// TODO(veblush): Remove this after C-core no longer uses gpr_mu.
inline gpr_mu* GetUnderlyingGprMu(Mutex* mutex) {
  return reinterpret_cast<gpr_mu*>(mutex);
}

#else

class Y_ABSL_LOCKABLE Mutex {
 public:
  Mutex() { gpr_mu_init(&mu_); }
  ~Mutex() { gpr_mu_destroy(&mu_); }

  Mutex(const Mutex&) = delete;
  Mutex& operator=(const Mutex&) = delete;

  void Lock() Y_ABSL_EXCLUSIVE_LOCK_FUNCTION() { gpr_mu_lock(&mu_); }
  void Unlock() Y_ABSL_UNLOCK_FUNCTION() { gpr_mu_unlock(&mu_); }
  bool TryLock() Y_ABSL_EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return gpr_mu_trylock(&mu_) != 0;
  }
  void AssertHeld() Y_ABSL_ASSERT_EXCLUSIVE_LOCK() {}

 private:
  gpr_mu mu_;

  friend class CondVar;
  friend gpr_mu* GetUnderlyingGprMu(Mutex* mutex);
};

// Returns the underlying gpr_mu from Mutex. This should be used only when
// it has to like passing the C++ mutex to C-core API.
// TODO(veblush): Remove this after C-core no longer uses gpr_mu.
inline gpr_mu* GetUnderlyingGprMu(Mutex* mutex) { return &mutex->mu_; }

class Y_ABSL_SCOPED_LOCKABLE MutexLock {
 public:
  explicit MutexLock(Mutex* mu) Y_ABSL_EXCLUSIVE_LOCK_FUNCTION(mu) : mu_(mu) {
    mu_->Lock();
  }
  ~MutexLock() Y_ABSL_UNLOCK_FUNCTION() { mu_->Unlock(); }

  MutexLock(const MutexLock&) = delete;
  MutexLock& operator=(const MutexLock&) = delete;

 private:
  Mutex* const mu_;
};

class Y_ABSL_SCOPED_LOCKABLE ReleasableMutexLock {
 public:
  explicit ReleasableMutexLock(Mutex* mu) Y_ABSL_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(mu) {
    mu_->Lock();
  }
  ~ReleasableMutexLock() Y_ABSL_UNLOCK_FUNCTION() {
    if (!released_) mu_->Unlock();
  }

  ReleasableMutexLock(const ReleasableMutexLock&) = delete;
  ReleasableMutexLock& operator=(const ReleasableMutexLock&) = delete;

  void Release() Y_ABSL_UNLOCK_FUNCTION() {
    GPR_DEBUG_ASSERT(!released_);
    released_ = true;
    mu_->Unlock();
  }

 private:
  Mutex* const mu_;
  bool released_ = false;
};

class CondVar {
 public:
  CondVar() { gpr_cv_init(&cv_); }
  ~CondVar() { gpr_cv_destroy(&cv_); }

  CondVar(const CondVar&) = delete;
  CondVar& operator=(const CondVar&) = delete;

  void Signal() { gpr_cv_signal(&cv_); }
  void SignalAll() { gpr_cv_broadcast(&cv_); }

  void Wait(Mutex* mu) { WaitWithDeadline(mu, y_absl::InfiniteFuture()); }
  bool WaitWithTimeout(Mutex* mu, y_absl::Duration timeout) {
    return gpr_cv_wait(&cv_, &mu->mu_, ToGprTimeSpec(timeout)) != 0;
  }
  bool WaitWithDeadline(Mutex* mu, y_absl::Time deadline) {
    return gpr_cv_wait(&cv_, &mu->mu_, ToGprTimeSpec(deadline)) != 0;
  }

 private:
  gpr_cv cv_;
};

#endif  // GPR_ABSEIL_SYNC

// Deprecated. Prefer MutexLock
class MutexLockForGprMu {
 public:
  explicit MutexLockForGprMu(gpr_mu* mu) : mu_(mu) { gpr_mu_lock(mu_); }
  ~MutexLockForGprMu() { gpr_mu_unlock(mu_); }

  MutexLockForGprMu(const MutexLock&) = delete;
  MutexLockForGprMu& operator=(const MutexLock&) = delete;

 private:
  gpr_mu* const mu_;
};

// Deprecated. Prefer MutexLock or ReleasableMutexLock
class Y_ABSL_SCOPED_LOCKABLE LockableAndReleasableMutexLock {
 public:
  explicit LockableAndReleasableMutexLock(Mutex* mu)
      Y_ABSL_EXCLUSIVE_LOCK_FUNCTION(mu)
      : mu_(mu) {
    mu_->Lock();
  }
  ~LockableAndReleasableMutexLock() Y_ABSL_UNLOCK_FUNCTION() {
    if (!released_) mu_->Unlock();
  }

  LockableAndReleasableMutexLock(const LockableAndReleasableMutexLock&) =
      delete;
  LockableAndReleasableMutexLock& operator=(
      const LockableAndReleasableMutexLock&) = delete;

  void Lock() Y_ABSL_EXCLUSIVE_LOCK_FUNCTION() {
    GPR_DEBUG_ASSERT(released_);
    mu_->Lock();
    released_ = false;
  }

  void Release() Y_ABSL_UNLOCK_FUNCTION() {
    GPR_DEBUG_ASSERT(!released_);
    released_ = true;
    mu_->Unlock();
  }

 private:
  Mutex* const mu_;
  bool released_ = false;
};

}  // namespace grpc_core

#endif  // GRPC_SRC_CORE_LIB_GPRPP_SYNC_H
