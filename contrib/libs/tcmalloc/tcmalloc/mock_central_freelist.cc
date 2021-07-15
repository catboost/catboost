// Copyright 2020 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tcmalloc/mock_central_freelist.h"

#include "absl/base/internal/spinlock.h"
#include "tcmalloc/internal/logging.h"

namespace tcmalloc {

void MinimalFakeCentralFreeList::AllocateBatch(void** batch, int n) {
  for (int i = 0; i < n; ++i) batch[i] = &batch[i];
}

void MinimalFakeCentralFreeList::FreeBatch(void** batch, int n) {
  for (int i = 0; i < n; ++i) CHECK_CONDITION(batch[i] != nullptr);
}

void MinimalFakeCentralFreeList::InsertRange(void** batch, int n) {
  absl::base_internal::SpinLockHolder h(&lock_);
  FreeBatch(batch, n);
}

int MinimalFakeCentralFreeList::RemoveRange(void** batch, int n) {
  absl::base_internal::SpinLockHolder h(&lock_);
  AllocateBatch(batch, n);
  return n;
}

void FakeCentralFreeList::AllocateBatch(void** batch, int n) {
  for (int i = 0; i < n; ++i) {
    batch[i] = ::operator new(4);
  }
}

void FakeCentralFreeList::FreeBatch(void** batch, int n) {
  for (int i = 0; i < n; ++i) {
    ::operator delete(batch[i]);
  }
}

void FakeCentralFreeList::InsertRange(void** batch, int n) {
  FreeBatch(batch, n);
}

int FakeCentralFreeList::RemoveRange(void** batch, int n) {
  AllocateBatch(batch, n);
  return n;
}

}  // namespace tcmalloc
