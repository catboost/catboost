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

#ifndef TCMALLOC_MOCK_CENTRAL_FREELIST_H_
#define TCMALLOC_MOCK_CENTRAL_FREELIST_H_

#include <stddef.h>

#include "gmock/gmock.h"
#include "absl/base/internal/spinlock.h"

namespace tcmalloc {

class FakeCentralFreeListBase {
 public:
  FakeCentralFreeListBase() : size_class_(0) {}
  FakeCentralFreeListBase(const FakeCentralFreeListBase&) = delete;
  FakeCentralFreeListBase& operator=(const FakeCentralFreeListBase&) = delete;

  void Init(size_t cl) { size_class_ = cl; }
  size_t length() { return 0; }
  size_t OverheadBytes() { return 0; }
  size_t size_class() const { return size_class_; }

 private:
  size_t size_class_;
};

// CentralFreeList implementation that backs onto the system's malloc.
//
// Useful for unit tests and fuzz tests where identifying leaks and correctness
// is important.
class FakeCentralFreeList : public FakeCentralFreeListBase {
 public:
  void InsertRange(void** batch, int N);
  int RemoveRange(void** batch, int N);

  void AllocateBatch(void** batch, int n);
  void FreeBatch(void** batch, int n);
};

// CentralFreeList implementation that does minimal work but no correctness
// checking.
//
// Useful for benchmarks where you want to avoid unrelated expensive operations.
class MinimalFakeCentralFreeList : public FakeCentralFreeListBase {
 public:
  void InsertRange(void** batch, int N);
  int RemoveRange(void** batch, int N);

  void AllocateBatch(void** batch, int n);
  void FreeBatch(void** batch, int n);

 private:
  absl::base_internal::SpinLock lock_;
};

// CentralFreeList implementation that allows intercepting specific calls.  By
// default backs onto the system's malloc.
//
// Useful for intrusive unit tests that want to verify internal behavior.
class RawMockCentralFreeList : public FakeCentralFreeList {
 public:
  RawMockCentralFreeList() : FakeCentralFreeList() {
    ON_CALL(*this, InsertRange).WillByDefault([this](void** batch, int n) {
      return static_cast<FakeCentralFreeList*>(this)->InsertRange(batch, n);
    });
    ON_CALL(*this, RemoveRange).WillByDefault([this](void** batch, int n) {
      return static_cast<FakeCentralFreeList*>(this)->RemoveRange(batch, n);
    });
  }

  MOCK_METHOD(void, InsertRange, (void** batch, int N));
  MOCK_METHOD(int, RemoveRange, (void** batch, int N));
};

using MockCentralFreeList = testing::NiceMock<RawMockCentralFreeList>;

}  // namespace tcmalloc

#endif  // TCMALLOC_MOCK_CENTRAL_FREELIST_H_
