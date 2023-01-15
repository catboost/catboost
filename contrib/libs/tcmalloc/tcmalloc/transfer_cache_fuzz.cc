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

#include <cstddef>
#include <cstdint>

#include "tcmalloc/mock_central_freelist.h"
#include "tcmalloc/mock_transfer_cache.h"
#include "tcmalloc/transfer_cache_internals.h"

namespace tcmalloc {
namespace {

using TransferCache =
    internal_transfer_cache::LockFreeTransferCache<MockCentralFreeList,
                                                   MockTransferCacheManager>;
using Env = FakeTransferCacheEnvironment<TransferCache>;

}  // namespace
}  // namespace tcmalloc

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  tcmalloc::Env env;
  for (int i = 0; i < size; ++i) {
    switch (data[i] % 10) {
      case 0:
        env.Grow();
        break;
      case 1:
        env.Shrink();
        break;
      default:
        if (++i < size) {
          int batch = data[i] % 32;
          if (data[i - 1] % 2) {
            env.Insert(batch);
          } else {
            env.Remove(batch);
          }
        }
        break;
    }
  }
  return 0;
}
