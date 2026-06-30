// Copyright 2021 The Abseil Authors.
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

#ifndef Y_ABSL_CLEANUP_INTERNAL_CLEANUP_H_
#define Y_ABSL_CLEANUP_INTERNAL_CLEANUP_H_

#include <new>
#include <type_traits>
#include <utility>

#include "y_absl/base/internal/invoke.h"
#include "y_absl/base/macros.h"
#include "y_absl/base/thread_annotations.h"
#include "y_absl/utility/utility.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

namespace cleanup_internal {

struct Tag {};

template <typename Arg, typename... Args>
constexpr bool WasDeduced() {
  return (std::is_same<cleanup_internal::Tag, Arg>::value) &&
         (sizeof...(Args) == 0);
}

template <typename Callback>
constexpr bool ReturnsVoid() {
  return (std::is_same<base_internal::invoke_result_t<Callback>, void>::value);
}

template <typename Callback>
class Storage {
 public:
  Storage() = delete;

  explicit Storage(Callback callback) {
    // Placement-new into a character buffer is used for eager destruction when
    // the cleanup is invoked or cancelled. To ensure this optimizes well, the
    // behavior is implemented locally instead of using an y_absl::optional.
    ::new (GetCallbackBuffer()) Callback(std::move(callback));
    is_callback_engaged_ = true;
  }

  Storage(Storage&& other) {
    Y_ABSL_HARDENING_ASSERT(other.IsCallbackEngaged());

    ::new (GetCallbackBuffer()) Callback(std::move(other.GetCallback()));
    is_callback_engaged_ = true;

    other.DestroyCallback();
  }

  Storage(const Storage& other) = delete;

  Storage& operator=(Storage&& other) = delete;

  Storage& operator=(const Storage& other) = delete;

  void* GetCallbackBuffer() { return static_cast<void*>(+callback_buffer_); }

  Callback& GetCallback() {
    return *reinterpret_cast<Callback*>(GetCallbackBuffer());
  }

  bool IsCallbackEngaged() const { return is_callback_engaged_; }

  void DestroyCallback() {
    is_callback_engaged_ = false;
    GetCallback().~Callback();
  }

  void InvokeCallback() Y_ABSL_NO_THREAD_SAFETY_ANALYSIS {
    std::move(GetCallback())();
  }

 private:
  bool is_callback_engaged_;
  alignas(Callback) char callback_buffer_[sizeof(Callback)];
};

}  // namespace cleanup_internal

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_CLEANUP_INTERNAL_CLEANUP_H_
