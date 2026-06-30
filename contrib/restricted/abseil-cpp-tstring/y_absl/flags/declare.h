//
//  Copyright 2019 The Abseil Authors.
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
// -----------------------------------------------------------------------------
// File: declare.h
// -----------------------------------------------------------------------------
//
// This file defines the Y_ABSL_DECLARE_FLAG macro, allowing you to declare an
// `y_absl::Flag` for use within a translation unit. You should place this
// declaration within the header file associated with the .cc file that defines
// and owns the `Flag`.

#ifndef Y_ABSL_FLAGS_DECLARE_H_
#define Y_ABSL_FLAGS_DECLARE_H_

#include "y_absl/base/config.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace flags_internal {

// y_absl::Flag<T> represents a flag of type 'T' created by Y_ABSL_FLAG.
template <typename T>
class Flag;

}  // namespace flags_internal

// Flag
//
// Forward declaration of the `y_absl::Flag` type for use in defining the macro.
template <typename T>
using Flag = flags_internal::Flag<T>;

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

// Y_ABSL_DECLARE_FLAG()
//
// This macro is a convenience for declaring use of an `y_absl::Flag` within a
// translation unit. This macro should be used within a header file to
// declare usage of the flag within any .cc file including that header file.
//
// The Y_ABSL_DECLARE_FLAG(type, name) macro expands to:
//
//   extern y_absl::Flag<type> FLAGS_name;
#define Y_ABSL_DECLARE_FLAG(type, name) Y_ABSL_DECLARE_FLAG_INTERNAL(type, name)

// Internal implementation of Y_ABSL_DECLARE_FLAG to allow macro expansion of its
// arguments. Clients must use Y_ABSL_DECLARE_FLAG instead.
#define Y_ABSL_DECLARE_FLAG_INTERNAL(type, name)               \
  extern y_absl::Flag<type> FLAGS_##name;                      \
  namespace y_absl /* block flags in namespaces */ {}          \
  /* second redeclaration is to allow applying attributes */ \
  extern y_absl::Flag<type> FLAGS_##name

#endif  // Y_ABSL_FLAGS_DECLARE_H_
