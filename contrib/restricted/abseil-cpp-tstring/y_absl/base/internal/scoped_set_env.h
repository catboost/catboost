//
// Copyright 2019 The Abseil Authors.
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

#ifndef Y_ABSL_BASE_INTERNAL_SCOPED_SET_ENV_H_
#define Y_ABSL_BASE_INTERNAL_SCOPED_SET_ENV_H_

#include <util/generic/string.h>

#include "y_absl/base/config.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace base_internal {

class ScopedSetEnv {
 public:
  ScopedSetEnv(const char* var_name, const char* new_value);
  ~ScopedSetEnv();

 private:
  TString var_name_;
  TString old_value_;

  // True if the environment variable was initially not set.
  bool was_unset_;
};

}  // namespace base_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_BASE_INTERNAL_SCOPED_SET_ENV_H_
