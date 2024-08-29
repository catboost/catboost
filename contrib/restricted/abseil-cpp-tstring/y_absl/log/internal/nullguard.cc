// Copyright 2023 The Abseil Authors
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

#include "y_absl/log/internal/nullguard.h"

#include <array>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace log_internal {

Y_ABSL_CONST_INIT Y_ABSL_DLL const std::array<char, 7> kCharNull{
    {'(', 'n', 'u', 'l', 'l', ')', '\0'}};
Y_ABSL_CONST_INIT Y_ABSL_DLL const std::array<signed char, 7> kSignedCharNull{
    {'(', 'n', 'u', 'l', 'l', ')', '\0'}};
Y_ABSL_CONST_INIT Y_ABSL_DLL const std::array<unsigned char, 7> kUnsignedCharNull{
    {'(', 'n', 'u', 'l', 'l', ')', '\0'}};

}  // namespace log_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
