// Copyright 2022 The Abseil Authors
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

#ifndef Y_ABSL_STRINGS_INTERNAL_DAMERAU_LEVENSHTEIN_DISTANCE_H_
#define Y_ABSL_STRINGS_INTERNAL_DAMERAU_LEVENSHTEIN_DISTANCE_H_

#include <cstdint>

#include "y_absl/strings/string_view.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace strings_internal {
// Calculate DamerauLevenshtein distance between two strings.
// When the distance is larger than cutoff, the code just returns cutoff + 1.
uint8_t CappedDamerauLevenshteinDistance(y_absl::string_view s1,
                                         y_absl::string_view s2, uint8_t cutoff);

}  // namespace strings_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STRINGS_INTERNAL_DAMERAU_LEVENSHTEIN_DISTANCE_H_
