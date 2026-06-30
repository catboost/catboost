// Copyright 2017 The Abseil Authors.
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

#include "y_absl/strings/substitute.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <util/generic/string.h>

#include "y_absl/base/config.h"
#include "y_absl/base/internal/raw_logging.h"
#include "y_absl/base/nullability.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/internal/resize_uninitialized.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/string_view.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace substitute_internal {

void SubstituteAndAppendArray(
    y_absl::Nonnull<TString*> output, y_absl::string_view format,
    y_absl::Nullable<const y_absl::string_view*> args_array, size_t num_args) {
  // Determine total size needed.
  size_t size = 0;
  for (size_t i = 0; i < format.size(); i++) {
    if (format[i] == '$') {
      if (i + 1 >= format.size()) {
#ifndef NDEBUG
        Y_ABSL_RAW_LOG(FATAL,
                     "Invalid y_absl::Substitute() format string: \"%s\".",
                     y_absl::CEscape(format).c_str());
#endif
        return;
      } else if (y_absl::ascii_isdigit(
                     static_cast<unsigned char>(format[i + 1]))) {
        int index = format[i + 1] - '0';
        if (static_cast<size_t>(index) >= num_args) {
#ifndef NDEBUG
          Y_ABSL_RAW_LOG(
              FATAL,
              "Invalid y_absl::Substitute() format string: asked for \"$"
              "%d\", but only %d args were given.  Full format string was: "
              "\"%s\".",
              index, static_cast<int>(num_args), y_absl::CEscape(format).c_str());
#endif
          return;
        }
        size += args_array[index].size();
        ++i;  // Skip next char.
      } else if (format[i + 1] == '$') {
        ++size;
        ++i;  // Skip next char.
      } else {
#ifndef NDEBUG
        Y_ABSL_RAW_LOG(FATAL,
                     "Invalid y_absl::Substitute() format string: \"%s\".",
                     y_absl::CEscape(format).c_str());
#endif
        return;
      }
    } else {
      ++size;
    }
  }

  if (size == 0) return;

  // Build the string.
  size_t original_size = output->size();
  Y_ABSL_INTERNAL_CHECK(
      size <= std::numeric_limits<size_t>::max() - original_size,
      "size_t overflow");
  strings_internal::STLStringResizeUninitializedAmortized(output,
                                                          original_size + size);
  char* target = &(*output)[original_size];
  for (size_t i = 0; i < format.size(); i++) {
    if (format[i] == '$') {
      if (y_absl::ascii_isdigit(static_cast<unsigned char>(format[i + 1]))) {
        const y_absl::string_view src = args_array[format[i + 1] - '0'];
        target = std::copy(src.begin(), src.end(), target);
        ++i;  // Skip next char.
      } else if (format[i + 1] == '$') {
        *target++ = '$';
        ++i;  // Skip next char.
      }
    } else {
      *target++ = format[i];
    }
  }

  assert(target == output->data() + output->size());
}

Arg::Arg(y_absl::Nullable<const void*> value) {
  static_assert(sizeof(scratch_) >= sizeof(value) * 2 + 2,
                "fix sizeof(scratch_)");
  if (value == nullptr) {
    piece_ = "NULL";
  } else {
    char* ptr = scratch_ + sizeof(scratch_);
    uintptr_t num = reinterpret_cast<uintptr_t>(value);
    do {
      *--ptr = y_absl::numbers_internal::kHexChar[num & 0xf];
      num >>= 4;
    } while (num != 0);
    *--ptr = 'x';
    *--ptr = '0';
    piece_ = y_absl::string_view(
        ptr, static_cast<size_t>(scratch_ + sizeof(scratch_) - ptr));
  }
}

// TODO(jorg): Don't duplicate so much code between here and str_cat.cc
Arg::Arg(Hex hex) {
  char* const end = &scratch_[numbers_internal::kFastToBufferSize];
  char* writer = end;
  uint64_t value = hex.value;
  do {
    *--writer = y_absl::numbers_internal::kHexChar[value & 0xF];
    value >>= 4;
  } while (value != 0);

  char* beg;
  if (end - writer < hex.width) {
    beg = end - hex.width;
    std::fill_n(beg, writer - beg, hex.fill);
  } else {
    beg = writer;
  }

  piece_ = y_absl::string_view(beg, static_cast<size_t>(end - beg));
}

// TODO(jorg): Don't duplicate so much code between here and str_cat.cc
Arg::Arg(Dec dec) {
  assert(dec.width <= numbers_internal::kFastToBufferSize);
  char* const end = &scratch_[numbers_internal::kFastToBufferSize];
  char* const minfill = end - dec.width;
  char* writer = end;
  uint64_t value = dec.value;
  bool neg = dec.neg;
  while (value > 9) {
    *--writer = '0' + (value % 10);
    value /= 10;
  }
  *--writer = '0' + static_cast<char>(value);
  if (neg) *--writer = '-';

  ptrdiff_t fillers = writer - minfill;
  if (fillers > 0) {
    // Tricky: if the fill character is ' ', then it's <fill><+/-><digits>
    // But...: if the fill character is '0', then it's <+/-><fill><digits>
    bool add_sign_again = false;
    if (neg && dec.fill == '0') {  // If filling with '0',
      ++writer;                    // ignore the sign we just added
      add_sign_again = true;       // and re-add the sign later.
    }
    writer -= fillers;
    std::fill_n(writer, fillers, dec.fill);
    if (add_sign_again) *--writer = '-';
  }

  piece_ = y_absl::string_view(writer, static_cast<size_t>(end - writer));
}

}  // namespace substitute_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
