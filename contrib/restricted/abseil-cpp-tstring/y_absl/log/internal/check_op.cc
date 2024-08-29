// Copyright 2022 The Abseil Authors.
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

#include "y_absl/log/internal/check_op.h"

#include <string.h>

#include <ostream>

#include "y_absl/strings/string_view.h"

#ifdef _MSC_VER
#define strcasecmp _stricmp
#else
#include <strings.h>  // for strcasecmp, but msvc does not have this header
#endif

#include <sstream>
#include <util/generic/string.h>

#include "y_absl/base/config.h"
#include "y_absl/strings/str_cat.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace log_internal {

#define Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(x) \
  template TString* MakeCheckOpString(x, x, const char*)
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(bool);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(int64_t);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(uint64_t);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(float);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(double);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(char);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(unsigned char);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(const TString&);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(const y_absl::string_view&);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(const char*);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(const signed char*);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(const unsigned char*);
Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING(const void*);
#undef Y_ABSL_LOGGING_INTERNAL_DEFINE_MAKE_CHECK_OP_STRING

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext) {
  stream_ << exprtext << " (";
}

std::ostream& CheckOpMessageBuilder::ForVar2() {
  stream_ << " vs. ";
  return stream_;
}

TString* CheckOpMessageBuilder::NewString() {
  stream_ << ")";
  return new TString(stream_.str());
}

void MakeCheckOpValueString(std::ostream& os, const char v) {
  if (v >= 32 && v <= 126) {
    os << "'" << v << "'";
  } else {
    os << "char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream& os, const signed char v) {
  if (v >= 32 && v <= 126) {
    os << "'" << v << "'";
  } else {
    os << "signed char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream& os, const unsigned char v) {
  if (v >= 32 && v <= 126) {
    os << "'" << v << "'";
  } else {
    os << "unsigned char value " << int{v};
  }
}

void MakeCheckOpValueString(std::ostream& os, const void* p) {
  if (p == nullptr) {
    os << "(null)";
  } else {
    os << p;
  }
}

// Helper functions for string comparisons.
#define DEFINE_CHECK_STROP_IMPL(name, func, expected)                      \
  TString* Check##func##expected##Impl(const char* s1, const char* s2, \
                                           const char* exprtext) {         \
    bool equal = s1 == s2 || (s1 && s2 && !func(s1, s2));                  \
    if (equal == expected) {                                               \
      return nullptr;                                                      \
    } else {                                                               \
      return new TString(                                              \
          y_absl::StrCat(exprtext, " (", s1, " vs. ", s2, ")"));             \
    }                                                                      \
  }
DEFINE_CHECK_STROP_IMPL(CHECK_STREQ, strcmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRNE, strcmp, false)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASEEQ, strcasecmp, true)
DEFINE_CHECK_STROP_IMPL(CHECK_STRCASENE, strcasecmp, false)
#undef DEFINE_CHECK_STROP_IMPL

namespace detect_specialization {

StringifySink::StringifySink(std::ostream& os) : os_(os) {}

void StringifySink::Append(y_absl::string_view text) { os_ << text; }

void StringifySink::Append(size_t length, char ch) {
  for (size_t i = 0; i < length; ++i) os_.put(ch);
}

void AbslFormatFlush(StringifySink* sink, y_absl::string_view text) {
  sink->Append(text);
}

}  // namespace detect_specialization

}  // namespace log_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
