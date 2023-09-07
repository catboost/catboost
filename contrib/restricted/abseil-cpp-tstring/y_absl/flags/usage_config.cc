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

#include "y_absl/flags/usage_config.h"

#include <functional>
#include <iostream>
#include <util/generic/string.h>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/const_init.h"
#include "y_absl/base/thread_annotations.h"
#include "y_absl/flags/internal/path_util.h"
#include "y_absl/flags/internal/program_name.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/strings/strip.h"
#include "y_absl/synchronization/mutex.h"

extern "C" {

// Additional report of fatal usage error message before we std::exit. Error is
// fatal if is_fatal argument to ReportUsageError is true.
Y_ABSL_ATTRIBUTE_WEAK void Y_ABSL_INTERNAL_C_SYMBOL(
    AbslInternalReportFatalUsageError)(y_absl::string_view) {}

}  // extern "C"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace flags_internal {

namespace {

// --------------------------------------------------------------------
// Returns true if flags defined in the filename should be reported with
// -helpshort flag.

bool ContainsHelpshortFlags(y_absl::string_view filename) {
  // By default we only want flags in binary's main. We expect the main
  // routine to reside in <program>.cc or <program>-main.cc or
  // <program>_main.cc, where the <program> is the name of the binary
  // (without .exe on Windows).
  auto suffix = flags_internal::Basename(filename);
  auto program_name = flags_internal::ShortProgramInvocationName();
  y_absl::string_view program_name_ref = program_name;
#if defined(_WIN32)
  y_absl::ConsumeSuffix(&program_name_ref, ".exe");
#endif
  if (!y_absl::ConsumePrefix(&suffix, program_name_ref))
    return false;
  return y_absl::StartsWith(suffix, ".") || y_absl::StartsWith(suffix, "-main.") ||
         y_absl::StartsWith(suffix, "_main.");
}

// --------------------------------------------------------------------
// Returns true if flags defined in the filename should be reported with
// -helppackage flag.

bool ContainsHelppackageFlags(y_absl::string_view filename) {
  // TODO(rogeeff): implement properly when registry is available.
  return ContainsHelpshortFlags(filename);
}

// --------------------------------------------------------------------
// Generates program version information into supplied output.

TString VersionString() {
  TString version_str(flags_internal::ShortProgramInvocationName());

  version_str += "\n";

#if !defined(NDEBUG)
  version_str += "Debug build (NDEBUG not #defined)\n";
#endif

  return version_str;
}

// --------------------------------------------------------------------
// Normalizes the filename specific to the build system/filesystem used.

TString NormalizeFilename(y_absl::string_view filename) {
  // Skip any leading slashes
  auto pos = filename.find_first_not_of("\\/");
  if (pos == y_absl::string_view::npos) return "";

  filename.remove_prefix(pos);
  return TString(filename);
}

// --------------------------------------------------------------------

Y_ABSL_CONST_INIT y_absl::Mutex custom_usage_config_guard(y_absl::kConstInit);
Y_ABSL_CONST_INIT FlagsUsageConfig* custom_usage_config
    Y_ABSL_GUARDED_BY(custom_usage_config_guard) = nullptr;

}  // namespace

FlagsUsageConfig GetUsageConfig() {
  y_absl::MutexLock l(&custom_usage_config_guard);

  if (custom_usage_config) return *custom_usage_config;

  FlagsUsageConfig default_config;
  default_config.contains_helpshort_flags = &ContainsHelpshortFlags;
  default_config.contains_help_flags = &ContainsHelppackageFlags;
  default_config.contains_helppackage_flags = &ContainsHelppackageFlags;
  default_config.version_string = &VersionString;
  default_config.normalize_filename = &NormalizeFilename;

  return default_config;
}

void ReportUsageError(y_absl::string_view msg, bool is_fatal) {
  std::cerr << "ERROR: " << msg << std::endl;

  if (is_fatal) {
    Y_ABSL_INTERNAL_C_SYMBOL(AbslInternalReportFatalUsageError)(msg);
  }
}

}  // namespace flags_internal

void SetFlagsUsageConfig(FlagsUsageConfig usage_config) {
  y_absl::MutexLock l(&flags_internal::custom_usage_config_guard);

  if (!usage_config.contains_helpshort_flags)
    usage_config.contains_helpshort_flags =
        flags_internal::ContainsHelpshortFlags;

  if (!usage_config.contains_help_flags)
    usage_config.contains_help_flags = flags_internal::ContainsHelppackageFlags;

  if (!usage_config.contains_helppackage_flags)
    usage_config.contains_helppackage_flags =
        flags_internal::ContainsHelppackageFlags;

  if (!usage_config.version_string)
    usage_config.version_string = flags_internal::VersionString;

  if (!usage_config.normalize_filename)
    usage_config.normalize_filename = flags_internal::NormalizeFilename;

  if (flags_internal::custom_usage_config)
    *flags_internal::custom_usage_config = usage_config;
  else
    flags_internal::custom_usage_config = new FlagsUsageConfig(usage_config);
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl
