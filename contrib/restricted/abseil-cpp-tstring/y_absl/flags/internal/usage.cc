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

#include "y_absl/flags/internal/usage.h"

#include <stdint.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <ostream>
#include <util/generic/string.h>
#include <utility>
#include <vector>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/const_init.h"
#include "y_absl/base/thread_annotations.h"
#include "y_absl/flags/commandlineflag.h"
#include "y_absl/flags/flag.h"
#include "y_absl/flags/internal/flag.h"
#include "y_absl/flags/internal/path_util.h"
#include "y_absl/flags/internal/private_handle_accessor.h"
#include "y_absl/flags/internal/program_name.h"
#include "y_absl/flags/internal/registry.h"
#include "y_absl/flags/usage_config.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/strings/strip.h"
#include "y_absl/synchronization/mutex.h"

// Dummy global variables to prevent anyone else defining these.
bool FLAGS_help = false;
bool FLAGS_helpfull = false;
bool FLAGS_helpshort = false;
bool FLAGS_helppackage = false;
bool FLAGS_version = false;
bool FLAGS_only_check_args = false;
bool FLAGS_helpon = false;
bool FLAGS_helpmatch = false;

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace flags_internal {
namespace {

using PerFlagFilter = std::function<bool(const y_absl::CommandLineFlag&)>;

// Maximum length size in a human readable format.
constexpr size_t kHrfMaxLineLength = 80;

// This class is used to emit an XML element with `tag` and `text`.
// It adds opening and closing tags and escapes special characters in the text.
// For example:
// std::cout << XMLElement("title", "Milk & Cookies");
// prints "<title>Milk &amp; Cookies</title>"
class XMLElement {
 public:
  XMLElement(y_absl::string_view tag, y_absl::string_view txt)
      : tag_(tag), txt_(txt) {}

  friend std::ostream& operator<<(std::ostream& out,
                                  const XMLElement& xml_elem) {
    out << "<" << xml_elem.tag_ << ">";

    for (auto c : xml_elem.txt_) {
      switch (c) {
        case '"':
          out << "&quot;";
          break;
        case '\'':
          out << "&apos;";
          break;
        case '&':
          out << "&amp;";
          break;
        case '<':
          out << "&lt;";
          break;
        case '>':
          out << "&gt;";
          break;
        case '\n':
        case '\v':
        case '\f':
        case '\t':
          out << " ";
          break;
        default:
          if (IsValidXmlCharacter(static_cast<unsigned char>(c))) {
            out << c;
          }
          break;
      }
    }

    return out << "</" << xml_elem.tag_ << ">";
  }

 private:
  static bool IsValidXmlCharacter(unsigned char c) { return c >= 0x20; }
  y_absl::string_view tag_;
  y_absl::string_view txt_;
};

// --------------------------------------------------------------------
// Helper class to pretty-print info about a flag.

class FlagHelpPrettyPrinter {
 public:
  // Pretty printer holds on to the std::ostream& reference to direct an output
  // to that stream.
  FlagHelpPrettyPrinter(size_t max_line_len, size_t min_line_len,
                        size_t wrapped_line_indent, std::ostream& out)
      : out_(out),
        max_line_len_(max_line_len),
        min_line_len_(min_line_len),
        wrapped_line_indent_(wrapped_line_indent),
        line_len_(0),
        first_line_(true) {}

  void Write(y_absl::string_view str, bool wrap_line = false) {
    // Empty string - do nothing.
    if (str.empty()) return;

    std::vector<y_absl::string_view> tokens;
    if (wrap_line) {
      for (auto line : y_absl::StrSplit(str, y_absl::ByAnyChar("\n\r"))) {
        if (!tokens.empty()) {
          // Keep line separators in the input string.
          tokens.emplace_back("\n");
        }
        for (auto token :
             y_absl::StrSplit(line, y_absl::ByAnyChar(" \t"), y_absl::SkipEmpty())) {
          tokens.push_back(token);
        }
      }
    } else {
      tokens.push_back(str);
    }

    for (auto token : tokens) {
      bool new_line = (line_len_ == 0);

      // Respect line separators in the input string.
      if (token == "\n") {
        EndLine();
        continue;
      }

      // Write the token, ending the string first if necessary/possible.
      if (!new_line && (line_len_ + token.size() >= max_line_len_)) {
        EndLine();
        new_line = true;
      }

      if (new_line) {
        StartLine();
      } else {
        out_ << ' ';
        ++line_len_;
      }

      out_ << token;
      line_len_ += token.size();
    }
  }

  void StartLine() {
    if (first_line_) {
      line_len_ = min_line_len_;
      first_line_ = false;
    } else {
      line_len_ = min_line_len_ + wrapped_line_indent_;
    }
    out_ << TString(line_len_, ' ');
  }
  void EndLine() {
    out_ << '\n';
    line_len_ = 0;
  }

 private:
  std::ostream& out_;
  const size_t max_line_len_;
  const size_t min_line_len_;
  const size_t wrapped_line_indent_;
  size_t line_len_;
  bool first_line_;
};

void FlagHelpHumanReadable(const CommandLineFlag& flag, std::ostream& out) {
  FlagHelpPrettyPrinter printer(kHrfMaxLineLength, 4, 2, out);

  // Flag name.
  printer.Write(y_absl::StrCat("--", flag.Name()));

  // Flag help.
  printer.Write(y_absl::StrCat("(", flag.Help(), ");"), /*wrap_line=*/true);

  // The listed default value will be the actual default from the flag
  // definition in the originating source file, unless the value has
  // subsequently been modified using SetCommandLineOption() with mode
  // SET_FLAGS_DEFAULT.
  TString dflt_val = flag.DefaultValue();
  TString curr_val = flag.CurrentValue();
  bool is_modified = curr_val != dflt_val;

  if (flag.IsOfType<TString>()) {
    dflt_val = y_absl::StrCat("\"", dflt_val, "\"");
  }
  printer.Write(y_absl::StrCat("default: ", dflt_val, ";"));

  if (is_modified) {
    if (flag.IsOfType<TString>()) {
      curr_val = y_absl::StrCat("\"", curr_val, "\"");
    }
    printer.Write(y_absl::StrCat("currently: ", curr_val, ";"));
  }

  printer.EndLine();
}

// Shows help for every filename which matches any of the filters
// If filters are empty, shows help for every file.
// If a flag's help message has been stripped (e.g. by adding '#define
// STRIP_FLAG_HELP 1' then this flag will not be displayed by '--help'
// and its variants.
void FlagsHelpImpl(std::ostream& out, PerFlagFilter filter_cb,
                   HelpFormat format, y_absl::string_view program_usage_message) {
  if (format == HelpFormat::kHumanReadable) {
    out << flags_internal::ShortProgramInvocationName() << ": "
        << program_usage_message << "\n\n";
  } else {
    // XML schema is not a part of our public API for now.
    out << "<?xml version=\"1.0\"?>\n"
        << "<!-- This output should be used with care. We do not report type "
           "names for flags with user defined types -->\n"
        << "<!-- Prefer flag only_check_args for validating flag inputs -->\n"
        // The document.
        << "<AllFlags>\n"
        // The program name and usage.
        << XMLElement("program", flags_internal::ShortProgramInvocationName())
        << '\n'
        << XMLElement("usage", program_usage_message) << '\n';
  }

  // Ordered map of package name to
  //   map of file name to
  //     vector of flags in the file.
  // This map is used to output matching flags grouped by package and file
  // name.
  std::map<TString,
           std::map<TString, std::vector<const y_absl::CommandLineFlag*>>>
      matching_flags;

  flags_internal::ForEachFlag([&](y_absl::CommandLineFlag& flag) {
    // Ignore retired flags.
    if (flag.IsRetired()) return;

    // If the flag has been stripped, pretend that it doesn't exist.
    if (flag.Help() == flags_internal::kStrippedFlagHelp) return;

    // Make sure flag satisfies the filter
    if (!filter_cb(flag)) return;

    TString flag_filename = flag.Filename();

    matching_flags[TString(flags_internal::Package(flag_filename))]
                  [flag_filename]
                      .push_back(&flag);
  });

  y_absl::string_view package_separator;  // controls blank lines between packages
  y_absl::string_view file_separator;     // controls blank lines between files
  for (auto& package : matching_flags) {
    if (format == HelpFormat::kHumanReadable) {
      out << package_separator;
      package_separator = "\n\n";
    }

    file_separator = "";
    for (auto& flags_in_file : package.second) {
      if (format == HelpFormat::kHumanReadable) {
        out << file_separator << "  Flags from " << flags_in_file.first
            << ":\n";
        file_separator = "\n";
      }

      std::sort(std::begin(flags_in_file.second),
                std::end(flags_in_file.second),
                [](const CommandLineFlag* lhs, const CommandLineFlag* rhs) {
                  return lhs->Name() < rhs->Name();
                });

      for (const auto* flag : flags_in_file.second) {
        flags_internal::FlagHelp(out, *flag, format);
      }
    }
  }

  if (format == HelpFormat::kHumanReadable) {
    FlagHelpPrettyPrinter printer(kHrfMaxLineLength, 0, 0, out);

    if (filter_cb && matching_flags.empty()) {
      printer.Write("No flags matched.\n", true);
    }
    printer.EndLine();
    printer.Write(
        "Try --helpfull to get a list of all flags or --help=substring "
        "shows help for flags which include specified substring in either "
        "in the name, or description or path.\n",
        true);
  } else {
    // The end of the document.
    out << "</AllFlags>\n";
  }
}

void FlagsHelpImpl(std::ostream& out,
                   flags_internal::FlagKindFilter filename_filter_cb,
                   HelpFormat format, y_absl::string_view program_usage_message) {
  FlagsHelpImpl(
      out,
      [&](const y_absl::CommandLineFlag& flag) {
        return filename_filter_cb && filename_filter_cb(flag.Filename());
      },
      format, program_usage_message);
}

}  // namespace

// --------------------------------------------------------------------
// Produces the help message describing specific flag.
void FlagHelp(std::ostream& out, const CommandLineFlag& flag,
              HelpFormat format) {
  if (format == HelpFormat::kHumanReadable)
    flags_internal::FlagHelpHumanReadable(flag, out);
}

// --------------------------------------------------------------------
// Produces the help messages for all flags matching the filename filter.
// If filter is empty produces help messages for all flags.
void FlagsHelp(std::ostream& out, y_absl::string_view filter, HelpFormat format,
               y_absl::string_view program_usage_message) {
  flags_internal::FlagKindFilter filter_cb = [&](y_absl::string_view filename) {
    return filter.empty() || y_absl::StrContains(filename, filter);
  };
  flags_internal::FlagsHelpImpl(out, filter_cb, format, program_usage_message);
}

// --------------------------------------------------------------------
// Checks all the 'usage' command line flags to see if any have been set.
// If so, handles them appropriately.
HelpMode HandleUsageFlags(std::ostream& out,
                          y_absl::string_view program_usage_message) {
  switch (GetFlagsHelpMode()) {
    case HelpMode::kNone:
      break;
    case HelpMode::kImportant:
      flags_internal::FlagsHelpImpl(
          out, flags_internal::GetUsageConfig().contains_help_flags,
          GetFlagsHelpFormat(), program_usage_message);
      break;

    case HelpMode::kShort:
      flags_internal::FlagsHelpImpl(
          out, flags_internal::GetUsageConfig().contains_helpshort_flags,
          GetFlagsHelpFormat(), program_usage_message);
      break;

    case HelpMode::kFull:
      flags_internal::FlagsHelp(out, "", GetFlagsHelpFormat(),
                                program_usage_message);
      break;

    case HelpMode::kPackage:
      flags_internal::FlagsHelpImpl(
          out, flags_internal::GetUsageConfig().contains_helppackage_flags,
          GetFlagsHelpFormat(), program_usage_message);
      break;

    case HelpMode::kMatch: {
      TString substr = GetFlagsHelpMatchSubstr();
      if (substr.empty()) {
        // show all options
        flags_internal::FlagsHelp(out, substr, GetFlagsHelpFormat(),
                                  program_usage_message);
      } else {
        auto filter_cb = [&substr](const y_absl::CommandLineFlag& flag) {
          if (y_absl::StrContains(flag.Name(), substr)) return true;
          if (y_absl::StrContains(flag.Filename(), substr)) return true;
          if (y_absl::StrContains(flag.Help(), substr)) return true;

          return false;
        };
        flags_internal::FlagsHelpImpl(
            out, filter_cb, HelpFormat::kHumanReadable, program_usage_message);
      }
      break;
    }
    case HelpMode::kVersion:
      if (flags_internal::GetUsageConfig().version_string)
        out << flags_internal::GetUsageConfig().version_string();
      // Unlike help, we may be asking for version in a script, so return 0
      break;

    case HelpMode::kOnlyCheckArgs:
      break;
  }

  return GetFlagsHelpMode();
}

// --------------------------------------------------------------------
// Globals representing usage reporting flags

namespace {

Y_ABSL_CONST_INIT y_absl::Mutex help_attributes_guard(y_absl::kConstInit);
Y_ABSL_CONST_INIT TString* match_substr
    Y_ABSL_GUARDED_BY(help_attributes_guard) = nullptr;
Y_ABSL_CONST_INIT HelpMode help_mode Y_ABSL_GUARDED_BY(help_attributes_guard) =
    HelpMode::kNone;
Y_ABSL_CONST_INIT HelpFormat help_format Y_ABSL_GUARDED_BY(help_attributes_guard) =
    HelpFormat::kHumanReadable;

}  // namespace

TString GetFlagsHelpMatchSubstr() {
  y_absl::MutexLock l(&help_attributes_guard);
  if (match_substr == nullptr) return "";
  return *match_substr;
}

void SetFlagsHelpMatchSubstr(y_absl::string_view substr) {
  y_absl::MutexLock l(&help_attributes_guard);
  if (match_substr == nullptr) match_substr = new TString;
  match_substr->assign(substr.data(), substr.size());
}

HelpMode GetFlagsHelpMode() {
  y_absl::MutexLock l(&help_attributes_guard);
  return help_mode;
}

void SetFlagsHelpMode(HelpMode mode) {
  y_absl::MutexLock l(&help_attributes_guard);
  help_mode = mode;
}

HelpFormat GetFlagsHelpFormat() {
  y_absl::MutexLock l(&help_attributes_guard);
  return help_format;
}

void SetFlagsHelpFormat(HelpFormat format) {
  y_absl::MutexLock l(&help_attributes_guard);
  help_format = format;
}

// Deduces usage flags from the input argument in a form --name=value or
// --name. argument is already split into name and value before we call this
// function.
bool DeduceUsageFlags(y_absl::string_view name, y_absl::string_view value) {
  if (y_absl::ConsumePrefix(&name, "help")) {
    if (name.empty()) {
      if (value.empty()) {
        SetFlagsHelpMode(HelpMode::kImportant);
      } else {
        SetFlagsHelpMode(HelpMode::kMatch);
        SetFlagsHelpMatchSubstr(value);
      }
      return true;
    }

    if (name == "match") {
      SetFlagsHelpMode(HelpMode::kMatch);
      SetFlagsHelpMatchSubstr(value);
      return true;
    }

    if (name == "on") {
      SetFlagsHelpMode(HelpMode::kMatch);
      SetFlagsHelpMatchSubstr(y_absl::StrCat("/", value, "."));
      return true;
    }

    if (name == "full") {
      SetFlagsHelpMode(HelpMode::kFull);
      return true;
    }

    if (name == "short") {
      SetFlagsHelpMode(HelpMode::kShort);
      return true;
    }

    if (name == "package") {
      SetFlagsHelpMode(HelpMode::kPackage);
      return true;
    }

    return false;
  }

  if (name == "version") {
    SetFlagsHelpMode(HelpMode::kVersion);
    return true;
  }

  if (name == "only_check_args") {
    SetFlagsHelpMode(HelpMode::kOnlyCheckArgs);
    return true;
  }

  return false;
}

// --------------------------------------------------------------------

void MaybeExit(HelpMode mode) {
  switch (mode) {
    case flags_internal::HelpMode::kNone:
      return;
    case flags_internal::HelpMode::kOnlyCheckArgs:
    case flags_internal::HelpMode::kVersion:
      std::exit(0);
    default:  // For all the other modes we exit with 1
      std::exit(1);
  }
}

// --------------------------------------------------------------------

}  // namespace flags_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl
