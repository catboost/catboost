// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "google/protobuf/compiler/python/helpers.h"

#include <algorithm>
#include <string>
#include <vector>

#include "y_absl/log/absl_check.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/str_replace.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/strings/strip.h"
#include "google/protobuf/compiler/code_generator.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"

namespace google {
namespace protobuf {
namespace compiler {
namespace python {

TProtoStringType FixEv(TProtoStringType filename) {
    constexpr auto kSuffixLen = 3;
    if (filename.EndsWith(".ev")) {
        return filename.substr(0, filename.length() - kSuffixLen) + "_ev.proto";
    }
    return filename;
}

// Returns the Python module name expected for a given .proto filename.
TProtoStringType ModuleName(y_absl::string_view filename) {
  TProtoStringType str(std::string{filename});
  TProtoStringType basename = StripProto(FixEv(str));
  y_absl::StrReplaceAll({{"-", "_"}, {"/", "."}}, &basename);
  return y_absl::StrCat(basename, "_pb2");
}

TProtoStringType StrippedModuleName(y_absl::string_view filename) {
  TProtoStringType module_name = ModuleName(filename);
  return module_name;
}

// Keywords reserved by the Python language.
const char* const kKeywords[] = {
    "False",  "None",     "True",  "and",    "as",       "assert",
    "async",  "await",    "break", "class",  "continue", "def",
    "del",    "elif",     "else",  "except", "finally",  "for",
    "from",   "global",   "if",    "import", "in",       "is",
    "lambda", "nonlocal", "not",   "or",     "pass",     "raise",
    "return", "try",      "while", "with",   "yield",
};
const char* const* kKeywordsEnd =
    kKeywords + (sizeof(kKeywords) / sizeof(kKeywords[0]));

bool ContainsPythonKeyword(y_absl::string_view module_name) {
  std::vector<y_absl::string_view> tokens = y_absl::StrSplit(module_name, '.');
  for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
    if (std::find(kKeywords, kKeywordsEnd, tokens[i]) != kKeywordsEnd) {
      return true;
    }
  }
  return false;
}

bool IsPythonKeyword(y_absl::string_view name) {
  return (std::find(kKeywords, kKeywordsEnd, name) != kKeywordsEnd);
}

TProtoStringType ResolveKeyword(y_absl::string_view name) {
  if (IsPythonKeyword(name)) {
    return y_absl::StrCat("globals()['", name, "']");
  }
  return TProtoStringType(name);
}

TProtoStringType GetFileName(const FileDescriptor* file_des,
                        y_absl::string_view suffix) {
  TProtoStringType module_name = ModuleName(file_des->name());
  TProtoStringType filename = module_name;
  y_absl::StrReplaceAll({{".", "/"}}, &filename);
  y_absl::StrAppend(&filename, suffix);
  return filename;
}

bool HasGenericServices(const FileDescriptor* file) {
  return file->service_count() > 0 && file->options().py_generic_services();
}

TProtoStringType GeneratedCodeToBase64(const GeneratedCodeInfo& annotations) {
  TProtoStringType result;
  y_absl::Base64Escape(annotations.SerializeAsString(), &result);
  return result;
}

template <typename DescriptorT>
TProtoStringType NamePrefixedWithNestedTypes(const DescriptorT& descriptor,
                                        y_absl::string_view separator) {
  TProtoStringType name = descriptor.name();
  const Descriptor* parent = descriptor.containing_type();
  if (parent != nullptr) {
    TProtoStringType prefix = NamePrefixedWithNestedTypes(*parent, separator);
    if (separator == "." && IsPythonKeyword(name)) {
      return y_absl::StrCat("getattr(", prefix, ", '", name, "')");
    } else {
      return y_absl::StrCat(prefix, separator, name);
    }
  }
  if (separator == ".") {
    name = ResolveKeyword(name);
  }
  return name;
}

template TProtoStringType NamePrefixedWithNestedTypes<Descriptor>(
    const Descriptor& descriptor, y_absl::string_view separator);
template TProtoStringType NamePrefixedWithNestedTypes<EnumDescriptor>(
    const EnumDescriptor& descriptor, y_absl::string_view separator);

}  // namespace python
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
