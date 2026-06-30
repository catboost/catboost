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

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include "google/protobuf/compiler/cpp/field.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "y_absl/container/flat_hash_map.h"
#include "y_absl/log/absl_check.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/types/optional.h"
#include "google/protobuf/compiler/cpp/field_generators/generators.h"
#include "google/protobuf/compiler/cpp/helpers.h"
#include "google/protobuf/compiler/cpp/options.h"
#include "google/protobuf/compiler/cpp/tracker.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/io/printer.h"
#include "google/protobuf/wire_format.h"

namespace google {
namespace protobuf {
namespace compiler {
namespace cpp {
using ::google::protobuf::internal::WireFormat;
using Sub = ::google::protobuf::io::Printer::Sub;

std::vector<Sub> FieldVars(const FieldDescriptor* field, const Options& opts) {
  bool split = ShouldSplit(field, opts);
  std::vector<Sub> vars = {
      // This will eventually be renamed to "field", once the existing "field"
      // variable is replaced with "field_" everywhere.
      {"name", FieldName(field)},

      {"index", field->index()},
      {"number", field->number()},
      {"pkg.Msg.field", field->full_name()},

      {"field_", FieldMemberName(field, split)},
      {"DeclaredType", DeclaredTypeMethodName(field->type())},
      {"kTagBytes", WireFormat::TagSize(field->number(), field->type())},
      Sub("PrepareSplitMessageForWrite",
          split ? "PrepareSplitMessageForWrite();" : "")
          .WithSuffix(";"),
      Sub("DEPRECATED", DeprecatedAttribute(opts, field)).WithSuffix(" "),

      // These variables are placeholders to pick out the beginning and ends of
      // identifiers for annotations (when doing so with existing variables
      // would be ambiguous or impossible). They should never be set to anything
      // but the empty string.
      {"{", ""},
      {"}", ""},

      // Old-style names.
      {"field", FieldMemberName(field, split)},
      {"maybe_prepare_split_message",
       split ? "PrepareSplitMessageForWrite();" : ""},
      {"declared_type", DeclaredTypeMethodName(field->type())},
      {"classname", ClassName(FieldScope(field), false)},
      {"ns", Namespace(field, opts)},
      {"tag_size", WireFormat::TagSize(field->number(), field->type())},
      {"deprecated_attr", DeprecatedAttribute(opts, field)},
  };

  if (const auto* oneof = field->containing_oneof()) {
    auto field_name = UnderscoresToCamelCase(field->name(), true);

    vars.push_back({"oneof_name", oneof->name()});
    vars.push_back({"field_name", field_name});
    vars.push_back({"oneof_index", oneof->index()});
    vars.push_back({"has_field", y_absl::StrFormat("%s_case() == k%s",
                                                 oneof->name(), field_name)});
    vars.push_back(
        {"not_has_field",
         y_absl::StrFormat("%s_case() != k%s", oneof->name(), field_name)});
  }

  return vars;
}

void FieldGeneratorBase::GenerateAggregateInitializer(io::Printer* p) const {
  Formatter format(p, variables_);
  if (ShouldSplit(descriptor_, options_)) {
    format("decltype(Impl_::Split::$name$_){arena}");
    return;
  }
  format("decltype($field$){arena}");
}

void FieldGeneratorBase::GenerateConstexprAggregateInitializer(
    io::Printer* p) const {
  Formatter format(p, variables_);
  format("/*decltype($field$)*/{}");
}

void FieldGeneratorBase::GenerateCopyAggregateInitializer(
    io::Printer* p) const {
  Formatter format(p, variables_);
  format("decltype($field$){from.$field$}");
}

void FieldGeneratorBase::GenerateCopyConstructorCode(io::Printer* p) const {
  if (ShouldSplit(descriptor_, options_)) {
    // There is no copy constructor for the `Split` struct, so we need to copy
    // the value here.
    Formatter format(p, variables_);
    format("$field$ = from.$field$;\n");
  }
}

void FieldGeneratorBase::GenerateIfHasField(io::Printer* p) const {
  Y_ABSL_CHECK(internal::cpp::HasHasbit(descriptor_));

  Formatter format(p);
  format("if (($has_hasbit$) != 0) {\n");
}

namespace {
std::unique_ptr<FieldGeneratorBase> MakeGenerator(const FieldDescriptor* field,
                                                  const Options& options,
                                                  MessageSCCAnalyzer* scc) {

  if (field->is_map()) {
    return MakeMapGenerator(field, options, scc);
  }
  if (field->is_repeated()) {
    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_MESSAGE:
        return MakeRepeatedMessageGenerator(field, options, scc);
      case FieldDescriptor::CPPTYPE_STRING:
        return MakeRepeatedStringGenerator(field, options, scc);
      case FieldDescriptor::CPPTYPE_ENUM:
        return MakeRepeatedEnumGenerator(field, options, scc);
      default:
        return MakeRepeatedPrimitiveGenerator(field, options, scc);
    }
  }

  if (field->real_containing_oneof() &&
      field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
    return MakeOneofMessageGenerator(field, options, scc);
  }

  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_MESSAGE:
      return MakeSinguarMessageGenerator(field, options, scc);
    case FieldDescriptor::CPPTYPE_STRING:
      return MakeSinguarStringGenerator(field, options, scc);
    case FieldDescriptor::CPPTYPE_ENUM:
      return MakeSinguarEnumGenerator(field, options, scc);
    default:
      return MakeSinguarPrimitiveGenerator(field, options, scc);
  }
}

void HasBitVars(const FieldDescriptor* field, const Options& opts,
                y_absl::optional<arc_ui32> idx, std::vector<Sub>& vars) {
  if (!idx.has_value()) {
    vars.emplace_back("set_hasbit", "");
    vars.emplace_back("clear_hasbit", "");
    return;
  }

  Y_ABSL_CHECK(internal::cpp::HasHasbit(field));

  arc_i32 index = *idx / 32;
  TProtoStringType mask = y_absl::StrFormat("0x%08xu", 1u << (*idx % 32));

  y_absl::string_view has_bits = IsMapEntryMessage(field->containing_type())
                                   ? "_has_bits_"
                                   : "_impl_._has_bits_";

  auto has = y_absl::StrFormat("%s[%d] & %s", has_bits, index, mask);
  auto set = y_absl::StrFormat("%s[%d] |= %s;", has_bits, index, mask);
  auto clr = y_absl::StrFormat("%s[%d] &= ~%s;", has_bits, index, mask);

  vars.emplace_back("has_hasbit", has);
  vars.emplace_back(Sub("set_hasbit", set).WithSuffix(";"));
  vars.emplace_back(Sub("clear_hasbit", clr).WithSuffix(";"));
}

void InlinedStringVars(const FieldDescriptor* field, const Options& opts,
                       y_absl::optional<arc_ui32> idx, std::vector<Sub>& vars) {
  if (!IsStringInlined(field, opts)) {
    Y_ABSL_CHECK(!idx.has_value());
    return;
  }

  // The first bit is the tracking bit for on demand registering ArenaDtor.
  Y_ABSL_CHECK_GT(*idx, 0)
      << "_inlined_string_donated_'s bit 0 is reserved for arena dtor tracking";

  arc_i32 index = *idx / 32;
  TProtoStringType mask = y_absl::StrFormat("0x%08xu", 1u << (*idx % 32));

  y_absl::string_view array = IsMapEntryMessage(field->containing_type())
                                ? "_inlined_string_donated_"
                                : "_impl_._inlined_string_donated_";

  vars.emplace_back("inlined_string_donated",
                    y_absl::StrFormat("(%s[%d] & %s) != 0;", array, index, mask));
  vars.emplace_back("donating_states_word",
                    y_absl::StrFormat("%s[%d]", array, index));
  vars.emplace_back("mask_for_undonate", y_absl::StrFormat("~%s", mask));
}
}  // namespace

FieldGenerator::FieldGenerator(const FieldDescriptor* field,
                               const Options& options,
                               MessageSCCAnalyzer* scc_analyzer,
                               y_absl::optional<arc_ui32> hasbit_index,
                               y_absl::optional<arc_ui32> inlined_string_index)
    : impl_(MakeGenerator(field, options, scc_analyzer)),
      field_vars_(FieldVars(field, options)),
      tracker_vars_(MakeTrackerCalls(field, options)),
      per_generator_vars_(impl_->MakeVars()) {
  HasBitVars(field, options, hasbit_index, field_vars_);
  InlinedStringVars(field, options, inlined_string_index, field_vars_);
}

void FieldGeneratorTable::Build(
    const Options& options, MessageSCCAnalyzer* scc,
    y_absl::Span<const arc_i32> has_bit_indices,
    y_absl::Span<const arc_i32> inlined_string_indices) {
  // Construct all the FieldGenerators.
  fields_.reserve(descriptor_->field_count());
  for (const auto* field : internal::FieldRange(descriptor_)) {
    y_absl::optional<arc_ui32> has_bit_index;
    if (!has_bit_indices.empty() && has_bit_indices[field->index()] >= 0) {
      has_bit_index = static_cast<arc_ui32>(has_bit_indices[field->index()]);
    }

    y_absl::optional<arc_ui32> inlined_string_index;
    if (!inlined_string_indices.empty() &&
        inlined_string_indices[field->index()] >= 0) {
      inlined_string_index =
          static_cast<arc_ui32>(inlined_string_indices[field->index()]);
    }

    fields_.push_back(FieldGenerator(field, options, scc, has_bit_index,
                                     inlined_string_index));
  }
}

}  // namespace cpp
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
