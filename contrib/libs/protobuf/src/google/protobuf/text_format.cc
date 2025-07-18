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

// Author: jschorr@google.com (Joseph Schorr)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include "google/protobuf/text_format.h"

#include <float.h>
#include <stdio.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "y_absl/container/btree_set.h"
#include "y_absl/memory/memory.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/numbers.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_join.h"
#include "y_absl/strings/string_view.h"
#include "google/protobuf/any.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/strtod.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/map_field.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/unknown_field_set.h"
#include "google/protobuf/wire_format_lite.h"

// Must be included last.
#include "google/protobuf/port_def.inc"

namespace google {
namespace protobuf {

namespace {

inline bool IsHexNumber(const TProtoStringType& str) {
  return (str.length() >= 2 && str[0] == '0' &&
          (str[1] == 'x' || str[1] == 'X'));
}

inline bool IsOctNumber(const TProtoStringType& str) {
  return (str.length() >= 2 && str[0] == '0' &&
          (str[1] >= '0' && str[1] < '8'));
}

}  // namespace

namespace internal {
const char kDebugStringSilentMarker[] = "";
const char kDebugStringSilentMarkerForDetection[] = "\t ";

// Controls insertion of a marker making debug strings non-parseable.
PROTOBUF_EXPORT std::atomic<bool> enable_debug_text_redaction_marker;

// Controls insertion of a randomized marker in debug strings.
PROTOBUF_EXPORT std::atomic<bool> enable_debug_text_random_marker;

// Controls insertion of kDebugStringSilentMarker.
PROTOBUF_EXPORT std::atomic<bool> enable_debug_text_format_marker;
}  // namespace internal

TProtoStringType Message::DebugString() const {
  TProtoStringType debug_string;

  TextFormat::Printer printer;
  printer.SetExpandAny(true);
  printer.SetInsertSilentMarker(internal::enable_debug_text_format_marker.load(
      std::memory_order_relaxed));
  printer.SetRedactDebugString(
      internal::enable_debug_text_redaction_marker.load(
          std::memory_order_relaxed));
  printer.SetRandomizeDebugString(
      internal::enable_debug_text_random_marker.load(
          std::memory_order_relaxed));

  printer.PrintToString(*this, &debug_string);

  return debug_string;
}

TProtoStringType Message::ShortDebugString() const {
  TProtoStringType debug_string;

  TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.SetExpandAny(true);
  printer.SetInsertSilentMarker(internal::enable_debug_text_format_marker.load(
      std::memory_order_relaxed));
  printer.SetRedactDebugString(
      internal::enable_debug_text_redaction_marker.load(
          std::memory_order_relaxed));
  printer.SetRandomizeDebugString(
      internal::enable_debug_text_random_marker.load(
          std::memory_order_relaxed));

  printer.PrintToString(*this, &debug_string);
  // Single line mode currently might have an extra space at the end.
  if (!debug_string.empty() && debug_string[debug_string.size() - 1] == ' ') {
    debug_string.resize(debug_string.size() - 1);
  }

  return debug_string;
}

TProtoStringType Message::Utf8DebugString() const {
  TProtoStringType debug_string;

  TextFormat::Printer printer;
  printer.SetUseUtf8StringEscaping(true);
  printer.SetExpandAny(true);
  printer.SetInsertSilentMarker(internal::enable_debug_text_format_marker.load(
      std::memory_order_relaxed));
  printer.SetRedactDebugString(
      internal::enable_debug_text_redaction_marker.load(
          std::memory_order_relaxed));
  printer.SetRandomizeDebugString(
      internal::enable_debug_text_random_marker.load(
          std::memory_order_relaxed));

  printer.PrintToString(*this, &debug_string);

  return debug_string;
}

void Message::PrintDebugString() const { printf("%s", DebugString().c_str()); }

namespace internal {

void PerformAbslStringify(const Message& message,
                          y_absl::FunctionRef<void(y_absl::string_view)> append) {
  // TODO(b/249835002): consider using the single line version for short
  TextFormat::Printer printer;
  printer.SetExpandAny(true);
  printer.SetRedactDebugString(true);
  printer.SetRandomizeDebugString(true);
  TProtoStringType result;
  printer.PrintToString(message, &result);
  append(result);
}

}  // namespace internal


// ===========================================================================
// Implementation of the parse information tree class.
void TextFormat::ParseInfoTree::RecordLocation(
    const FieldDescriptor* field, TextFormat::ParseLocationRange range) {
  locations_[field].push_back(range);
}

TextFormat::ParseInfoTree* TextFormat::ParseInfoTree::CreateNested(
    const FieldDescriptor* field) {
  // Owned by us in the map.
  auto& vec = nested_[field];
  vec.emplace_back(new TextFormat::ParseInfoTree());
  return vec.back().get();
}

void CheckFieldIndex(const FieldDescriptor* field, int index) {
  if (field == nullptr) {
    return;
  }

  if (field->is_repeated() && index == -1) {
    Y_ABSL_DLOG(FATAL) << "Index must be in range of repeated field values. "
                     << "Field: " << field->name();
  } else if (!field->is_repeated() && index != -1) {
    Y_ABSL_DLOG(FATAL) << "Index must be -1 for singular fields."
                     << "Field: " << field->name();
  }
}

TextFormat::ParseLocationRange TextFormat::ParseInfoTree::GetLocationRange(
    const FieldDescriptor* field, int index) const {
  CheckFieldIndex(field, index);
  if (index == -1) {
    index = 0;
  }

  auto it = locations_.find(field);
  if (it == locations_.end() ||
      index >= static_cast<arc_i64>(it->second.size())) {
    return TextFormat::ParseLocationRange();
  }

  return it->second[static_cast<size_t>(index)];
}

TextFormat::ParseInfoTree* TextFormat::ParseInfoTree::GetTreeForNested(
    const FieldDescriptor* field, int index) const {
  CheckFieldIndex(field, index);
  if (index == -1) {
    index = 0;
  }

  auto it = nested_.find(field);
  if (it == nested_.end() || index >= static_cast<arc_i64>(it->second.size())) {
    return nullptr;
  }

  return it->second[static_cast<size_t>(index)].get();
}

namespace {
// These functions implement the behavior of the "default" TextFormat::Finder,
// they are defined as standalone to be called when finder_ is nullptr.
const FieldDescriptor* DefaultFinderFindExtension(Message* message,
                                                  const TProtoStringType& name) {
  const Descriptor* descriptor = message->GetDescriptor();
  return descriptor->file()->pool()->FindExtensionByPrintableName(descriptor,
                                                                  name);
}

const FieldDescriptor* DefaultFinderFindExtensionByNumber(
    const Descriptor* descriptor, int number) {
  return descriptor->file()->pool()->FindExtensionByNumber(descriptor, number);
}

const Descriptor* DefaultFinderFindAnyType(const Message& message,
                                           const TProtoStringType& prefix,
                                           const TProtoStringType& name) {
  if (prefix != internal::kTypeGoogleApisComPrefix &&
      prefix != internal::kTypeGoogleProdComPrefix) {
    return nullptr;
  }
  return message.GetDescriptor()->file()->pool()->FindMessageTypeByName(name);
}
}  // namespace

// ===========================================================================
// Internal class for parsing an ASCII representation of a Protocol Message.
// This class makes use of the Protocol Message compiler's tokenizer found
// in //third_party/protobuf/io/tokenizer.h. Note that class's Parse
// method is *not* thread-safe and should only be used in a single thread at
// a time.

// Makes code slightly more readable.  The meaning of "DO(foo)" is
// "Execute foo and fail if it fails.", where failure is indicated by
// returning false. Borrowed from parser.cc (Thanks Kenton!).
#define DO(STATEMENT) \
  if (STATEMENT) {    \
  } else {            \
    return false;     \
  }

class TextFormat::Parser::ParserImpl {
 public:
  // Determines if repeated values for non-repeated fields and
  // oneofs are permitted, e.g., the string "foo: 1 foo: 2" for a
  // required/optional field named "foo", or "baz: 1 bar: 2"
  // where "baz" and "bar" are members of the same oneof.
  enum SingularOverwritePolicy {
    ALLOW_SINGULAR_OVERWRITES = 0,   // the last value is retained
    FORBID_SINGULAR_OVERWRITES = 1,  // an error is issued
  };

  ParserImpl(const Descriptor* root_message_type,
             io::ZeroCopyInputStream* input_stream,
             io::ErrorCollector* error_collector,
             const TextFormat::Finder* finder, ParseInfoTree* parse_info_tree,
             SingularOverwritePolicy singular_overwrite_policy,
             bool allow_case_insensitive_field, bool allow_unknown_field,
             bool allow_unknown_extension, bool allow_unknown_enum,
             bool allow_field_number, bool allow_relaxed_whitespace,
             bool allow_partial, int recursion_limit)
      : error_collector_(error_collector),
        finder_(finder),
        parse_info_tree_(parse_info_tree),
        tokenizer_error_collector_(this),
        tokenizer_(input_stream, &tokenizer_error_collector_),
        root_message_type_(root_message_type),
        singular_overwrite_policy_(singular_overwrite_policy),
        allow_case_insensitive_field_(allow_case_insensitive_field),
        allow_unknown_field_(allow_unknown_field),
        allow_unknown_extension_(allow_unknown_extension),
        allow_unknown_enum_(allow_unknown_enum),
        allow_field_number_(allow_field_number),
        allow_partial_(allow_partial),
        initial_recursion_limit_(recursion_limit),
        recursion_limit_(recursion_limit),
        had_silent_marker_(false),
        had_errors_(false) {
    // For backwards-compatibility with proto1, we need to allow the 'f' suffix
    // for floats.
    tokenizer_.set_allow_f_after_float(true);

    // '#' starts a comment.
    tokenizer_.set_comment_style(io::Tokenizer::SH_COMMENT_STYLE);

    if (allow_relaxed_whitespace) {
      tokenizer_.set_require_space_after_number(false);
      tokenizer_.set_allow_multiline_strings(true);
    }

    // Consume the starting token.
    tokenizer_.Next();
  }
  ParserImpl(const ParserImpl&) = delete;
  ParserImpl& operator=(const ParserImpl&) = delete;
  ~ParserImpl() {}

  // Parses the ASCII representation specified in input and saves the
  // information into the output pointer (a Message). Returns
  // false if an error occurs (an error will also be logged to
  // Y_ABSL_LOG(ERROR)).
  bool Parse(Message* output) {
    // Consume fields until we cannot do so anymore.
    while (true) {
      if (LookingAtType(io::Tokenizer::TYPE_END)) {
        // Ensures recursion limit properly unwinded, but only for success
        // cases. This implicitly avoids the check when `Parse` returns false
        // via `DO(...)`.
        Y_ABSL_DCHECK(had_errors_ || recursion_limit_ == initial_recursion_limit_)
            << "Recursion limit at end of parse should be "
            << initial_recursion_limit_ << ", but was " << recursion_limit_
            << ". Difference of " << initial_recursion_limit_ - recursion_limit_
            << " stack frames not accounted for stack unwind.";

        return !had_errors_;
      }

      DO(ConsumeField(output));
    }
  }

  bool ParseField(const FieldDescriptor* field, Message* output) {
    bool suc;
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      suc = ConsumeFieldMessage(output, output->GetReflection(), field);
    } else {
      suc = ConsumeFieldValue(output, output->GetReflection(), field);
    }
    return suc && LookingAtType(io::Tokenizer::TYPE_END);
  }

  void ReportError(int line, int col, y_absl::string_view message) {
    had_errors_ = true;
    if (error_collector_ == nullptr) {
      if (line >= 0) {
        Y_ABSL_LOG(ERROR) << "Error parsing text-format "
                        << root_message_type_->full_name() << ": " << (line + 1)
                        << ":" << (col + 1) << ": " << message;
      } else {
        Y_ABSL_LOG(ERROR) << "Error parsing text-format "
                        << root_message_type_->full_name() << ": " << message;
      }
    } else {
      error_collector_->RecordError(line, col, message);
    }
  }

  void ReportWarning(int line, int col, const y_absl::string_view message) {
    if (error_collector_ == nullptr) {
      if (line >= 0) {
        Y_ABSL_LOG(WARNING) << "Warning parsing text-format "
                          << root_message_type_->full_name() << ": "
                          << (line + 1) << ":" << (col + 1) << ": " << message;
      } else {
        Y_ABSL_LOG(WARNING) << "Warning parsing text-format "
                          << root_message_type_->full_name() << ": " << message;
      }
    } else {
      error_collector_->RecordWarning(line, col, message);
    }
  }

 private:
  static constexpr arc_i32 kint32max = std::numeric_limits<arc_i32>::max();
  static constexpr arc_ui32 kuint32max = std::numeric_limits<arc_ui32>::max();
  static constexpr arc_i64 kint64min = std::numeric_limits<arc_i64>::min();
  static constexpr arc_i64 kint64max = std::numeric_limits<arc_i64>::max();
  static constexpr arc_ui64 kuint64max = std::numeric_limits<arc_ui64>::max();

  // Reports an error with the given message with information indicating
  // the position (as derived from the current token).
  void ReportError(y_absl::string_view message) {
    ReportError(tokenizer_.current().line, tokenizer_.current().column,
                message);
  }

  // Reports a warning with the given message with information indicating
  // the position (as derived from the current token).
  void ReportWarning(y_absl::string_view message) {
    ReportWarning(tokenizer_.current().line, tokenizer_.current().column,
                  message);
  }

  // Consumes the specified message with the given starting delimiter.
  // This method checks to see that the end delimiter at the conclusion of
  // the consumption matches the starting delimiter passed in here.
  bool ConsumeMessage(Message* message, const TProtoStringType delimiter) {
    while (!LookingAt(">") && !LookingAt("}")) {
      DO(ConsumeField(message));
    }

    // Confirm that we have a valid ending delimiter.
    DO(Consume(delimiter));
    return true;
  }

  // Consume either "<" or "{".
  bool ConsumeMessageDelimiter(TProtoStringType* delimiter) {
    if (TryConsume("<")) {
      *delimiter = ">";
    } else {
      DO(Consume("{"));
      *delimiter = "}";
    }
    return true;
  }


  // Consumes the current field (as returned by the tokenizer) on the
  // passed in message.
  bool ConsumeField(Message* message) {
    const Reflection* reflection = message->GetReflection();
    const Descriptor* descriptor = message->GetDescriptor();

    TProtoStringType field_name;
    bool reserved_field = false;
    const FieldDescriptor* field = nullptr;
    int start_line = tokenizer_.current().line;
    int start_column = tokenizer_.current().column;

    const FieldDescriptor* any_type_url_field;
    const FieldDescriptor* any_value_field;
    if (internal::GetAnyFieldDescriptors(*message, &any_type_url_field,
                                         &any_value_field) &&
        TryConsume("[")) {
      TProtoStringType full_type_name, prefix;
      DO(ConsumeAnyTypeUrl(&full_type_name, &prefix));
      TProtoStringType prefix_and_full_type_name =
          y_absl::StrCat(prefix, full_type_name);
      DO(ConsumeBeforeWhitespace("]"));
      TryConsumeWhitespace();
      // ':' is optional between message labels and values.
      if (TryConsumeBeforeWhitespace(":")) {
        TryConsumeWhitespace();
      }
      TProtoStringType serialized_value;
      const Descriptor* value_descriptor =
          finder_ ? finder_->FindAnyType(*message, prefix, full_type_name)
                  : DefaultFinderFindAnyType(*message, prefix, full_type_name);
      if (value_descriptor == nullptr) {
        ReportError(y_absl::StrCat("Could not find type \"",
                                 prefix_and_full_type_name,
                                 "\" stored in google.protobuf.Any."));
        return false;
      }
      DO(ConsumeAnyValue(value_descriptor, &serialized_value));
      if (singular_overwrite_policy_ == FORBID_SINGULAR_OVERWRITES) {
        // Fail if any_type_url_field has already been specified.
        if ((!any_type_url_field->is_repeated() &&
             reflection->HasField(*message, any_type_url_field)) ||
            (!any_value_field->is_repeated() &&
             reflection->HasField(*message, any_value_field))) {
          ReportError("Non-repeated Any specified multiple times.");
          return false;
        }
      }
      reflection->SetString(message, any_type_url_field,
                            std::move(prefix_and_full_type_name));
      reflection->SetString(message, any_value_field,
                            std::move(serialized_value));
      return true;
    }
    if (TryConsume("[")) {
      // Extension.
      DO(ConsumeFullTypeName(&field_name));
      DO(ConsumeBeforeWhitespace("]"));
      TryConsumeWhitespace();

      field = finder_ ? finder_->FindExtension(message, field_name)
                      : DefaultFinderFindExtension(message, field_name);

      if (field == nullptr) {
        if (!allow_unknown_field_ && !allow_unknown_extension_) {
          ReportError(y_absl::StrCat("Extension \"", field_name,
                                   "\" is not defined or "
                                   "is not an extension of \"",
                                   descriptor->full_name(), "\"."));
          return false;
        } else {
          ReportWarning(y_absl::StrCat(
              "Ignoring extension \"", field_name,
              "\" which is not defined or is not an extension of \"",
              descriptor->full_name(), "\"."));
        }
      }
    } else {
      DO(ConsumeIdentifierBeforeWhitespace(&field_name));
      TryConsumeWhitespace();

      arc_i32 field_number;
      if (allow_field_number_ && y_absl::SimpleAtoi(field_name, &field_number)) {
        if (descriptor->IsExtensionNumber(field_number)) {
          field = finder_
                      ? finder_->FindExtensionByNumber(descriptor, field_number)
                      : DefaultFinderFindExtensionByNumber(descriptor,
                                                           field_number);
        } else if (descriptor->IsReservedNumber(field_number)) {
          reserved_field = true;
        } else {
          field = descriptor->FindFieldByNumber(field_number);
        }
      } else {
        field = descriptor->FindFieldByName(field_name);
        // Group names are expected to be capitalized as they appear in the
        // .proto file, which actually matches their type names, not their
        // field names.
        if (field == nullptr) {
          TProtoStringType lower_field_name = field_name;
          y_absl::AsciiStrToLower(&lower_field_name);
          field = descriptor->FindFieldByName(lower_field_name);
          // If the case-insensitive match worked but the field is NOT a group,
          if (field != nullptr &&
              field->type() != FieldDescriptor::TYPE_GROUP) {
            field = nullptr;
          }
        }
        // Again, special-case group names as described above.
        if (field != nullptr && field->type() == FieldDescriptor::TYPE_GROUP &&
            field->message_type()->name() != field_name) {
          field = nullptr;
        }

        if (field == nullptr && allow_case_insensitive_field_) {
          TProtoStringType lower_field_name = field_name;
          y_absl::AsciiStrToLower(&lower_field_name);
          field = descriptor->FindFieldByLowercaseName(lower_field_name);
        }

        if (field == nullptr) {
          reserved_field = descriptor->IsReservedName(field_name);
        }
      }

      if (field == nullptr && !reserved_field) {
        if (!allow_unknown_field_) {
          ReportError(y_absl::StrCat("Message type \"", descriptor->full_name(),
                                   "\" has no field named \"", field_name,
                                   "\"."));
          return false;
        } else {
          ReportWarning(y_absl::StrCat("Message type \"", descriptor->full_name(),
                                     "\" has no field named \"", field_name,
                                     "\"."));
        }
      }
    }

    // Skips unknown or reserved fields.
    if (field == nullptr) {
      Y_ABSL_CHECK(allow_unknown_field_ || allow_unknown_extension_ ||
                 reserved_field);

      // Try to guess the type of this field.
      // If this field is not a message, there should be a ":" between the
      // field name and the field value and also the field value should not
      // start with "{" or "<" which indicates the beginning of a message body.
      // If there is no ":" or there is a "{" or "<" after ":", this field has
      // to be a message or the input is ill-formed.
      bool skip;
      if (TryConsumeBeforeWhitespace(":")) {
        TryConsumeWhitespace();
        if (!LookingAt("{") && !LookingAt("<")) {
          skip = SkipFieldValue();
        } else {
          skip = SkipFieldMessage();
        }
      } else {
        skip = SkipFieldMessage();
      }
      TryConsume(";") || TryConsume(",");
      return skip;
    }

    if (field->options().deprecated()) {
      ReportWarning(y_absl::StrCat("text format contains deprecated field \"",
                                 field_name, "\""));
    }

    if (singular_overwrite_policy_ == FORBID_SINGULAR_OVERWRITES) {
      // Fail if the field is not repeated and it has already been specified.
      if (!field->is_repeated() && reflection->HasField(*message, field)) {
        ReportError(y_absl::StrCat("Non-repeated field \"", field_name,
                                 "\" is specified multiple times."));
        return false;
      }
      // Fail if the field is a member of a oneof and another member has already
      // been specified.
      const OneofDescriptor* oneof = field->containing_oneof();
      if (oneof != nullptr && reflection->HasOneof(*message, oneof)) {
        const FieldDescriptor* other_field =
            reflection->GetOneofFieldDescriptor(*message, oneof);
        ReportError(y_absl::StrCat("Field \"", field_name,
                                 "\" is specified along with "
                                 "field \"",
                                 other_field->name(),
                                 "\", another member "
                                 "of oneof \"",
                                 oneof->name(), "\"."));
        return false;
      }
    }

    // Perform special handling for embedded message types.
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      // ':' is optional here.
      bool consumed_semicolon = TryConsumeBeforeWhitespace(":");
      if (consumed_semicolon) {
        TryConsumeWhitespace();
      }
      if (consumed_semicolon && field->options().weak() &&
          LookingAtType(io::Tokenizer::TYPE_STRING)) {
        // we are getting a bytes string for a weak field.
        TProtoStringType tmp;
        DO(ConsumeString(&tmp));
        MessageFactory* factory =
            finder_ ? finder_->FindExtensionFactory(field) : nullptr;
        reflection->MutableMessage(message, field, factory)
            ->ParseFromString(tmp);
        goto label_skip_parsing;
      }
    } else {
      // ':' is required here.
      DO(ConsumeBeforeWhitespace(":"));
      TryConsumeWhitespace();
    }

    if (field->is_repeated() && TryConsume("[")) {
      // Short repeated format, e.g.  "foo: [1, 2, 3]".
      if (!TryConsume("]")) {
        // "foo: []" is treated as empty.
        while (true) {
          if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
            // Perform special handling for embedded message types.
            DO(ConsumeFieldMessage(message, reflection, field));
          } else {
            DO(ConsumeFieldValue(message, reflection, field));
          }
          if (TryConsume("]")) {
            break;
          }
          DO(Consume(","));
        }
      }
    } else if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      DO(ConsumeFieldMessage(message, reflection, field));
    } else {
      DO(ConsumeFieldValue(message, reflection, field));
    }
  label_skip_parsing:
    // For historical reasons, fields may optionally be separated by commas or
    // semicolons.
    TryConsume(";") || TryConsume(",");

    // If a parse info tree exists, add the location for the parsed
    // field.
    if (parse_info_tree_ != nullptr) {
      int end_line = tokenizer_.previous().line;
      int end_column = tokenizer_.previous().end_column;

      RecordLocation(parse_info_tree_, field,
                     ParseLocationRange(ParseLocation(start_line, start_column),
                                        ParseLocation(end_line, end_column)));
    }

    return true;
  }

  // Skips the next field including the field's name and value.
  bool SkipField() {
    TProtoStringType field_name;
    if (TryConsume("[")) {
      // Extension name or type URL.
      DO(ConsumeTypeUrlOrFullTypeName(&field_name));
      DO(ConsumeBeforeWhitespace("]"));
    } else {
      DO(ConsumeIdentifierBeforeWhitespace(&field_name));
    }
    TryConsumeWhitespace();

    // Try to guess the type of this field.
    // If this field is not a message, there should be a ":" between the
    // field name and the field value and also the field value should not
    // start with "{" or "<" which indicates the beginning of a message body.
    // If there is no ":" or there is a "{" or "<" after ":", this field has
    // to be a message or the input is ill-formed.
    if (TryConsumeBeforeWhitespace(":")) {
      TryConsumeWhitespace();
      if (!LookingAt("{") && !LookingAt("<")) {
        DO(SkipFieldValue());
      } else {
        DO(SkipFieldMessage());
      }
    } else {
      DO(SkipFieldMessage());
    }
    // For historical reasons, fields may optionally be separated by commas or
    // semicolons.
    TryConsume(";") || TryConsume(",");
    return true;
  }

  bool ConsumeFieldMessage(Message* message, const Reflection* reflection,
                           const FieldDescriptor* field) {
    if (--recursion_limit_ < 0) {
      ReportError(
          y_absl::StrCat("Message is too deep, the parser exceeded the "
                       "configured recursion limit of ",
                       initial_recursion_limit_, "."));
      return false;
    }
    // If the parse information tree is not nullptr, create a nested one
    // for the nested message.
    ParseInfoTree* parent = parse_info_tree_;
    if (parent != nullptr) {
      parse_info_tree_ = CreateNested(parent, field);
    }

    TProtoStringType delimiter;
    DO(ConsumeMessageDelimiter(&delimiter));
    MessageFactory* factory =
        finder_ ? finder_->FindExtensionFactory(field) : nullptr;
    if (field->is_repeated()) {
      DO(ConsumeMessage(reflection->AddMessage(message, field, factory),
                        delimiter));
    } else {
      DO(ConsumeMessage(reflection->MutableMessage(message, field, factory),
                        delimiter));
    }

    ++recursion_limit_;

    // Reset the parse information tree.
    parse_info_tree_ = parent;
    return true;
  }

  // Skips the whole body of a message including the beginning delimiter and
  // the ending delimiter.
  bool SkipFieldMessage() {
    if (--recursion_limit_ < 0) {
      ReportError(
          y_absl::StrCat("Message is too deep, the parser exceeded the "
                       "configured recursion limit of ",
                       initial_recursion_limit_, "."));
      return false;
    }

    TProtoStringType delimiter;
    DO(ConsumeMessageDelimiter(&delimiter));
    while (!LookingAt(">") && !LookingAt("}")) {
      DO(SkipField());
    }
    DO(Consume(delimiter));

    ++recursion_limit_;
    return true;
  }

  bool ConsumeFieldValue(Message* message, const Reflection* reflection,
                         const FieldDescriptor* field) {
// Define an easy to use macro for setting fields. This macro checks
// to see if the field is repeated (in which case we need to use the Add
// methods or not (in which case we need to use the Set methods).
#define SET_FIELD(CPPTYPE, VALUE)                    \
  if (field->is_repeated()) {                        \
    reflection->Add##CPPTYPE(message, field, VALUE); \
  } else {                                           \
    reflection->Set##CPPTYPE(message, field, VALUE); \
  }

    switch (field->cpp_type()) {
      case FieldDescriptor::CPPTYPE_INT32: {
        arc_i64 value;
        DO(ConsumeSignedInteger(&value, kint32max));
        SET_FIELD(Int32, static_cast<arc_i32>(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_UINT32: {
        arc_ui64 value;
        DO(ConsumeUnsignedInteger(&value, kuint32max));
        SET_FIELD(UInt32, static_cast<arc_ui32>(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_INT64: {
        arc_i64 value;
        DO(ConsumeSignedInteger(&value, kint64max));
        SET_FIELD(Int64, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_UINT64: {
        arc_ui64 value;
        DO(ConsumeUnsignedInteger(&value, kuint64max));
        SET_FIELD(UInt64, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_FLOAT: {
        double value;
        DO(ConsumeDouble(&value));
        SET_FIELD(Float, io::SafeDoubleToFloat(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_DOUBLE: {
        double value;
        DO(ConsumeDouble(&value));
        SET_FIELD(Double, value);
        break;
      }

      case FieldDescriptor::CPPTYPE_STRING: {
        TProtoStringType value;
        DO(ConsumeString(&value));
        SET_FIELD(String, std::move(value));
        break;
      }

      case FieldDescriptor::CPPTYPE_BOOL: {
        if (LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
          arc_ui64 value;
          DO(ConsumeUnsignedInteger(&value, 1));
          SET_FIELD(Bool, value);
        } else {
          TProtoStringType value;
          DO(ConsumeIdentifier(&value));
          if (value == "true" || value == "True" || value == "t") {
            SET_FIELD(Bool, true);
          } else if (value == "false" || value == "False" || value == "f") {
            SET_FIELD(Bool, false);
          } else {
            ReportError(y_absl::StrCat("Invalid value for boolean field \"",
                                     field->name(), "\". Value: \"", value,
                                     "\"."));
            return false;
          }
        }
        break;
      }

      case FieldDescriptor::CPPTYPE_ENUM: {
        TProtoStringType value;
        arc_i64 int_value = kint64max;
        const EnumDescriptor* enum_type = field->enum_type();
        const EnumValueDescriptor* enum_value = nullptr;

        if (LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
          DO(ConsumeIdentifier(&value));
          // Find the enumeration value.
          enum_value = enum_type->FindValueByName(value);

        } else if (LookingAt("-") ||
                   LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
          DO(ConsumeSignedInteger(&int_value, kint32max));
          value = y_absl::StrCat(int_value);  // for error reporting
          enum_value = enum_type->FindValueByNumber(int_value);
        } else {
          ReportError(y_absl::StrCat("Expected integer or identifier, got: ",
                                   tokenizer_.current().text));
          return false;
        }

        if (enum_value == nullptr) {
          if (int_value != kint64max &&
              reflection->SupportsUnknownEnumValues()) {
            SET_FIELD(EnumValue, int_value);
            return true;
          } else if (!allow_unknown_enum_) {
            ReportError(y_absl::StrCat("Unknown enumeration value of \"", value,
                                     "\" for field \"", field->name(), "\"."));
            return false;
          } else {
            ReportWarning(y_absl::StrCat("Unknown enumeration value of \"", value,
                                       "\" for field \"", field->name(),
                                       "\"."));
            return true;
          }
        }

        SET_FIELD(Enum, enum_value);
        break;
      }

      case FieldDescriptor::CPPTYPE_MESSAGE: {
        // We should never get here. Put here instead of a default
        // so that if new types are added, we get a nice compiler warning.
        Y_ABSL_LOG(FATAL) << "Reached an unintended state: CPPTYPE_MESSAGE";
        break;
      }
    }
#undef SET_FIELD
    return true;
  }

  bool SkipFieldValue() {
    if (--recursion_limit_ < 0) {
      ReportError(
          y_absl::StrCat("Message is too deep, the parser exceeded the "
                       "configured recursion limit of ",
                       initial_recursion_limit_, "."));
      return false;
    }

    if (LookingAtType(io::Tokenizer::TYPE_STRING)) {
      while (LookingAtType(io::Tokenizer::TYPE_STRING)) {
        tokenizer_.Next();
      }
      ++recursion_limit_;
      return true;
    }
    if (TryConsume("[")) {
      if (!TryConsume("]")) {
        while (true) {
          if (!LookingAt("{") && !LookingAt("<")) {
            DO(SkipFieldValue());
          } else {
            DO(SkipFieldMessage());
          }
          if (TryConsume("]")) {
            break;
          }
          DO(Consume(","));
        }
      }
      ++recursion_limit_;
      return true;
    }
    // Possible field values other than string:
    //   12345        => TYPE_INTEGER
    //   -12345       => TYPE_SYMBOL + TYPE_INTEGER
    //   1.2345       => TYPE_FLOAT
    //   -1.2345      => TYPE_SYMBOL + TYPE_FLOAT
    //   inf          => TYPE_IDENTIFIER
    //   -inf         => TYPE_SYMBOL + TYPE_IDENTIFIER
    //   TYPE_INTEGER => TYPE_IDENTIFIER
    // Divides them into two group, one with TYPE_SYMBOL
    // and the other without:
    //   Group one:
    //     12345        => TYPE_INTEGER
    //     1.2345       => TYPE_FLOAT
    //     inf          => TYPE_IDENTIFIER
    //     TYPE_INTEGER => TYPE_IDENTIFIER
    //   Group two:
    //     -12345       => TYPE_SYMBOL + TYPE_INTEGER
    //     -1.2345      => TYPE_SYMBOL + TYPE_FLOAT
    //     -inf         => TYPE_SYMBOL + TYPE_IDENTIFIER
    // As we can see, the field value consists of an optional '-' and one of
    // TYPE_INTEGER, TYPE_FLOAT and TYPE_IDENTIFIER.
    bool has_minus = TryConsume("-");
    if (!LookingAtType(io::Tokenizer::TYPE_INTEGER) &&
        !LookingAtType(io::Tokenizer::TYPE_FLOAT) &&
        !LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      TProtoStringType text = tokenizer_.current().text;
      ReportError(
          y_absl::StrCat("Cannot skip field value, unexpected token: ", text));
      ++recursion_limit_;
      return false;
    }
    // Combination of '-' and TYPE_IDENTIFIER may result in an invalid field
    // value while other combinations all generate valid values.
    // We check if the value of this combination is valid here.
    // TYPE_IDENTIFIER after a '-' should be one of the float values listed
    // below:
    //   inf, inff, infinity, nan
    if (has_minus && LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      TProtoStringType text = tokenizer_.current().text;
      y_absl::AsciiStrToLower(&text);
      if (text != "inf" &&
          text != "infinity" && text != "nan") {
        ReportError(y_absl::StrCat("Invalid float number: ", text));
        ++recursion_limit_;
        return false;
      }
    }
    tokenizer_.Next();
    ++recursion_limit_;
    return true;
  }

  // Returns true if the current token's text is equal to that specified.
  bool LookingAt(const TProtoStringType& text) {
    return tokenizer_.current().text == text;
  }

  // Returns true if the current token's type is equal to that specified.
  bool LookingAtType(io::Tokenizer::TokenType token_type) {
    return tokenizer_.current().type == token_type;
  }

  // Consumes an identifier and saves its value in the identifier parameter.
  // Returns false if the token is not of type IDENTIFIER.
  bool ConsumeIdentifier(TProtoStringType* identifier) {
    if (LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      *identifier = tokenizer_.current().text;
      tokenizer_.Next();
      return true;
    }

    // If allow_field_numer_ or allow_unknown_field_ is true, we should able
    // to parse integer identifiers.
    if ((allow_field_number_ || allow_unknown_field_ ||
         allow_unknown_extension_) &&
        LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      *identifier = tokenizer_.current().text;
      tokenizer_.Next();
      return true;
    }

    ReportError(
        y_absl::StrCat("Expected identifier, got: ", tokenizer_.current().text));
    return false;
  }

  // Similar to `ConsumeIdentifier`, but any following whitespace token may
  // be reported.
  bool ConsumeIdentifierBeforeWhitespace(TProtoStringType* identifier) {
    tokenizer_.set_report_whitespace(true);
    bool result = ConsumeIdentifier(identifier);
    tokenizer_.set_report_whitespace(false);
    return result;
  }

  // Consume a string of form "<id1>.<id2>....<idN>".
  bool ConsumeFullTypeName(TProtoStringType* name) {
    DO(ConsumeIdentifier(name));
    while (TryConsume(".")) {
      TProtoStringType part;
      DO(ConsumeIdentifier(&part));
      y_absl::StrAppend(name, ".", part);
    }
    return true;
  }

  bool ConsumeTypeUrlOrFullTypeName(TProtoStringType* name) {
    DO(ConsumeIdentifier(name));
    while (true) {
      TProtoStringType connector;
      if (TryConsume(".")) {
        connector = ".";
      } else if (TryConsume("/")) {
        connector = "/";
      } else {
        break;
      }
      TProtoStringType part;
      DO(ConsumeIdentifier(&part));
      *name += connector;
      *name += part;
    }
    return true;
  }

  // Consumes a string and saves its value in the text parameter.
  // Returns false if the token is not of type STRING.
  bool ConsumeString(TProtoStringType* text) {
    if (!LookingAtType(io::Tokenizer::TYPE_STRING)) {
      ReportError(
          y_absl::StrCat("Expected string, got: ", tokenizer_.current().text));
      return false;
    }

    text->clear();
    while (LookingAtType(io::Tokenizer::TYPE_STRING)) {
      io::Tokenizer::ParseStringAppend(tokenizer_.current().text, text);

      tokenizer_.Next();
    }

    return true;
  }

  // Consumes a arc_ui64 and saves its value in the value parameter.
  // Returns false if the token is not of type INTEGER.
  bool ConsumeUnsignedInteger(arc_ui64* value, arc_ui64 max_value) {
    if (!LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      ReportError(
          y_absl::StrCat("Expected integer, got: ", tokenizer_.current().text));
      return false;
    }

    if (!io::Tokenizer::ParseInteger(tokenizer_.current().text, max_value,
                                     value)) {
      ReportError(y_absl::StrCat("Integer out of range (",
                               tokenizer_.current().text, ")"));
      return false;
    }

    tokenizer_.Next();
    return true;
  }

  // Consumes an arc_i64 and saves its value in the value parameter.
  // Note that since the tokenizer does not support negative numbers,
  // we actually may consume an additional token (for the minus sign) in this
  // method. Returns false if the token is not an integer
  // (signed or otherwise).
  bool ConsumeSignedInteger(arc_i64* value, arc_ui64 max_value) {
    bool negative = false;

    if (TryConsume("-")) {
      negative = true;
      // Two's complement always allows one more negative integer than
      // positive.
      ++max_value;
    }

    arc_ui64 unsigned_value;

    DO(ConsumeUnsignedInteger(&unsigned_value, max_value));

    if (negative) {
      if ((static_cast<arc_ui64>(kint64max) + 1) == unsigned_value) {
        *value = kint64min;
      } else {
        *value = -static_cast<arc_i64>(unsigned_value);
      }
    } else {
      *value = static_cast<arc_i64>(unsigned_value);
    }

    return true;
  }

  // Consumes a double and saves its value in the value parameter.
  // Accepts decimal numbers only, rejects hex or oct numbers.
  bool ConsumeUnsignedDecimalAsDouble(double* value, arc_ui64 max_value) {
    if (!LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      ReportError(
          y_absl::StrCat("Expected integer, got: ", tokenizer_.current().text));
      return false;
    }

    const TProtoStringType& text = tokenizer_.current().text;
    if (IsHexNumber(text) || IsOctNumber(text)) {
      ReportError(y_absl::StrCat("Expect a decimal number, got: ", text));
      return false;
    }

    arc_ui64 uint64_value;
    if (io::Tokenizer::ParseInteger(text, max_value, &uint64_value)) {
      *value = static_cast<double>(uint64_value);
    } else {
      // Uint64 overflow, attempt to parse as a double instead.
      *value = io::Tokenizer::ParseFloat(text);
    }

    tokenizer_.Next();
    return true;
  }

  // Consumes a double and saves its value in the value parameter.
  // Note that since the tokenizer does not support negative numbers,
  // we actually may consume an additional token (for the minus sign) in this
  // method. Returns false if the token is not a double
  // (signed or otherwise).
  bool ConsumeDouble(double* value) {
    bool negative = false;

    if (TryConsume("-")) {
      negative = true;
    }

    // A double can actually be an integer, according to the tokenizer.
    // Therefore, we must check both cases here.
    if (LookingAtType(io::Tokenizer::TYPE_INTEGER)) {
      // We have found an integer value for the double.
      DO(ConsumeUnsignedDecimalAsDouble(value, kuint64max));
    } else if (LookingAtType(io::Tokenizer::TYPE_FLOAT)) {
      // We have found a float value for the double.
      *value = io::Tokenizer::ParseFloat(tokenizer_.current().text);

      // Mark the current token as consumed.
      tokenizer_.Next();
    } else if (LookingAtType(io::Tokenizer::TYPE_IDENTIFIER)) {
      TProtoStringType text = tokenizer_.current().text;
      y_absl::AsciiStrToLower(&text);
      if (text == "inf" ||
          text == "infinity") {
        *value = std::numeric_limits<double>::infinity();
        tokenizer_.Next();
      } else if (text == "nan") {
        *value = std::numeric_limits<double>::quiet_NaN();
        tokenizer_.Next();
      } else {
        ReportError(y_absl::StrCat("Expected double, got: ", text));
        return false;
      }
    } else {
      ReportError(
          y_absl::StrCat("Expected double, got: ", tokenizer_.current().text));
      return false;
    }

    if (negative) {
      *value = -*value;
    }

    return true;
  }

  // Consumes Any::type_url value, of form "type.googleapis.com/full.type.Name"
  // or "type.googleprod.com/full.type.Name"
  bool ConsumeAnyTypeUrl(TProtoStringType* full_type_name, TProtoStringType* prefix) {
    // TODO(saito) Extend Consume() to consume multiple tokens at once, so that
    // this code can be written as just DO(Consume(kGoogleApisTypePrefix)).
    DO(ConsumeIdentifier(prefix));
    while (TryConsume(".")) {
      TProtoStringType url;
      DO(ConsumeIdentifier(&url));
      y_absl::StrAppend(prefix, ".", url);
    }
    DO(Consume("/"));
    y_absl::StrAppend(prefix, "/");
    DO(ConsumeFullTypeName(full_type_name));

    return true;
  }

  // A helper function for reconstructing Any::value. Consumes a text of
  // full_type_name, then serializes it into serialized_value.
  bool ConsumeAnyValue(const Descriptor* value_descriptor,
                       TProtoStringType* serialized_value) {
    DynamicMessageFactory factory;
    const Message* value_prototype = factory.GetPrototype(value_descriptor);
    if (value_prototype == nullptr) {
      return false;
    }
    std::unique_ptr<Message> value(value_prototype->New());
    TProtoStringType sub_delimiter;
    DO(ConsumeMessageDelimiter(&sub_delimiter));
    DO(ConsumeMessage(value.get(), sub_delimiter));

    if (allow_partial_) {
      value->AppendPartialToString(serialized_value);
    } else {
      if (!value->IsInitialized()) {
        ReportError(y_absl::StrCat(
            "Value of type \"", value_descriptor->full_name(),
            "\" stored in google.protobuf.Any has missing required fields"));
        return false;
      }
      value->AppendToString(serialized_value);
    }
    return true;
  }

  // Consumes a token and confirms that it matches that specified in the
  // value parameter. Returns false if the token found does not match that
  // which was specified.
  bool Consume(const TProtoStringType& value) {
    const TProtoStringType& current_value = tokenizer_.current().text;

    if (current_value != value) {
      ReportError(y_absl::StrCat("Expected \"", value, "\", found \"",
                               current_value, "\"."));
      return false;
    }

    tokenizer_.Next();

    return true;
  }

  // Similar to `Consume`, but the following token may be tokenized as
  // TYPE_WHITESPACE.
  bool ConsumeBeforeWhitespace(const TProtoStringType& value) {
    // Report whitespace after this token, but only once.
    tokenizer_.set_report_whitespace(true);
    bool result = Consume(value);
    tokenizer_.set_report_whitespace(false);
    return result;
  }

  // Attempts to consume the supplied value. Returns false if a the
  // token found does not match the value specified.
  bool TryConsume(const TProtoStringType& value) {
    if (tokenizer_.current().text == value) {
      tokenizer_.Next();
      return true;
    } else {
      return false;
    }
  }

  // Similar to `TryConsume`, but the following token may be tokenized as
  // TYPE_WHITESPACE.
  bool TryConsumeBeforeWhitespace(const TProtoStringType& value) {
    // Report whitespace after this token, but only once.
    tokenizer_.set_report_whitespace(true);
    bool result = TryConsume(value);
    tokenizer_.set_report_whitespace(false);
    return result;
  }

  bool TryConsumeWhitespace() {
    had_silent_marker_ = false;
    if (LookingAtType(io::Tokenizer::TYPE_WHITESPACE)) {
      if (tokenizer_.current().text ==
          y_absl::StrCat(" ", internal::kDebugStringSilentMarkerForDetection)) {
        had_silent_marker_ = true;
      }
      tokenizer_.Next();
      return true;
    }
    return false;
  }

  // An internal instance of the Tokenizer's error collector, used to
  // collect any base-level parse errors and feed them to the ParserImpl.
  class ParserErrorCollector : public io::ErrorCollector {
   public:
    explicit ParserErrorCollector(TextFormat::Parser::ParserImpl* parser)
        : parser_(parser) {}

    ParserErrorCollector(const ParserErrorCollector&) = delete;
    ParserErrorCollector& operator=(const ParserErrorCollector&) = delete;
    ~ParserErrorCollector() override {}

    void RecordError(int line, int column, y_absl::string_view message) override {
      parser_->ReportError(line, column, message);
    }

    void RecordWarning(int line, int column,
                       y_absl::string_view message) override {
      parser_->ReportWarning(line, column, message);
    }

   private:
    TextFormat::Parser::ParserImpl* parser_;
  };

  io::ErrorCollector* error_collector_;
  const TextFormat::Finder* finder_;
  ParseInfoTree* parse_info_tree_;
  ParserErrorCollector tokenizer_error_collector_;
  io::Tokenizer tokenizer_;
  const Descriptor* root_message_type_;
  SingularOverwritePolicy singular_overwrite_policy_;
  const bool allow_case_insensitive_field_;
  const bool allow_unknown_field_;
  const bool allow_unknown_extension_;
  const bool allow_unknown_enum_;
  const bool allow_field_number_;
  const bool allow_partial_;
  const int initial_recursion_limit_;
  int recursion_limit_;
  bool had_silent_marker_;
  bool had_errors_;
};

// ===========================================================================
// Internal class for writing text to the io::ZeroCopyOutputStream. Adapted
// from the Printer found in //third_party/protobuf/io/printer.h
class TextFormat::Printer::TextGenerator
    : public TextFormat::BaseTextGenerator {
 public:
  explicit TextGenerator(io::ZeroCopyOutputStream* output,
                         int initial_indent_level)
      : output_(output),
        buffer_(nullptr),
        buffer_size_(0),
        at_start_of_line_(true),
        failed_(false),
        insert_silent_marker_(false),
        indent_level_(initial_indent_level),
        initial_indent_level_(initial_indent_level) {}

  explicit TextGenerator(io::ZeroCopyOutputStream* output,
                         bool insert_silent_marker, int initial_indent_level)
      : output_(output),
        buffer_(nullptr),
        buffer_size_(0),
        at_start_of_line_(true),
        failed_(false),
        insert_silent_marker_(insert_silent_marker),
        indent_level_(initial_indent_level),
        initial_indent_level_(initial_indent_level) {}

  TextGenerator(const TextGenerator&) = delete;
  TextGenerator& operator=(const TextGenerator&) = delete;
  ~TextGenerator() override {
    // Only BackUp() if we're sure we've successfully called Next() at least
    // once.
    if (!failed_) {
      output_->BackUp(buffer_size_);
    }
  }

  // Indent text by two spaces.  After calling Indent(), two spaces will be
  // inserted at the beginning of each line of text.  Indent() may be called
  // multiple times to produce deeper indents.
  void Indent() override { ++indent_level_; }

  // Reduces the current indent level by two spaces, or crashes if the indent
  // level is zero.
  void Outdent() override {
    if (indent_level_ == 0 || indent_level_ < initial_indent_level_) {
      Y_ABSL_DLOG(FATAL) << " Outdent() without matching Indent().";
      return;
    }

    --indent_level_;
  }

  size_t GetCurrentIndentationSize() const override {
    return 2 * indent_level_;
  }

  // Print text to the output stream.
  void Print(const char* text, size_t size) override {
    if (indent_level_ > 0) {
      size_t pos = 0;  // The number of bytes we've written so far.
      for (size_t i = 0; i < size; i++) {
        if (text[i] == '\n') {
          // Saw newline.  If there is more text, we may need to insert an
          // indent here.  So, write what we have so far, including the '\n'.
          Write(text + pos, i - pos + 1);
          pos = i + 1;

          // Setting this true will cause the next Write() to insert an indent
          // first.
          at_start_of_line_ = true;
        }
      }
      // Write the rest.
      Write(text + pos, size - pos);
    } else {
      Write(text, size);
      if (size > 0 && text[size - 1] == '\n') {
        at_start_of_line_ = true;
      }
    }
  }

  // True if any write to the underlying stream failed.  (We don't just
  // crash in this case because this is an I/O failure, not a programming
  // error.)
  bool failed() const { return failed_; }

  void PrintMaybeWithMarker(MarkerToken, y_absl::string_view text) override {
    Print(text.data(), text.size());
    if (ConsumeInsertSilentMarker()) {
      PrintLiteral(internal::kDebugStringSilentMarker);
    }
  }

  void PrintMaybeWithMarker(MarkerToken, y_absl::string_view text_head,
                            y_absl::string_view text_tail) override {
    Print(text_head.data(), text_head.size());
    if (ConsumeInsertSilentMarker()) {
      PrintLiteral(internal::kDebugStringSilentMarker);
    }
    Print(text_tail.data(), text_tail.size());
  }

 private:
  void Write(const char* data, size_t size) {
    if (failed_) return;
    if (size == 0) return;

    if (at_start_of_line_) {
      // Insert an indent.
      at_start_of_line_ = false;
      WriteIndent();
      if (failed_) return;
    }

    while (static_cast<arc_i64>(size) > buffer_size_) {
      // Data exceeds space in the buffer.  Copy what we can and request a
      // new buffer.
      if (buffer_size_ > 0) {
        memcpy(buffer_, data, buffer_size_);
        data += buffer_size_;
        size -= buffer_size_;
      }
      void* void_buffer = nullptr;
      failed_ = !output_->Next(&void_buffer, &buffer_size_);
      if (failed_) return;
      buffer_ = reinterpret_cast<char*>(void_buffer);
    }

    // Buffer is big enough to receive the data; copy it.
    memcpy(buffer_, data, size);
    buffer_ += size;
    buffer_size_ -= size;
  }

  void WriteIndent() {
    if (indent_level_ == 0) {
      return;
    }
    Y_ABSL_DCHECK(!failed_);
    int size = GetCurrentIndentationSize();

    while (size > buffer_size_) {
      // Data exceeds space in the buffer. Write what we can and request a new
      // buffer.
      if (buffer_size_ > 0) {
        memset(buffer_, ' ', buffer_size_);
      }
      size -= buffer_size_;
      void* void_buffer;
      failed_ = !output_->Next(&void_buffer, &buffer_size_);
      if (failed_) return;
      buffer_ = reinterpret_cast<char*>(void_buffer);
    }

    // Buffer is big enough to receive the data; copy it.
    memset(buffer_, ' ', size);
    buffer_ += size;
    buffer_size_ -= size;
  }

  // Return the current value of insert_silent_marker_. If it is true, set it
  // to false as we assume that a silent marker is inserted after a call to this
  // function.
  bool ConsumeInsertSilentMarker() {
    if (insert_silent_marker_) {
      insert_silent_marker_ = false;
      return true;
    }
    return false;
  }

  io::ZeroCopyOutputStream* const output_;
  char* buffer_;
  int buffer_size_;
  bool at_start_of_line_;
  bool failed_;
  // This flag is false when inserting silent marker is disabled or a silent
  // marker has been inserted.
  bool insert_silent_marker_;

  int indent_level_;
  int initial_indent_level_;
};

// ===========================================================================
//  An internal field value printer that may insert a silent marker in
//  DebugStrings.
class TextFormat::Printer::DebugStringFieldValuePrinter
    : public TextFormat::FastFieldValuePrinter {
 public:
  void PrintMessageStart(const Message& /*message*/, int /*field_index*/,
                         int /*field_count*/, bool single_line_mode,
                         BaseTextGenerator* generator) const override {
    if (single_line_mode) {
      generator->PrintMaybeWithMarker(MarkerToken(), " ", "{ ");
    } else {
      generator->PrintMaybeWithMarker(MarkerToken(), " ", "{\n");
    }
  }
};

// ===========================================================================
//  An internal field value printer that escape UTF8 strings.
class TextFormat::Printer::FastFieldValuePrinterUtf8Escaping
    : public TextFormat::Printer::DebugStringFieldValuePrinter {
 public:
  void PrintString(const TProtoStringType& val,
                   TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintLiteral("\"");
    generator->PrintString(y_absl::Utf8SafeCEscape(val));
    generator->PrintLiteral("\"");
  }
  void PrintBytes(const TProtoStringType& val,
                  TextFormat::BaseTextGenerator* generator) const override {
    return FastFieldValuePrinter::PrintString(val, generator);
  }
};

// ===========================================================================
// Implementation of the default Finder for extensions.
TextFormat::Finder::~Finder() {}

const FieldDescriptor* TextFormat::Finder::FindExtension(
    Message* message, const TProtoStringType& name) const {
  return DefaultFinderFindExtension(message, name);
}

const FieldDescriptor* TextFormat::Finder::FindExtensionByNumber(
    const Descriptor* descriptor, int number) const {
  return DefaultFinderFindExtensionByNumber(descriptor, number);
}

const Descriptor* TextFormat::Finder::FindAnyType(
    const Message& message, const TProtoStringType& prefix,
    const TProtoStringType& name) const {
  return DefaultFinderFindAnyType(message, prefix, name);
}

MessageFactory* TextFormat::Finder::FindExtensionFactory(
    const FieldDescriptor* /*field*/) const {
  return nullptr;
}

// ===========================================================================

TextFormat::Parser::Parser()
    : error_collector_(nullptr),
      finder_(nullptr),
      parse_info_tree_(nullptr),
      allow_partial_(false),
      allow_case_insensitive_field_(false),
      allow_unknown_field_(false),
      allow_unknown_extension_(false),
      allow_unknown_enum_(false),
      allow_field_number_(false),
      allow_relaxed_whitespace_(false),
      allow_singular_overwrites_(false),
      recursion_limit_(std::numeric_limits<int>::max()) {}

TextFormat::Parser::~Parser() {}

namespace {

bool CheckParseInputSize(y_absl::string_view input,
                         io::ErrorCollector* error_collector) {
  if (input.size() > INT_MAX) {
    error_collector->RecordError(
        -1, 0,
        y_absl::StrCat(
            "Input size too large: ", static_cast<arc_i64>(input.size()),
            " bytes", " > ", INT_MAX, " bytes."));
    return false;
  }
  return true;
}

}  // namespace

bool TextFormat::Parser::Parse(io::ZeroCopyInputStream* input,
                               Message* output) {
  output->Clear();

  ParserImpl::SingularOverwritePolicy overwrites_policy =
      allow_singular_overwrites_ ? ParserImpl::ALLOW_SINGULAR_OVERWRITES
                                 : ParserImpl::FORBID_SINGULAR_OVERWRITES;

  ParserImpl parser(output->GetDescriptor(), input, error_collector_, finder_,
                    parse_info_tree_, overwrites_policy,
                    allow_case_insensitive_field_, allow_unknown_field_,
                    allow_unknown_extension_, allow_unknown_enum_,
                    allow_field_number_, allow_relaxed_whitespace_,
                    allow_partial_, recursion_limit_);
  return MergeUsingImpl(input, output, &parser);
}

bool TextFormat::Parser::ParseFromString(y_absl::string_view input,
                                         Message* output) {
  DO(CheckParseInputSize(input, error_collector_));
  io::ArrayInputStream input_stream(input.data(), input.size());
  return Parse(&input_stream, output);
}

bool TextFormat::Parser::Merge(io::ZeroCopyInputStream* input,
                               Message* output) {
  ParserImpl parser(output->GetDescriptor(), input, error_collector_, finder_,
                    parse_info_tree_, ParserImpl::ALLOW_SINGULAR_OVERWRITES,
                    allow_case_insensitive_field_, allow_unknown_field_,
                    allow_unknown_extension_, allow_unknown_enum_,
                    allow_field_number_, allow_relaxed_whitespace_,
                    allow_partial_, recursion_limit_);
  return MergeUsingImpl(input, output, &parser);
}

bool TextFormat::Parser::MergeFromString(y_absl::string_view input,
                                         Message* output) {
  DO(CheckParseInputSize(input, error_collector_));
  io::ArrayInputStream input_stream(input.data(), input.size());
  return Merge(&input_stream, output);
}

bool TextFormat::Parser::MergeUsingImpl(io::ZeroCopyInputStream* /* input */,
                                        Message* output,
                                        ParserImpl* parser_impl) {
  if (!parser_impl->Parse(output)) return false;
  if (!allow_partial_ && !output->IsInitialized()) {
    std::vector<TProtoStringType> missing_fields;
    output->FindInitializationErrors(&missing_fields);
    parser_impl->ReportError(-1, 0,
                             y_absl::StrCat("Message missing required fields: ",
                                          y_absl::StrJoin(missing_fields, ", ")));
    return false;
  }
  return true;
}

bool TextFormat::Parser::ParseFieldValueFromString(const TProtoStringType& input,
                                                   const FieldDescriptor* field,
                                                   Message* output) {
  io::ArrayInputStream input_stream(input.data(), input.size());
  ParserImpl parser(
      output->GetDescriptor(), &input_stream, error_collector_, finder_,
      parse_info_tree_, ParserImpl::ALLOW_SINGULAR_OVERWRITES,
      allow_case_insensitive_field_, allow_unknown_field_,
      allow_unknown_extension_, allow_unknown_enum_, allow_field_number_,
      allow_relaxed_whitespace_, allow_partial_, recursion_limit_);
  return parser.ParseField(field, output);
}

/* static */ bool TextFormat::Parse(io::ZeroCopyInputStream* input,
                                    Message* output) {
  return Parser().Parse(input, output);
}

/* static */ bool TextFormat::Merge(io::ZeroCopyInputStream* input,
                                    Message* output) {
  return Parser().Merge(input, output);
}

/* static */ bool TextFormat::ParseFromString(y_absl::string_view input,
                                              Message* output) {
  return Parser().ParseFromString(input, output);
}

/* static */ bool TextFormat::MergeFromString(y_absl::string_view input,
                                              Message* output) {
  return Parser().MergeFromString(input, output);
}

#undef DO

// ===========================================================================

TextFormat::BaseTextGenerator::~BaseTextGenerator() {}

namespace {

// A BaseTextGenerator that writes to a string.
class StringBaseTextGenerator : public TextFormat::BaseTextGenerator {
 public:
  void Print(const char* text, size_t size) override {
    output_.append(text, size);
  }

  TProtoStringType Consume() && { return std::move(output_); }

 private:
  TProtoStringType output_;
};

}  // namespace

// The default implementation for FieldValuePrinter. We just delegate the
// implementation to the default FastFieldValuePrinter to avoid duplicating the
// logic.
TextFormat::FieldValuePrinter::FieldValuePrinter() {}
TextFormat::FieldValuePrinter::~FieldValuePrinter() {}

#define FORWARD_IMPL(fn, ...)            \
  StringBaseTextGenerator generator;     \
  delegate_.fn(__VA_ARGS__, &generator); \
  return std::move(generator).Consume()

TProtoStringType TextFormat::FieldValuePrinter::PrintBool(bool val) const {
  FORWARD_IMPL(PrintBool, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintInt32(arc_i32 val) const {
  FORWARD_IMPL(PrintInt32, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintUInt32(arc_ui32 val) const {
  FORWARD_IMPL(PrintUInt32, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintInt64(arc_i64 val) const {
  FORWARD_IMPL(PrintInt64, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintUInt64(arc_ui64 val) const {
  FORWARD_IMPL(PrintUInt64, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintFloat(float val) const {
  FORWARD_IMPL(PrintFloat, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintDouble(double val) const {
  FORWARD_IMPL(PrintDouble, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintString(
    const TProtoStringType& val) const {
  FORWARD_IMPL(PrintString, val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintBytes(
    const TProtoStringType& val) const {
  return PrintString(val);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintEnum(
    arc_i32 val, const TProtoStringType& name) const {
  FORWARD_IMPL(PrintEnum, val, name);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintFieldName(
    const Message& message, const Reflection* reflection,
    const FieldDescriptor* field) const {
  FORWARD_IMPL(PrintFieldName, message, reflection, field);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintMessageStart(
    const Message& message, int field_index, int field_count,
    bool single_line_mode) const {
  FORWARD_IMPL(PrintMessageStart, message, field_index, field_count,
               single_line_mode);
}
TProtoStringType TextFormat::FieldValuePrinter::PrintMessageEnd(
    const Message& message, int field_index, int field_count,
    bool single_line_mode) const {
  FORWARD_IMPL(PrintMessageEnd, message, field_index, field_count,
               single_line_mode);
}
#undef FORWARD_IMPL

TextFormat::FastFieldValuePrinter::FastFieldValuePrinter() {}
TextFormat::FastFieldValuePrinter::~FastFieldValuePrinter() {}
void TextFormat::FastFieldValuePrinter::PrintBool(
    bool val, BaseTextGenerator* generator) const {
  if (val) {
    generator->PrintLiteral("true");
  } else {
    generator->PrintLiteral("false");
  }
}
void TextFormat::FastFieldValuePrinter::PrintInt32(
    arc_i32 val, BaseTextGenerator* generator) const {
  generator->PrintString(y_absl::StrCat(val));
}
void TextFormat::FastFieldValuePrinter::PrintUInt32(
    arc_ui32 val, BaseTextGenerator* generator) const {
  generator->PrintString(y_absl::StrCat(val));
}
void TextFormat::FastFieldValuePrinter::PrintInt64(
    arc_i64 val, BaseTextGenerator* generator) const {
  generator->PrintString(y_absl::StrCat(val));
}
void TextFormat::FastFieldValuePrinter::PrintUInt64(
    arc_ui64 val, BaseTextGenerator* generator) const {
  generator->PrintString(y_absl::StrCat(val));
}
void TextFormat::FastFieldValuePrinter::PrintFloat(
    float val, BaseTextGenerator* generator) const {
  generator->PrintString(!std::isnan(val) ? io::SimpleFtoa(val) : "nan");
}
void TextFormat::FastFieldValuePrinter::PrintDouble(
    double val, BaseTextGenerator* generator) const {
  generator->PrintString(!std::isnan(val) ? io::SimpleDtoa(val) : "nan");
}
void TextFormat::FastFieldValuePrinter::PrintEnum(
    arc_i32 /*val*/, const TProtoStringType& name,
    BaseTextGenerator* generator) const {
  generator->PrintString(name);
}

void TextFormat::FastFieldValuePrinter::PrintString(
    const TProtoStringType& val, BaseTextGenerator* generator) const {
  generator->PrintLiteral("\"");
  generator->PrintString(y_absl::CEscape(val));
  generator->PrintLiteral("\"");
}
void TextFormat::FastFieldValuePrinter::PrintBytes(
    const TProtoStringType& val, BaseTextGenerator* generator) const {
  PrintString(val, generator);
}
void TextFormat::FastFieldValuePrinter::PrintFieldName(
    const Message& message, int /*field_index*/, int /*field_count*/,
    const Reflection* reflection, const FieldDescriptor* field,
    BaseTextGenerator* generator) const {
  PrintFieldName(message, reflection, field, generator);
}
void TextFormat::FastFieldValuePrinter::PrintFieldName(
    const Message& /*message*/, const Reflection* /*reflection*/,
    const FieldDescriptor* field, BaseTextGenerator* generator) const {
  if (field->is_extension()) {
    generator->PrintLiteral("[");
    generator->PrintString(field->PrintableNameForExtension());
    generator->PrintLiteral("]");
  } else if (field->type() == FieldDescriptor::TYPE_GROUP) {
    // Groups must be serialized with their original capitalization.
    generator->PrintString(field->message_type()->name());
  } else {
    generator->PrintString(field->name());
  }
}
void TextFormat::FastFieldValuePrinter::PrintMessageStart(
    const Message& /*message*/, int /*field_index*/, int /*field_count*/,
    bool single_line_mode, BaseTextGenerator* generator) const {
  if (single_line_mode) {
    generator->PrintLiteral(" { ");
  } else {
    generator->PrintLiteral(" {\n");
  }
}
bool TextFormat::FastFieldValuePrinter::PrintMessageContent(
    const Message& /*message*/, int /*field_index*/, int /*field_count*/,
    bool /*single_line_mode*/, BaseTextGenerator* /*generator*/) const {
  return false;  // Use the default printing function.
}
void TextFormat::FastFieldValuePrinter::PrintMessageEnd(
    const Message& /*message*/, int /*field_index*/, int /*field_count*/,
    bool single_line_mode, BaseTextGenerator* generator) const {
  if (single_line_mode) {
    generator->PrintLiteral("} ");
  } else {
    generator->PrintLiteral("}\n");
  }
}

namespace {

// A legacy compatibility wrapper. Takes ownership of the delegate.
class FieldValuePrinterWrapper : public TextFormat::FastFieldValuePrinter {
 public:
  explicit FieldValuePrinterWrapper(
      const TextFormat::FieldValuePrinter* delegate)
      : delegate_(delegate) {}

  void SetDelegate(const TextFormat::FieldValuePrinter* delegate) {
    delegate_.reset(delegate);
  }

  void PrintBool(bool val,
                 TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintBool(val));
  }
  void PrintInt32(arc_i32 val,
                  TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintInt32(val));
  }
  void PrintUInt32(arc_ui32 val,
                   TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintUInt32(val));
  }
  void PrintInt64(arc_i64 val,
                  TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintInt64(val));
  }
  void PrintUInt64(arc_ui64 val,
                   TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintUInt64(val));
  }
  void PrintFloat(float val,
                  TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintFloat(val));
  }
  void PrintDouble(double val,
                   TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintDouble(val));
  }
  void PrintString(const TProtoStringType& val,
                   TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintString(val));
  }
  void PrintBytes(const TProtoStringType& val,
                  TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintBytes(val));
  }
  void PrintEnum(arc_i32 val, const TProtoStringType& name,
                 TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintEnum(val, name));
  }
  void PrintFieldName(const Message& message, int /*field_index*/,
                      int /*field_count*/, const Reflection* reflection,
                      const FieldDescriptor* field,
                      TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(
        delegate_->PrintFieldName(message, reflection, field));
  }
  void PrintFieldName(const Message& message, const Reflection* reflection,
                      const FieldDescriptor* field,
                      TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(
        delegate_->PrintFieldName(message, reflection, field));
  }
  void PrintMessageStart(
      const Message& message, int field_index, int field_count,
      bool single_line_mode,
      TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintMessageStart(
        message, field_index, field_count, single_line_mode));
  }
  void PrintMessageEnd(
      const Message& message, int field_index, int field_count,
      bool single_line_mode,
      TextFormat::BaseTextGenerator* generator) const override {
    generator->PrintString(delegate_->PrintMessageEnd(
        message, field_index, field_count, single_line_mode));
  }

 private:
  std::unique_ptr<const TextFormat::FieldValuePrinter> delegate_;
};

}  // namespace

TextFormat::Printer::Printer()
    : initial_indent_level_(0),
      single_line_mode_(false),
      use_field_number_(false),
      use_short_repeated_primitives_(false),
      insert_silent_marker_(false),
      redact_debug_string_(false),
      randomize_debug_string_(false),
      hide_unknown_fields_(false),
      print_message_fields_in_index_order_(false),
      expand_any_(false),
      truncate_string_field_longer_than_(0LL),
      finder_(nullptr) {
  SetUseUtf8StringEscaping(false);
}

void TextFormat::Printer::SetUseUtf8StringEscaping(bool as_utf8) {
  SetDefaultFieldValuePrinter(as_utf8 ? new FastFieldValuePrinterUtf8Escaping()
                                      : new DebugStringFieldValuePrinter());
}

void TextFormat::Printer::SetDefaultFieldValuePrinter(
    const FieldValuePrinter* printer) {
  default_field_value_printer_.reset(new FieldValuePrinterWrapper(printer));
}

void TextFormat::Printer::SetDefaultFieldValuePrinter(
    const FastFieldValuePrinter* printer) {
  default_field_value_printer_.reset(printer);
}

bool TextFormat::Printer::RegisterFieldValuePrinter(
    const FieldDescriptor* field, const FieldValuePrinter* printer) {
  if (field == nullptr || printer == nullptr) {
    return false;
  }
  std::unique_ptr<FieldValuePrinterWrapper> wrapper(
      new FieldValuePrinterWrapper(nullptr));
  auto pair = custom_printers_.emplace(field, nullptr);
  if (pair.second) {
    wrapper->SetDelegate(printer);
    pair.first->second = std::move(wrapper);
    return true;
  } else {
    return false;
  }
}

bool TextFormat::Printer::RegisterFieldValuePrinter(
    const FieldDescriptor* field, const FastFieldValuePrinter* printer) {
  if (field == nullptr || printer == nullptr) {
    return false;
  }
  auto pair = custom_printers_.emplace(field, nullptr);
  if (pair.second) {
    pair.first->second.reset(printer);
    return true;
  } else {
    return false;
  }
}

bool TextFormat::Printer::RegisterMessagePrinter(
    const Descriptor* descriptor, const MessagePrinter* printer) {
  if (descriptor == nullptr || printer == nullptr) {
    return false;
  }
  auto pair = custom_message_printers_.emplace(descriptor, nullptr);
  if (pair.second) {
    pair.first->second.reset(printer);
    return true;
  } else {
    return false;
  }
}

bool TextFormat::Printer::PrintToString(const Message& message,
                                        TProtoStringType* output) const {
  Y_ABSL_DCHECK(output) << "output specified is nullptr";

  output->clear();
  io::StringOutputStream output_stream(output);

  return Print(message, &output_stream);
}

bool TextFormat::Printer::PrintUnknownFieldsToString(
    const UnknownFieldSet& unknown_fields, TProtoStringType* output) const {
  Y_ABSL_DCHECK(output) << "output specified is nullptr";

  output->clear();
  io::StringOutputStream output_stream(output);
  return PrintUnknownFields(unknown_fields, &output_stream);
}

bool TextFormat::Printer::Print(const Message& message,
                                io::ZeroCopyOutputStream* output) const {
  TextGenerator generator(output, insert_silent_marker_, initial_indent_level_);


  Print(message, &generator);

  // Output false if the generator failed internally.
  return !generator.failed();
}

// Maximum recursion depth for heuristically printing out length-delimited
// unknown fields as messages.
static constexpr int kUnknownFieldRecursionLimit = 10;

bool TextFormat::Printer::PrintUnknownFields(
    const UnknownFieldSet& unknown_fields,
    io::ZeroCopyOutputStream* output) const {
  TextGenerator generator(output, initial_indent_level_);

  PrintUnknownFields(unknown_fields, &generator, kUnknownFieldRecursionLimit);

  // Output false if the generator failed internally.
  return !generator.failed();
}

namespace {
// Comparison functor for sorting FieldDescriptors by field index.
// Normal fields have higher precedence than extensions.
struct FieldIndexSorter {
  bool operator()(const FieldDescriptor* left,
                  const FieldDescriptor* right) const {
    if (left->is_extension() && right->is_extension()) {
      return left->number() < right->number();
    } else if (left->is_extension()) {
      return false;
    } else if (right->is_extension()) {
      return true;
    } else {
      return left->index() < right->index();
    }
  }
};

}  // namespace

bool TextFormat::Printer::PrintAny(const Message& message,
                                   BaseTextGenerator* generator) const {
  const FieldDescriptor* type_url_field;
  const FieldDescriptor* value_field;
  if (!internal::GetAnyFieldDescriptors(message, &type_url_field,
                                        &value_field)) {
    return false;
  }

  const Reflection* reflection = message.GetReflection();

  // Extract the full type name from the type_url field.
  const TProtoStringType& type_url = reflection->GetString(message, type_url_field);
  TProtoStringType url_prefix;
  TProtoStringType full_type_name;
  if (!internal::ParseAnyTypeUrl(type_url, &url_prefix, &full_type_name)) {
    return false;
  }

  // Print the "value" in text.
  const Descriptor* value_descriptor =
      finder_ ? finder_->FindAnyType(message, url_prefix, full_type_name)
              : DefaultFinderFindAnyType(message, url_prefix, full_type_name);
  if (value_descriptor == nullptr) {
    Y_ABSL_LOG(WARNING) << "Can't print proto content: proto type " << type_url
                      << " not found";
    return false;
  }
  DynamicMessageFactory factory;
  std::unique_ptr<Message> value_message(
      factory.GetPrototype(value_descriptor)->New());
  TProtoStringType serialized_value = reflection->GetString(message, value_field);
  if (!value_message->ParseFromString(serialized_value)) {
    Y_ABSL_LOG(WARNING) << type_url << ": failed to parse contents";
    return false;
  }
  generator->PrintLiteral("[");
  generator->PrintString(type_url);
  generator->PrintLiteral("]");
  const FastFieldValuePrinter* printer = GetFieldPrinter(value_field);
  printer->PrintMessageStart(message, -1, 0, single_line_mode_, generator);
  generator->Indent();
  Print(*value_message, generator);
  generator->Outdent();
  printer->PrintMessageEnd(message, -1, 0, single_line_mode_, generator);
  return true;
}

void TextFormat::Printer::Print(const Message& message,
                                BaseTextGenerator* generator) const {
  const Reflection* reflection = message.GetReflection();
  if (!reflection) {
    // This message does not provide any way to describe its structure.
    // Parse it again in an UnknownFieldSet, and display this instead.
    UnknownFieldSet unknown_fields;
    {
      TProtoStringType serialized = message.SerializeAsString();
      io::ArrayInputStream input(serialized.data(), serialized.size());
      unknown_fields.ParseFromZeroCopyStream(&input);
    }
    PrintUnknownFields(unknown_fields, generator, kUnknownFieldRecursionLimit);
    return;
  }
  const Descriptor* descriptor = message.GetDescriptor();
  auto itr = custom_message_printers_.find(descriptor);
  if (itr != custom_message_printers_.end()) {
    itr->second->Print(message, single_line_mode_, generator);
    return;
  }
  PrintMessage(message, generator);
}

void TextFormat::Printer::PrintMessage(const Message& message,
                                       BaseTextGenerator* generator) const {
  if (generator == nullptr) {
    return;
  }
  const Descriptor* descriptor = message.GetDescriptor();
  if (descriptor->full_name() == internal::kAnyFullTypeName && expand_any_ &&
      PrintAny(message, generator)) {
    return;
  }
  const Reflection* reflection = message.GetReflection();
  std::vector<const FieldDescriptor*> fields;
  if (descriptor->options().map_entry()) {
    fields.push_back(descriptor->field(0));
    fields.push_back(descriptor->field(1));
  } else {
    reflection->ListFields(message, &fields);
  }

  if (print_message_fields_in_index_order_) {
    std::sort(fields.begin(), fields.end(), FieldIndexSorter());
  }
  for (const FieldDescriptor* field : fields) {
    PrintField(message, reflection, field, generator);
  }
  if (!hide_unknown_fields_) {
    PrintUnknownFields(reflection->GetUnknownFields(message), generator,
                       kUnknownFieldRecursionLimit);
  }
}

void TextFormat::Printer::PrintFieldValueToString(const Message& message,
                                                  const FieldDescriptor* field,
                                                  int index,
                                                  TProtoStringType* output) const {
  Y_ABSL_DCHECK(output) << "output specified is nullptr";

  output->clear();
  io::StringOutputStream output_stream(output);
  TextGenerator generator(&output_stream, initial_indent_level_);

  PrintFieldValue(message, message.GetReflection(), field, index, &generator);
}

class MapEntryMessageComparator {
 public:
  explicit MapEntryMessageComparator(const Descriptor* descriptor)
      : field_(descriptor->field(0)) {}

  bool operator()(const Message* a, const Message* b) {
    const Reflection* reflection = a->GetReflection();
    switch (field_->cpp_type()) {
      case FieldDescriptor::CPPTYPE_BOOL: {
        bool first = reflection->GetBool(*a, field_);
        bool second = reflection->GetBool(*b, field_);
        return first < second;
      }
      case FieldDescriptor::CPPTYPE_INT32: {
        arc_i32 first = reflection->GetInt32(*a, field_);
        arc_i32 second = reflection->GetInt32(*b, field_);
        return first < second;
      }
      case FieldDescriptor::CPPTYPE_INT64: {
        arc_i64 first = reflection->GetInt64(*a, field_);
        arc_i64 second = reflection->GetInt64(*b, field_);
        return first < second;
      }
      case FieldDescriptor::CPPTYPE_UINT32: {
        arc_ui32 first = reflection->GetUInt32(*a, field_);
        arc_ui32 second = reflection->GetUInt32(*b, field_);
        return first < second;
      }
      case FieldDescriptor::CPPTYPE_UINT64: {
        arc_ui64 first = reflection->GetUInt64(*a, field_);
        arc_ui64 second = reflection->GetUInt64(*b, field_);
        return first < second;
      }
      case FieldDescriptor::CPPTYPE_STRING: {
        TProtoStringType first = reflection->GetString(*a, field_);
        TProtoStringType second = reflection->GetString(*b, field_);
        return first < second;
      }
      default:
        Y_ABSL_DLOG(FATAL) << "Invalid key for map field.";
        return true;
    }
  }

 private:
  const FieldDescriptor* field_;
};

namespace internal {

struct MapEntries {
  std::vector<std::unique_ptr<const Message>> owned_entries;
  std::vector<const Message*> all_entries;
};

class MapFieldPrinterHelper {
 public:
  // DynamicMapSorter::Sort cannot be used because it enforces syncing with
  // repeated field.
  static MapEntries SortMap(const Message& message,
                            const Reflection* reflection,
                            const FieldDescriptor* field);
  static void CopyKey(const MapKey& key, Message* message,
                      const FieldDescriptor* field_desc);
  static void CopyValue(const MapValueRef& value, Message* message,
                        const FieldDescriptor* field_desc);
};

MapEntries MapFieldPrinterHelper::SortMap(const Message& message,
                                          const Reflection* reflection,
                                          const FieldDescriptor* field) {
  const MapFieldBase& base = *reflection->GetMapData(message, field);

  std::vector<const Message*> all_entries;
  std::vector<std::unique_ptr<const Message>> owned_entries;
  if (base.IsRepeatedFieldValid()) {
    const RepeatedPtrField<Message>& map_field =
        reflection->GetRepeatedPtrFieldInternal<Message>(message, field);
    all_entries.reserve(map_field.size());
    for (int i = 0; i < map_field.size(); ++i) {
      all_entries.push_back(
          const_cast<RepeatedPtrField<Message>*>(&map_field)->Mutable(i));
    }
  } else {
    // TODO(teboring): For performance, instead of creating map entry message
    // for each element, just store map keys and sort them.
    const Descriptor* map_entry_desc = field->message_type();
    const Message* prototype =
        reflection->GetMessageFactory()->GetPrototype(map_entry_desc);
    all_entries.reserve(reflection->MapSize(message, field));
    owned_entries.reserve(reflection->MapSize(message, field));
    for (MapIterator iter =
             reflection->MapBegin(const_cast<Message*>(&message), field);
         iter != reflection->MapEnd(const_cast<Message*>(&message), field);
         ++iter) {
      std::unique_ptr<Message> map_entry_message =
          y_absl::WrapUnique(prototype->New());
      CopyKey(iter.GetKey(), map_entry_message.get(), map_entry_desc->field(0));
      CopyValue(iter.GetValueRef(), map_entry_message.get(),
                map_entry_desc->field(1));
      all_entries.push_back(map_entry_message.get());
      owned_entries.push_back(std::move(map_entry_message));
    }
  }

  std::stable_sort(all_entries.begin(), all_entries.end(),
                   MapEntryMessageComparator(field->message_type()));
  return {std::move(owned_entries), std::move(all_entries)};
}

void MapFieldPrinterHelper::CopyKey(const MapKey& key, Message* message,
                                    const FieldDescriptor* field_desc) {
  const Reflection* reflection = message->GetReflection();
  switch (field_desc->cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
    case FieldDescriptor::CPPTYPE_FLOAT:
    case FieldDescriptor::CPPTYPE_ENUM:
    case FieldDescriptor::CPPTYPE_MESSAGE:
      Y_ABSL_LOG(ERROR) << "Not supported.";
      break;
    case FieldDescriptor::CPPTYPE_STRING:
      reflection->SetString(message, field_desc, key.GetStringValue());
      return;
    case FieldDescriptor::CPPTYPE_INT64:
      reflection->SetInt64(message, field_desc, key.GetInt64Value());
      return;
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->SetInt32(message, field_desc, key.GetInt32Value());
      return;
    case FieldDescriptor::CPPTYPE_UINT64:
      reflection->SetUInt64(message, field_desc, key.GetUInt64Value());
      return;
    case FieldDescriptor::CPPTYPE_UINT32:
      reflection->SetUInt32(message, field_desc, key.GetUInt32Value());
      return;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(message, field_desc, key.GetBoolValue());
      return;
  }
}

void MapFieldPrinterHelper::CopyValue(const MapValueRef& value,
                                      Message* message,
                                      const FieldDescriptor* field_desc) {
  const Reflection* reflection = message->GetReflection();
  switch (field_desc->cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetDouble(message, field_desc, value.GetDoubleValue());
      return;
    case FieldDescriptor::CPPTYPE_FLOAT:
      reflection->SetFloat(message, field_desc, value.GetFloatValue());
      return;
    case FieldDescriptor::CPPTYPE_ENUM:
      reflection->SetEnumValue(message, field_desc, value.GetEnumValue());
      return;
    case FieldDescriptor::CPPTYPE_MESSAGE: {
      Message* sub_message = value.GetMessageValue().New();
      sub_message->CopyFrom(value.GetMessageValue());
      reflection->SetAllocatedMessage(message, sub_message, field_desc);
      return;
    }
    case FieldDescriptor::CPPTYPE_STRING:
      reflection->SetString(message, field_desc, value.GetStringValue());
      return;
    case FieldDescriptor::CPPTYPE_INT64:
      reflection->SetInt64(message, field_desc, value.GetInt64Value());
      return;
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->SetInt32(message, field_desc, value.GetInt32Value());
      return;
    case FieldDescriptor::CPPTYPE_UINT64:
      reflection->SetUInt64(message, field_desc, value.GetUInt64Value());
      return;
    case FieldDescriptor::CPPTYPE_UINT32:
      reflection->SetUInt32(message, field_desc, value.GetUInt32Value());
      return;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(message, field_desc, value.GetBoolValue());
      return;
  }
}
}  // namespace internal

void TextFormat::Printer::PrintField(const Message& message,
                                     const Reflection* reflection,
                                     const FieldDescriptor* field,
                                     BaseTextGenerator* generator) const {
  if (use_short_repeated_primitives_ && field->is_repeated() &&
      field->cpp_type() != FieldDescriptor::CPPTYPE_STRING &&
      field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
    PrintShortRepeatedField(message, reflection, field, generator);
    return;
  }

  int count = 0;

  if (field->is_repeated()) {
    count = reflection->FieldSize(message, field);
  } else if (reflection->HasField(message, field) ||
             field->containing_type()->options().map_entry()) {
    count = 1;
  }

  bool is_map = field->is_map();
  const internal::MapEntries map_entries =
      is_map
          ? internal::MapFieldPrinterHelper::SortMap(message, reflection, field)
          : internal::MapEntries();

  for (int j = 0; j < count; ++j) {
    const int field_index = field->is_repeated() ? j : -1;

    PrintFieldName(message, field_index, count, reflection, field, generator);

    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      if (TryRedactFieldValue(message, field, generator,
                              /*insert_value_separator=*/true)) {
        break;
      }
      const FastFieldValuePrinter* printer = GetFieldPrinter(field);
      const Message& sub_message =
          field->is_repeated()
              ? (is_map ? *map_entries.all_entries[j]
                        : reflection->GetRepeatedMessage(message, field, j))
              : reflection->GetMessage(message, field);
      printer->PrintMessageStart(sub_message, field_index, count,
                                 single_line_mode_, generator);
      generator->Indent();
      if (!printer->PrintMessageContent(sub_message, field_index, count,
                                        single_line_mode_, generator)) {
        Print(sub_message, generator);
      }
      generator->Outdent();
      printer->PrintMessageEnd(sub_message, field_index, count,
                               single_line_mode_, generator);
    } else {
      generator->PrintMaybeWithMarker(MarkerToken(), ": ");
      // Write the field value.
      PrintFieldValue(message, reflection, field, field_index, generator);
      if (single_line_mode_) {
        generator->PrintLiteral(" ");
      } else {
        generator->PrintLiteral("\n");
      }
    }
  }
}

void TextFormat::Printer::PrintShortRepeatedField(
    const Message& message, const Reflection* reflection,
    const FieldDescriptor* field, BaseTextGenerator* generator) const {
  // Print primitive repeated field in short form.
  int size = reflection->FieldSize(message, field);
  PrintFieldName(message, /*field_index=*/-1, /*field_count=*/size, reflection,
                 field, generator);
  generator->PrintMaybeWithMarker(MarkerToken(), ": ", "[");
  for (int i = 0; i < size; i++) {
    if (i > 0) generator->PrintLiteral(", ");
    PrintFieldValue(message, reflection, field, i, generator);
  }
  if (single_line_mode_) {
    generator->PrintLiteral("] ");
  } else {
    generator->PrintLiteral("]\n");
  }
}

void TextFormat::Printer::PrintFieldName(const Message& message,
                                         int field_index, int field_count,
                                         const Reflection* reflection,
                                         const FieldDescriptor* field,
                                         BaseTextGenerator* generator) const {
  // if use_field_number_ is true, prints field number instead
  // of field name.
  if (use_field_number_) {
    generator->PrintString(y_absl::StrCat(field->number()));
    return;
  }

  const FastFieldValuePrinter* printer = GetFieldPrinter(field);
  printer->PrintFieldName(message, field_index, field_count, reflection, field,
                          generator);
}

void TextFormat::Printer::PrintFieldValue(const Message& message,
                                          const Reflection* reflection,
                                          const FieldDescriptor* field,
                                          int index,
                                          BaseTextGenerator* generator) const {
  Y_ABSL_DCHECK(field->is_repeated() || (index == -1))
      << "Index must be -1 for non-repeated fields";

  const FastFieldValuePrinter* printer = GetFieldPrinter(field);
  if (TryRedactFieldValue(message, field, generator,
                          /*insert_value_separator=*/false)) {
    return;
  }

  switch (field->cpp_type()) {
#define OUTPUT_FIELD(CPPTYPE, METHOD)                                \
  case FieldDescriptor::CPPTYPE_##CPPTYPE:                           \
    printer->Print##METHOD(                                          \
        field->is_repeated()                                         \
            ? reflection->GetRepeated##METHOD(message, field, index) \
            : reflection->Get##METHOD(message, field),               \
        generator);                                                  \
    break

    OUTPUT_FIELD(INT32, Int32);
    OUTPUT_FIELD(INT64, Int64);
    OUTPUT_FIELD(UINT32, UInt32);
    OUTPUT_FIELD(UINT64, UInt64);
    OUTPUT_FIELD(FLOAT, Float);
    OUTPUT_FIELD(DOUBLE, Double);
    OUTPUT_FIELD(BOOL, Bool);
#undef OUTPUT_FIELD

    case FieldDescriptor::CPPTYPE_STRING: {
      TProtoStringType scratch;
      const TProtoStringType& value =
          field->is_repeated()
              ? reflection->GetRepeatedStringReference(message, field, index,
                                                       &scratch)
              : reflection->GetStringReference(message, field, &scratch);
      const TProtoStringType* value_to_print = &value;
      TProtoStringType truncated_value;
      if (truncate_string_field_longer_than_ > 0 &&
          static_cast<size_t>(truncate_string_field_longer_than_) <
              value.size()) {
        truncated_value = value.substr(0, truncate_string_field_longer_than_) +
                          "...<truncated>...";
        value_to_print = &truncated_value;
      }
      if (field->type() == FieldDescriptor::TYPE_STRING) {
        printer->PrintString(*value_to_print, generator);
      } else {
        Y_ABSL_DCHECK_EQ(field->type(), FieldDescriptor::TYPE_BYTES);
        printer->PrintBytes(*value_to_print, generator);
      }
      break;
    }

    case FieldDescriptor::CPPTYPE_ENUM: {
      int enum_value =
          field->is_repeated()
              ? reflection->GetRepeatedEnumValue(message, field, index)
              : reflection->GetEnumValue(message, field);
      const EnumValueDescriptor* enum_desc =
          field->enum_type()->FindValueByNumber(enum_value);
      if (enum_desc != nullptr) {
        printer->PrintEnum(enum_value, enum_desc->name(), generator);
      } else {
        // Ordinarily, enum_desc should not be null, because proto2 has the
        // invariant that set enum field values must be in-range, but with the
        // new integer-based API for enums (or the RepeatedField<int> loophole),
        // it is possible for the user to force an unknown integer value.  So we
        // simply use the integer value itself as the enum value name in this
        // case.
        printer->PrintEnum(enum_value, y_absl::StrCat(enum_value), generator);
      }
      break;
    }

    case FieldDescriptor::CPPTYPE_MESSAGE:
      Print(field->is_repeated()
                ? reflection->GetRepeatedMessage(message, field, index)
                : reflection->GetMessage(message, field),
            generator);
      break;
  }
}

/* static */ bool TextFormat::Print(const Message& message,
                                    io::ZeroCopyOutputStream* output) {
  return Printer().Print(message, output);
}

/* static */ bool TextFormat::PrintUnknownFields(
    const UnknownFieldSet& unknown_fields, io::ZeroCopyOutputStream* output) {
  return Printer().PrintUnknownFields(unknown_fields, output);
}

/* static */ bool TextFormat::PrintToString(const Message& message,
                                            TProtoStringType* output) {
  return Printer().PrintToString(message, output);
}

/* static */ bool TextFormat::PrintUnknownFieldsToString(
    const UnknownFieldSet& unknown_fields, TProtoStringType* output) {
  return Printer().PrintUnknownFieldsToString(unknown_fields, output);
}

/* static */ void TextFormat::PrintFieldValueToString(
    const Message& message, const FieldDescriptor* field, int index,
    TProtoStringType* output) {
  return Printer().PrintFieldValueToString(message, field, index, output);
}

/* static */ bool TextFormat::ParseFieldValueFromString(
    const TProtoStringType& input, const FieldDescriptor* field, Message* message) {
  return Parser().ParseFieldValueFromString(input, field, message);
}

void TextFormat::Printer::PrintUnknownFields(
    const UnknownFieldSet& unknown_fields, BaseTextGenerator* generator,
    int recursion_budget) const {
  for (int i = 0; i < unknown_fields.field_count(); i++) {
    const UnknownField& field = unknown_fields.field(i);
    TProtoStringType field_number = y_absl::StrCat(field.number());

    switch (field.type()) {
      case UnknownField::TYPE_VARINT:
        generator->PrintString(field_number);
        generator->PrintMaybeWithMarker(MarkerToken(), ": ");
        generator->PrintString(y_absl::StrCat(field.varint()));
        if (single_line_mode_) {
          generator->PrintLiteral(" ");
        } else {
          generator->PrintLiteral("\n");
        }
        break;
      case UnknownField::TYPE_FIXED32: {
        generator->PrintString(field_number);
        generator->PrintMaybeWithMarker(MarkerToken(), ": ", "0x");
        generator->PrintString(
            y_absl::StrCat(y_absl::Hex(field.fixed32(), y_absl::kZeroPad8)));
        if (single_line_mode_) {
          generator->PrintLiteral(" ");
        } else {
          generator->PrintLiteral("\n");
        }
        break;
      }
      case UnknownField::TYPE_FIXED64: {
        generator->PrintString(field_number);
        generator->PrintMaybeWithMarker(MarkerToken(), ": ", "0x");
        generator->PrintString(
            y_absl::StrCat(y_absl::Hex(field.fixed64(), y_absl::kZeroPad16)));
        if (single_line_mode_) {
          generator->PrintLiteral(" ");
        } else {
          generator->PrintLiteral("\n");
        }
        break;
      }
      case UnknownField::TYPE_LENGTH_DELIMITED: {
        generator->PrintString(field_number);
        const TProtoStringType& value = field.length_delimited();
        // We create a CodedInputStream so that we can adhere to our recursion
        // budget when we attempt to parse the data. UnknownFieldSet parsing is
        // recursive because of groups.
        io::CodedInputStream input_stream(
            reinterpret_cast<const uint8_t*>(value.data()), value.size());
        input_stream.SetRecursionLimit(recursion_budget);
        UnknownFieldSet embedded_unknown_fields;
        if (!value.empty() && recursion_budget > 0 &&
            embedded_unknown_fields.ParseFromCodedStream(&input_stream)) {
          // This field is parseable as a Message.
          // So it is probably an embedded message.
          if (single_line_mode_) {
            generator->PrintMaybeWithMarker(MarkerToken(), " ", "{ ");
          } else {
            generator->PrintMaybeWithMarker(MarkerToken(), " ", "{\n");
            generator->Indent();
          }
          PrintUnknownFields(embedded_unknown_fields, generator,
                             recursion_budget - 1);
          if (single_line_mode_) {
            generator->PrintLiteral("} ");
          } else {
            generator->Outdent();
            generator->PrintLiteral("}\n");
          }
        } else {
          // This field is not parseable as a Message (or we ran out of
          // recursion budget). So it is probably just a plain string.
          generator->PrintMaybeWithMarker(MarkerToken(), ": ", "\"");
          generator->PrintString(y_absl::CEscape(value));
          if (single_line_mode_) {
            generator->PrintLiteral("\" ");
          } else {
            generator->PrintLiteral("\"\n");
          }
        }
        break;
      }
      case UnknownField::TYPE_GROUP:
        generator->PrintString(field_number);
        if (single_line_mode_) {
          generator->PrintMaybeWithMarker(MarkerToken(), " ", "{ ");
        } else {
          generator->PrintMaybeWithMarker(MarkerToken(), " ", "{\n");
          generator->Indent();
        }
        // For groups, we recurse without checking the budget. This is OK,
        // because if the groups were too deeply nested then we would have
        // already rejected the message when we originally parsed it.
        PrintUnknownFields(field.group(), generator, recursion_budget - 1);
        if (single_line_mode_) {
          generator->PrintLiteral("} ");
        } else {
          generator->Outdent();
          generator->PrintLiteral("}\n");
        }
        break;
    }
  }
}

bool TextFormat::Printer::TryRedactFieldValue(
    const Message& message, const FieldDescriptor* field,
    BaseTextGenerator* generator, bool insert_value_separator) const {
  auto do_redact = [&](const TProtoStringType& replacement) {
    if (insert_value_separator) {
      generator->PrintMaybeWithMarker(MarkerToken(), ": ");
    }
    generator->PrintString(replacement);
    if (insert_value_separator) {
      if (single_line_mode_) {
        generator->PrintLiteral(" ");
      } else {
        generator->PrintLiteral("\n");
      }
    }
  };

  if (redact_debug_string_ && field->options().debug_redact()) {
    do_redact("[REDACTED]");
    return true;
  }
  return false;
}

}  // namespace protobuf
}  // namespace google

#include "google/protobuf/port_undef.inc"
