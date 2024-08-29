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

#include "google/protobuf/compiler/objectivec/helpers.h"

#include <string>
#include <vector>

#include "y_absl/log/absl_log.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/str_replace.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/string_view.h"
#include "google/protobuf/compiler/objectivec/names.h"
#include "google/protobuf/io/strtod.h"
#include "google/protobuf/stubs/common.h"
#include <google/protobuf/stubs/port.h>

// NOTE: src/google/protobuf/compiler/plugin.cc makes use of cerr for some
// error cases, so it seems to be ok to use as a back door for errors.

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

TProtoStringType EscapeTrigraphs(y_absl::string_view to_escape) {
  return y_absl::StrReplaceAll(to_escape, {{"?", "\\?"}});
}

namespace {

TProtoStringType GetZeroEnumNameForFlagType(const FlagType flag_type) {
  switch (flag_type) {
    case FLAGTYPE_DESCRIPTOR_INITIALIZATION:
      return "GPBDescriptorInitializationFlag_None";
    case FLAGTYPE_EXTENSION:
      return "GPBExtensionNone";
    case FLAGTYPE_FIELD:
      return "GPBFieldNone";
    default:
      Y_ABSL_LOG(FATAL) << "Can't get here.";
      return "0";
  }
}

TProtoStringType GetEnumNameForFlagType(const FlagType flag_type) {
  switch (flag_type) {
    case FLAGTYPE_DESCRIPTOR_INITIALIZATION:
      return "GPBDescriptorInitializationFlags";
    case FLAGTYPE_EXTENSION:
      return "GPBExtensionOptions";
    case FLAGTYPE_FIELD:
      return "GPBFieldFlags";
    default:
      Y_ABSL_LOG(FATAL) << "Can't get here.";
      return TProtoStringType();
  }
}

TProtoStringType HandleExtremeFloatingPoint(TProtoStringType val, bool add_float_suffix) {
  if (val == "nan") {
    return "NAN";
  } else if (val == "inf") {
    return "INFINITY";
  } else if (val == "-inf") {
    return "-INFINITY";
  } else {
    // float strings with ., e or E need to have f appended
    if (add_float_suffix &&
        (y_absl::StrContains(val, '.') || y_absl::StrContains(val, 'e') ||
         y_absl::StrContains(val, 'E'))) {
      return y_absl::StrCat(val, "f");
    }
    return val;
  }
}

}  // namespace

TProtoStringType GetCapitalizedType(const FieldDescriptor* field) {
  switch (field->type()) {
    case FieldDescriptor::TYPE_INT32:
      return "Int32";
    case FieldDescriptor::TYPE_UINT32:
      return "UInt32";
    case FieldDescriptor::TYPE_SINT32:
      return "SInt32";
    case FieldDescriptor::TYPE_FIXED32:
      return "Fixed32";
    case FieldDescriptor::TYPE_SFIXED32:
      return "SFixed32";
    case FieldDescriptor::TYPE_INT64:
      return "Int64";
    case FieldDescriptor::TYPE_UINT64:
      return "UInt64";
    case FieldDescriptor::TYPE_SINT64:
      return "SInt64";
    case FieldDescriptor::TYPE_FIXED64:
      return "Fixed64";
    case FieldDescriptor::TYPE_SFIXED64:
      return "SFixed64";
    case FieldDescriptor::TYPE_FLOAT:
      return "Float";
    case FieldDescriptor::TYPE_DOUBLE:
      return "Double";
    case FieldDescriptor::TYPE_BOOL:
      return "Bool";
    case FieldDescriptor::TYPE_STRING:
      return "String";
    case FieldDescriptor::TYPE_BYTES:
      return "Bytes";
    case FieldDescriptor::TYPE_ENUM:
      return "Enum";
    case FieldDescriptor::TYPE_GROUP:
      return "Group";
    case FieldDescriptor::TYPE_MESSAGE:
      return "Message";
  }

  // Some compilers report reaching end of function even though all cases of
  // the enum are handed in the switch.
  Y_ABSL_LOG(FATAL) << "Can't get here.";
  return TProtoStringType();
}

ObjectiveCType GetObjectiveCType(FieldDescriptor::Type field_type) {
  switch (field_type) {
    case FieldDescriptor::TYPE_INT32:
    case FieldDescriptor::TYPE_SINT32:
    case FieldDescriptor::TYPE_SFIXED32:
      return OBJECTIVECTYPE_INT32;

    case FieldDescriptor::TYPE_UINT32:
    case FieldDescriptor::TYPE_FIXED32:
      return OBJECTIVECTYPE_UINT32;

    case FieldDescriptor::TYPE_INT64:
    case FieldDescriptor::TYPE_SINT64:
    case FieldDescriptor::TYPE_SFIXED64:
      return OBJECTIVECTYPE_INT64;

    case FieldDescriptor::TYPE_UINT64:
    case FieldDescriptor::TYPE_FIXED64:
      return OBJECTIVECTYPE_UINT64;

    case FieldDescriptor::TYPE_FLOAT:
      return OBJECTIVECTYPE_FLOAT;

    case FieldDescriptor::TYPE_DOUBLE:
      return OBJECTIVECTYPE_DOUBLE;

    case FieldDescriptor::TYPE_BOOL:
      return OBJECTIVECTYPE_BOOLEAN;

    case FieldDescriptor::TYPE_STRING:
      return OBJECTIVECTYPE_STRING;

    case FieldDescriptor::TYPE_BYTES:
      return OBJECTIVECTYPE_DATA;

    case FieldDescriptor::TYPE_ENUM:
      return OBJECTIVECTYPE_ENUM;

    case FieldDescriptor::TYPE_GROUP:
    case FieldDescriptor::TYPE_MESSAGE:
      return OBJECTIVECTYPE_MESSAGE;
  }

  // Some compilers report reaching end of function even though all cases of
  // the enum are handed in the switch.
  Y_ABSL_LOG(FATAL) << "Can't get here.";
  return OBJECTIVECTYPE_INT32;
}

TProtoStringType GPBGenericValueFieldName(const FieldDescriptor* field) {
  // Returns the field within the GPBGenericValue union to use for the given
  // field.
  if (field->is_repeated()) {
    return "valueMessage";
  }
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32:
      return "valueInt32";
    case FieldDescriptor::CPPTYPE_UINT32:
      return "valueUInt32";
    case FieldDescriptor::CPPTYPE_INT64:
      return "valueInt64";
    case FieldDescriptor::CPPTYPE_UINT64:
      return "valueUInt64";
    case FieldDescriptor::CPPTYPE_FLOAT:
      return "valueFloat";
    case FieldDescriptor::CPPTYPE_DOUBLE:
      return "valueDouble";
    case FieldDescriptor::CPPTYPE_BOOL:
      return "valueBool";
    case FieldDescriptor::CPPTYPE_STRING:
      if (field->type() == FieldDescriptor::TYPE_BYTES) {
        return "valueData";
      } else {
        return "valueString";
      }
    case FieldDescriptor::CPPTYPE_ENUM:
      return "valueEnum";
    case FieldDescriptor::CPPTYPE_MESSAGE:
      return "valueMessage";
  }

  // Some compilers report reaching end of function even though all cases of
  // the enum are handed in the switch.
  Y_ABSL_LOG(FATAL) << "Can't get here.";
  return TProtoStringType();
}

TProtoStringType DefaultValue(const FieldDescriptor* field) {
  // Repeated fields don't have defaults.
  if (field->is_repeated()) {
    return "nil";
  }

  // Switch on cpp_type since we need to know which default_value_* method
  // of FieldDescriptor to call.
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_INT32:
      // gcc and llvm reject the decimal form of kint32min and kint64min.
      if (field->default_value_int32() == INT_MIN) {
        return "-0x80000000";
      }
      return y_absl::StrCat(field->default_value_int32());
    case FieldDescriptor::CPPTYPE_UINT32:
      return y_absl::StrCat(field->default_value_uint32(), "U");
    case FieldDescriptor::CPPTYPE_INT64:
      // gcc and llvm reject the decimal form of kint32min and kint64min.
      if (field->default_value_int64() == LLONG_MIN) {
        return "-0x8000000000000000LL";
      }
      return y_absl::StrCat(field->default_value_int64(), "LL");
    case FieldDescriptor::CPPTYPE_UINT64:
      return y_absl::StrCat(field->default_value_uint64(), "ULL");
    case FieldDescriptor::CPPTYPE_DOUBLE:
      return HandleExtremeFloatingPoint(
          io::SimpleDtoa(field->default_value_double()), false);
    case FieldDescriptor::CPPTYPE_FLOAT:
      return HandleExtremeFloatingPoint(
          io::SimpleFtoa(field->default_value_float()), true);
    case FieldDescriptor::CPPTYPE_BOOL:
      return field->default_value_bool() ? "YES" : "NO";
    case FieldDescriptor::CPPTYPE_STRING: {
      const bool has_default_value = field->has_default_value();
      y_absl::string_view default_string = field->default_value_string();
      if (!has_default_value || default_string.length() == 0) {
        // If the field is defined as being the empty string,
        // then we will just assign to nil, as the empty string is the
        // default for both strings and data.
        return "nil";
      }
      if (field->type() == FieldDescriptor::TYPE_BYTES) {
        // We want constant fields in our data structures so we can
        // declare them as static. To achieve this we cheat and stuff
        // a escaped c string (prefixed with a length) into the data
        // field, and cast it to an (NSData*) so it will compile.
        // The runtime library knows how to handle it.

        // Must convert to a standard byte order for packing length into
        // a cstring.
        arc_ui32 length = ghtonl(default_string.length());
        TProtoStringType bytes((const char*)&length, sizeof(length));
        y_absl::StrAppend(&bytes, default_string);
        return y_absl::StrCat("(NSData*)\"",
                            EscapeTrigraphs(y_absl::CEscape(bytes)), "\"");
      } else {
        return y_absl::StrCat(
            "@\"", EscapeTrigraphs(y_absl::CEscape(default_string)), "\"");
      }
    }
    case FieldDescriptor::CPPTYPE_ENUM:
      return EnumValueName(field->default_value_enum());
    case FieldDescriptor::CPPTYPE_MESSAGE:
      return "nil";
  }

  // Some compilers report reaching end of function even though all cases of
  // the enum are handed in the switch.
  Y_ABSL_LOG(FATAL) << "Can't get here.";
  return TProtoStringType();
}

TProtoStringType BuildFlagsString(FlagType flag_type,
                             const std::vector<TProtoStringType>& strings) {
  if (strings.empty()) {
    return GetZeroEnumNameForFlagType(flag_type);
  } else if (strings.size() == 1) {
    return strings[0];
  }
  TProtoStringType string =
      y_absl::StrCat("(", GetEnumNameForFlagType(flag_type), ")(");
  for (size_t i = 0; i != strings.size(); ++i) {
    if (i > 0) {
      string.append(" | ");
    }
    string.append(strings[i]);
  }
  string.append(")");
  return string;
}

TProtoStringType ObjCClass(y_absl::string_view class_name) {
  return y_absl::StrCat("GPBObjCClass(", class_name, ")");
}

TProtoStringType ObjCClassDeclaration(y_absl::string_view class_name) {
  return y_absl::StrCat("GPBObjCClassDeclaration(", class_name, ");");
}

TProtoStringType BuildCommentsString(const SourceLocation& location,
                                bool prefer_single_line) {
  y_absl::string_view comments = location.leading_comments.empty()
                                   ? location.trailing_comments
                                   : location.leading_comments;
  std::vector<y_absl::string_view> lines;
  lines = y_absl::StrSplit(comments, '\n', y_absl::AllowEmpty());
  while (!lines.empty() && lines.back().empty()) {
    lines.pop_back();
  }
  // If there are no comments, just return an empty string.
  if (lines.empty()) {
    return "";
  }

  TProtoStringType prefix;
  TProtoStringType suffix;
  TProtoStringType final_comments;
  TProtoStringType epilogue;

  bool add_leading_space = false;

  if (prefer_single_line && lines.size() == 1) {
    prefix = "/** ";
    suffix = " */\n";
  } else {
    prefix = "* ";
    suffix = "\n";
    y_absl::StrAppend(&final_comments, "/**\n");
    epilogue = " **/\n";
    add_leading_space = true;
  }

  for (size_t i = 0; i < lines.size(); i++) {
    TProtoStringType line = y_absl::StrReplaceAll(
        y_absl::StripPrefix(lines[i], " "),
        {// HeaderDoc and appledoc use '\' and '@' for markers; escape them.
         {"\\", "\\\\"},
         {"@", "\\@"},
         // Decouple / from * to not have inline comments inside comments.
         {"/*", "/\\*"},
         {"*/", "*\\/"}});
    line = prefix + line;
    y_absl::StripAsciiWhitespace(&line);
    // If not a one line, need to add the first space before *, as
    // y_absl::StripAsciiWhitespace would have removed it.
    line = y_absl::StrCat(add_leading_space ? " " : "", line);
    y_absl::StrAppend(&final_comments, line, suffix);
  }
  return y_absl::StrCat(final_comments, epilogue);
}

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google
