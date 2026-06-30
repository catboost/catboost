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

// Internal JSON tokenization utilities; not public API.
#ifndef GOOGLE_PROTOBUF_JSON_INTERNAL_LEXER_H__
#define GOOGLE_PROTOBUF_JSON_INTERNAL_LEXER_H__

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

#include "google/protobuf/descriptor.h"
#include "y_absl/status/status.h"
#include "y_absl/status/statusor.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/strings/string_view.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/json/internal/message_path.h"
#include "google/protobuf/json/internal/zero_copy_buffered_stream.h"
#include "google/protobuf/stubs/status_macros.h"


// Must be included last.
#include "google/protobuf/port_def.inc"

namespace google {
namespace protobuf {
namespace json_internal {
// This is a duplicate of JsonParseOptions from json_util.h; it must be
// re-defined here so that :json_lexer does not need to depend on :json_util.
struct ParseOptions {
  bool ignore_unknown_fields = false;
  bool case_insensitive_enum_parsing = false;

  static constexpr size_t kDefaultDepth = 100;

  // The number of times we may recurse before bailing out on the grounds of
  // avoiding pathological input.
  int recursion_depth = kDefaultDepth;

  // The original parser used by json_util2 accepted a number of non-standard
  // options. Setting this flag enables them.
  //
  // What those extensions were is explicitly not documented, beyond what exists
  // in the unit tests; we intend to remove this setting eventually. See
  // b/234868512.
  bool allow_legacy_syntax = false;
};

// A position in JSON input, for error context.
struct JsonLocation {
  // This type exists to work around an absl type that has not yet been
  // released.
  struct SourceLocation {
    static SourceLocation current() { return {}; }
  };

  // Line and column are both zero-indexed in-memory.
  size_t offset = 0;
  size_t line = 0;
  size_t col = 0;
  const MessagePath* path = nullptr;

  // Creates an y_absl::InvalidArgumentError with line/column information.
  y_absl::Status Invalid(y_absl::string_view message,
                       SourceLocation sl = SourceLocation::current()) const;
};

template <typename T>
struct LocationWith {
  T value;
  JsonLocation loc;
};

class JsonLexer {
 public:
  // A kind of token that PeekKind() can detect.
  enum Kind {
    kObj,
    kArr,
    kStr,
    kNum,
    kTrue,
    kFalse,
    kNull,
  };

  using SourceLocation = JsonLocation::SourceLocation;

  JsonLexer(io::ZeroCopyInputStream* stream, const ParseOptions& options,
            MessagePath* path = nullptr, JsonLocation start = {})
      : stream_(stream), options_(options), json_loc_(start), path_(path) {
    json_loc_.path = path_;
  }

  const ParseOptions& options() const { return options_; }

  const MessagePath& path() const { return *path_; }
  MessagePath& path() { return *path_; }

  // Creates an y_absl::InvalidArgumentError with line/column information.
  y_absl::Status Invalid(y_absl::string_view message,
                       SourceLocation sl = SourceLocation::current()) {
    return json_loc_.Invalid(message, sl);
  }

  // Expects the next bytes to be parsed (after consuming whitespace) to be
  // exactly `literal`. If they are, consumes them; otherwise returns an error.
  y_absl::Status Expect(y_absl::string_view literal,
                      SourceLocation sl = SourceLocation::current()) {
    RETURN_IF_ERROR(SkipToToken());
    auto buffering = stream_.BufferAtLeast(literal.size());
    RETURN_IF_ERROR(buffering.status());

    if (!y_absl::StartsWith(stream_.Unread(), literal)) {
      return Invalid(
          y_absl::StrFormat("unexpected character: '%c'; expected '%s'",
                          stream_.PeekChar(), literal),
          sl);
    }

    return Advance(literal.size());
  }

  // Like Expect(), but returns a boolean. This makes it clear that the
  // lookahead is failible.
  bool Peek(y_absl::string_view literal) {
    // Suppress the error; this can only fail on EOF in which case we would
    // return false regardless.
    (void)SkipToToken();
    auto ignored = stream_.BufferAtLeast(literal.size());
    if (!y_absl::StartsWith(stream_.Unread(), literal)) {
      return false;
    }

    // We just ensured we had enough buffered so we can suppress this error.
    (void)Advance(literal.size());
    return true;
  }

  // Like Peek(string), but returns true if and only if a token of the given
  // kind can be lexed next. Returns false on EOF, just like Peek(string).
  bool Peek(Kind needle) {
    auto kind = PeekKind();
    return kind.ok() && *kind == needle;
  }

  // Consumes all whitespace and other ignored characters until the next
  // token.
  //
  // This function returns an error on EOF, so PeekChar() can be safely
  // called if it returns ok.
  y_absl::Status SkipToToken();

  // Returns which kind of value token (i.e., something that can occur after
  // a `:`) is next up to be parsed.
  y_absl::StatusOr<Kind> PeekKind();

  // Parses a JSON number.
  y_absl::StatusOr<LocationWith<double>> ParseNumber();

  // Parses a number as a string, without turning it into an integer.
  y_absl::StatusOr<LocationWith<MaybeOwnedString>> ParseRawNumber();

  // Parses a UTF-8 string. If the contents of the string happen to actually be
  // UTF-8, it will return a zero-copy view; otherwise it will allocate.
  y_absl::StatusOr<LocationWith<MaybeOwnedString>> ParseUtf8();

  // Walks over an array, calling `f` each time an element is reached.
  //
  // `f` should have type `() -> y_absl::Status`.
  template <typename F>
  y_absl::Status VisitArray(F f);

  // Walks over an object, calling `f` just after parsing each `:`.
  //
  // `f` should have type `(y_absl::string_view) -> y_absl::Status`.
  template <typename F>
  y_absl::Status VisitObject(F f);

  // Parses a single value and discards it.
  y_absl::Status SkipValue();

  // Forwards of functions from ZeroCopyBufferedStream.

  bool AtEof() {
    // Ignore whitespace for the purposes of finding the EOF. This will return
    // an error if we hit EOF, so we discard it.
    (void)SkipToToken();
    return stream_.AtEof();
  }

  y_absl::StatusOr<LocationWith<MaybeOwnedString>> Take(size_t len) {
    JsonLocation loc = json_loc_;
    auto taken = stream_.Take(len);
    RETURN_IF_ERROR(taken.status());
    return LocationWith<MaybeOwnedString>{*std::move(taken), loc};
  }

  template <typename Pred>
  y_absl::StatusOr<LocationWith<MaybeOwnedString>> TakeWhile(Pred p) {
    JsonLocation loc = json_loc_;
    auto taken = stream_.TakeWhile(std::move(p));
    RETURN_IF_ERROR(taken.status());
    return LocationWith<MaybeOwnedString>{*std::move(taken), loc};
  }

  LocationWith<Mark> BeginMark() { return {stream_.BeginMark(), json_loc_}; }

 private:
  friend BufferingGuard;
  friend Mark;
  friend MaybeOwnedString;

  y_absl::Status Push() {
    if (options_.recursion_depth == 0) {
      return Invalid("JSON content was too deeply nested");
    }
    --options_.recursion_depth;
    return y_absl::OkStatus();
  }

  void Pop() { ++options_.recursion_depth; }

  // Parses the next four bytes as a 16-bit hex numeral.
  y_absl::StatusOr<uint16_t> ParseU16HexCodepoint();

  // Parses a Unicode escape (\uXXXX); this may be a surrogate pair, so it may
  // consume the character that follows. Both are encoded as utf8 into
  // `out_utf8`; returns the number of bytes written.
  y_absl::StatusOr<size_t> ParseUnicodeEscape(char out_utf8[4]);

  // Parses an alphanumeric "identifier", for use with the non-standard
  // "unquoted keys" extension.
  y_absl::StatusOr<LocationWith<MaybeOwnedString>> ParseBareWord();

  y_absl::Status Advance(size_t bytes) {
    RETURN_IF_ERROR(stream_.Advance(bytes));
    json_loc_.offset += static_cast<int>(bytes);
    json_loc_.col += static_cast<int>(bytes);
    return y_absl::OkStatus();
  }

  ZeroCopyBufferedStream stream_;

  ParseOptions options_;
  JsonLocation json_loc_;
  MessagePath* path_;
};

template <typename F>
y_absl::Status JsonLexer::VisitArray(F f) {
  RETURN_IF_ERROR(Expect("["));
  RETURN_IF_ERROR(Push());

  if (Peek("]")) {
    Pop();
    return y_absl::OkStatus();
  }

  bool has_comma = true;
  do {
    if (!has_comma) {
      return Invalid("expected ','");
    }
    RETURN_IF_ERROR(f());
    has_comma = Peek(",");
  } while (!Peek("]"));

  if (!options_.allow_legacy_syntax && has_comma) {
    return Invalid("expected ']'");
  }

  Pop();
  return y_absl::OkStatus();
}

// Walks over an object, calling `f` just after parsing each `:`.
//
// `f` should have type `(MaybeOwnedString&) -> y_absl::Status`.
template <typename F>
y_absl::Status JsonLexer::VisitObject(F f) {
  RETURN_IF_ERROR(Expect("{"));
  RETURN_IF_ERROR(Push());

  if (Peek("}")) {
    Pop();
    return y_absl::OkStatus();
  }

  bool has_comma = true;
  do {
    if (!has_comma) {
      return Invalid("expected ','");
    }
    RETURN_IF_ERROR(SkipToToken());

    y_absl::StatusOr<LocationWith<MaybeOwnedString>> key;
    if (stream_.PeekChar() == '"' || stream_.PeekChar() == '\'') {
      key = ParseUtf8();
    } else if (options_.allow_legacy_syntax) {
      key = ParseBareWord();
    } else {
      return Invalid("expected '\"'");
    }

    RETURN_IF_ERROR(key.status());
    RETURN_IF_ERROR(Expect(":"));
    RETURN_IF_ERROR(f(*key));
    has_comma = Peek(",");
  } while (!Peek("}"));
  Pop();

  if (!options_.allow_legacy_syntax && has_comma) {
    return Invalid("expected '}'");
  }

  return y_absl::OkStatus();
}
}  // namespace json_internal
}  // namespace protobuf
}  // namespace google

#include "google/protobuf/port_undef.inc"
#endif  // GOOGLE_PROTOBUF_JSON_INTERNAL_LEXER_H__
