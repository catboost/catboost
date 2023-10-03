//
// Copyright 2015 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <grpc/support/port_platform.h>

#include "src/core/lib/uri/uri_parser.h"

#include <ctype.h>
#include <stddef.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <map>
#include <util/generic/string.h>
#include <util/string/cast.h>
#include <utility>

#include "y_absl/status/status.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/match.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/strings/str_join.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/strip.h"

#include <grpc/support/log.h>

namespace grpc_core {

namespace {

// Returns true for any sub-delim character, as defined in:
// https://datatracker.ietf.org/doc/html/rfc3986#section-2.2
bool IsSubDelimChar(char c) {
  switch (c) {
    case '!':
    case '$':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case ';':
    case '=':
      return true;
  }
  return false;
}

// Returns true for any unreserved character, as defined in:
// https://datatracker.ietf.org/doc/html/rfc3986#section-2.3
bool IsUnreservedChar(char c) {
  if (y_absl::ascii_isalnum(c)) return true;
  switch (c) {
    case '-':
    case '.':
    case '_':
    case '~':
      return true;
  }
  return false;
}

// Returns true for any character in scheme, as defined in:
// https://datatracker.ietf.org/doc/html/rfc3986#section-3.1
bool IsSchemeChar(char c) {
  if (y_absl::ascii_isalnum(c)) return true;
  switch (c) {
    case '+':
    case '-':
    case '.':
      return true;
  }
  return false;
}

// Returns true for any character in authority, as defined in:
// https://datatracker.ietf.org/doc/html/rfc3986#section-3.2
bool IsAuthorityChar(char c) {
  if (IsUnreservedChar(c)) return true;
  if (IsSubDelimChar(c)) return true;
  switch (c) {
    case ':':
    case '[':
    case ']':
    case '@':
      return true;
  }
  return false;
}

// Returns true for any character in pchar, as defined in:
// https://datatracker.ietf.org/doc/html/rfc3986#section-3.3
bool IsPChar(char c) {
  if (IsUnreservedChar(c)) return true;
  if (IsSubDelimChar(c)) return true;
  switch (c) {
    case ':':
    case '@':
      return true;
  }
  return false;
}

// Returns true for any character allowed in a URI path, as defined in:
// https://datatracker.ietf.org/doc/html/rfc3986#section-3.3
bool IsPathChar(char c) { return IsPChar(c) || c == '/'; }

// Returns true for any character allowed in a URI query or fragment,
// as defined in:
// See https://tools.ietf.org/html/rfc3986#section-3.4
bool IsQueryOrFragmentChar(char c) {
  return IsPChar(c) || c == '/' || c == '?';
}

// Same as IsQueryOrFragmentChar(), but excludes '&' and '='.
bool IsQueryKeyOrValueChar(char c) {
  return c != '&' && c != '=' && IsQueryOrFragmentChar(c);
}

// Returns a copy of str, percent-encoding any character for which
// is_allowed_char() returns false.
TString PercentEncode(y_absl::string_view str,
                          std::function<bool(char)> is_allowed_char) {
  TString out;
  for (char c : str) {
    if (!is_allowed_char(c)) {
      TString hex = y_absl::BytesToHexString(y_absl::string_view(&c, 1));
      GPR_ASSERT(hex.size() == 2);
      // BytesToHexString() returns lower case, but
      // https://datatracker.ietf.org/doc/html/rfc3986#section-6.2.2.1 says
      // to prefer upper-case.
      y_absl::AsciiStrToUpper(&hex);
      out.push_back('%');
      out.append(hex);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// Checks if this string is made up of query/fragment chars and '%' exclusively.
// See https://tools.ietf.org/html/rfc3986#section-3.4
bool IsQueryOrFragmentString(y_absl::string_view str) {
  for (char c : str) {
    if (!IsQueryOrFragmentChar(c) && c != '%') return false;
  }
  return true;
}

y_absl::Status MakeInvalidURIStatus(y_absl::string_view part_name,
                                  y_absl::string_view uri,
                                  y_absl::string_view extra) {
  return y_absl::InvalidArgumentError(y_absl::StrFormat(
      "Could not parse '%s' from uri '%s'. %s", part_name, uri, extra));
}

}  // namespace

TString URI::PercentEncodeAuthority(y_absl::string_view str) {
  return PercentEncode(str, IsAuthorityChar);
}

TString URI::PercentEncodePath(y_absl::string_view str) {
  return PercentEncode(str, IsPathChar);
}

// Similar to `grpc_permissive_percent_decode_slice`, this %-decodes all valid
// triplets, and passes through the rest verbatim.
TString URI::PercentDecode(y_absl::string_view str) {
  if (str.empty() || !y_absl::StrContains(str, "%")) {
    return TString(str);
  }
  TString out;
  TString unescaped;
  out.reserve(str.size());
  for (size_t i = 0; i < str.length(); i++) {
    unescaped = "";
    if (str[i] == '%' && i + 3 <= str.length() &&
        y_absl::CUnescape(y_absl::StrCat("\\x", str.substr(i + 1, 2)),
                        &unescaped) &&
        unescaped.length() == 1) {
      out += unescaped[0];
      i += 2;
    } else {
      out += str[i];
    }
  }
  return out;
}

y_absl::StatusOr<URI> URI::Parse(y_absl::string_view uri_text) {
  y_absl::string_view remaining = uri_text;
  // parse scheme
  size_t offset = remaining.find(':');
  if (offset == remaining.npos || offset == 0) {
    return MakeInvalidURIStatus("scheme", uri_text, "Scheme not found.");
  }
  TString scheme(remaining.substr(0, offset));
  if (scheme.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               "abcdefghijklmnopqrstuvwxyz"
                               "0123456789+-.") != TString::npos) {
    return MakeInvalidURIStatus("scheme", uri_text,
                                "Scheme contains invalid characters.");
  }
  if (!isalpha(scheme[0])) {
    return MakeInvalidURIStatus(
        "scheme", uri_text,
        "Scheme must begin with an alpha character [A-Za-z].");
  }
  remaining.remove_prefix(offset + 1);
  // parse authority
  TString authority;
  if (y_absl::ConsumePrefix(&remaining, "//")) {
    offset = remaining.find_first_of("/?#");
    authority = PercentDecode(remaining.substr(0, offset));
    if (offset == remaining.npos) {
      remaining = "";
    } else {
      remaining.remove_prefix(offset);
    }
  }
  // parse path
  TString path;
  if (!remaining.empty()) {
    offset = remaining.find_first_of("?#");
    path = PercentDecode(remaining.substr(0, offset));
    if (offset == remaining.npos) {
      remaining = "";
    } else {
      remaining.remove_prefix(offset);
    }
  }
  // parse query
  std::vector<QueryParam> query_param_pairs;
  if (y_absl::ConsumePrefix(&remaining, "?")) {
    offset = remaining.find('#');
    y_absl::string_view tmp_query = remaining.substr(0, offset);
    if (tmp_query.empty()) {
      return MakeInvalidURIStatus("query", uri_text, "Invalid query string.");
    }
    if (!IsQueryOrFragmentString(tmp_query)) {
      return MakeInvalidURIStatus("query string", uri_text,
                                  "Query string contains invalid characters.");
    }
    for (y_absl::string_view query_param : y_absl::StrSplit(tmp_query, '&')) {
      const std::pair<y_absl::string_view, y_absl::string_view> possible_kv =
          y_absl::StrSplit(query_param, y_absl::MaxSplits('=', 1));
      if (possible_kv.first.empty()) continue;
      query_param_pairs.push_back({PercentDecode(possible_kv.first),
                                   PercentDecode(possible_kv.second)});
    }
    if (offset == remaining.npos) {
      remaining = "";
    } else {
      remaining.remove_prefix(offset);
    }
  }
  TString fragment;
  if (y_absl::ConsumePrefix(&remaining, "#")) {
    if (!IsQueryOrFragmentString(remaining)) {
      return MakeInvalidURIStatus("fragment", uri_text,
                                  "Fragment contains invalid characters.");
    }
    fragment = PercentDecode(remaining);
  }
  return URI(std::move(scheme), std::move(authority), std::move(path),
             std::move(query_param_pairs), std::move(fragment));
}

y_absl::StatusOr<URI> URI::Create(TString scheme, TString authority,
                                TString path,
                                std::vector<QueryParam> query_parameter_pairs,
                                TString fragment) {
  if (!authority.empty() && !path.empty() && path[0] != '/') {
    return y_absl::InvalidArgumentError(
        "if authority is present, path must start with a '/'");
  }
  return URI(std::move(scheme), std::move(authority), std::move(path),
             std::move(query_parameter_pairs), std::move(fragment));
}

URI::URI(TString scheme, TString authority, TString path,
         std::vector<QueryParam> query_parameter_pairs, TString fragment)
    : scheme_(std::move(scheme)),
      authority_(std::move(authority)),
      path_(std::move(path)),
      query_parameter_pairs_(std::move(query_parameter_pairs)),
      fragment_(std::move(fragment)) {
  for (const auto& kv : query_parameter_pairs_) {
    query_parameter_map_[kv.key] = kv.value;
  }
}

URI::URI(const URI& other)
    : scheme_(other.scheme_),
      authority_(other.authority_),
      path_(other.path_),
      query_parameter_pairs_(other.query_parameter_pairs_),
      fragment_(other.fragment_) {
  for (const auto& kv : query_parameter_pairs_) {
    query_parameter_map_[kv.key] = kv.value;
  }
}

URI& URI::operator=(const URI& other) {
  if (this == &other) {
    return *this;
  }
  scheme_ = other.scheme_;
  authority_ = other.authority_;
  path_ = other.path_;
  query_parameter_pairs_ = other.query_parameter_pairs_;
  fragment_ = other.fragment_;
  for (const auto& kv : query_parameter_pairs_) {
    query_parameter_map_[kv.key] = kv.value;
  }
  return *this;
}

namespace {

// A pair formatter for use with y_absl::StrJoin() for formatting query params.
struct QueryParameterFormatter {
  void operator()(TString* out, const URI::QueryParam& query_param) const {
    out->append(
        y_absl::StrCat(PercentEncode(query_param.key, IsQueryKeyOrValueChar), "=",
                     PercentEncode(query_param.value, IsQueryKeyOrValueChar)));
  }
};

}  // namespace

TString URI::ToString() const {
  std::vector<TString> parts = {PercentEncode(scheme_, IsSchemeChar), ":"};
  if (!authority_.empty()) {
    parts.emplace_back("//");
    parts.emplace_back(PercentEncode(authority_, IsAuthorityChar));
  }
  if (!path_.empty()) {
    parts.emplace_back(PercentEncode(path_, IsPathChar));
  }
  if (!query_parameter_pairs_.empty()) {
    parts.push_back("?");
    parts.push_back(
        y_absl::StrJoin(query_parameter_pairs_, "&", QueryParameterFormatter()));
  }
  if (!fragment_.empty()) {
    parts.push_back("#");
    parts.push_back(PercentEncode(fragment_, IsQueryOrFragmentChar));
  }
  return y_absl::StrJoin(parts, "");
}

}  // namespace grpc_core
