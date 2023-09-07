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

#ifndef GRPC_SRC_CORE_LIB_URI_URI_PARSER_H
#define GRPC_SRC_CORE_LIB_URI_URI_PARSER_H

#include <grpc/support/port_platform.h>

#include <map>
#include <util/generic/string.h>
#include <util/string/cast.h>
#include <vector>

#include "y_absl/status/statusor.h"
#include "y_absl/strings/string_view.h"

namespace grpc_core {

class URI {
 public:
  struct QueryParam {
    TString key;
    TString value;
    bool operator==(const QueryParam& other) const {
      return key == other.key && value == other.value;
    }
    bool operator<(const QueryParam& other) const {
      int c = key.compare(other.key);
      if (c != 0) return c < 0;
      return value < other.value;
    }
  };

  // Creates a URI by parsing an rfc3986 URI string. Returns an
  // InvalidArgumentError on failure.
  static y_absl::StatusOr<URI> Parse(y_absl::string_view uri_text);
  // Creates a URI from components. Returns an InvalidArgumentError on failure.
  static y_absl::StatusOr<URI> Create(
      TString scheme, TString authority, TString path,
      std::vector<QueryParam> query_parameter_pairs, TString fragment);

  URI() = default;

  // Copy construction and assignment
  URI(const URI& other);
  URI& operator=(const URI& other);
  // Move construction and assignment
  URI(URI&&) = default;
  URI& operator=(URI&&) = default;

  static TString PercentEncodeAuthority(y_absl::string_view str);
  static TString PercentEncodePath(y_absl::string_view str);

  static TString PercentDecode(y_absl::string_view str);

  const TString& scheme() const { return scheme_; }
  const TString& authority() const { return authority_; }
  const TString& path() const { return path_; }
  // Stores the *last* value appearing for each repeated key in the query
  // string. If you need to capture repeated query parameters, use
  // `query_parameter_pairs`.
  const std::map<y_absl::string_view, y_absl::string_view>& query_parameter_map()
      const {
    return query_parameter_map_;
  }
  // A vector of key:value query parameter pairs, kept in order of appearance
  // within the URI search string. Repeated keys are represented as separate
  // key:value elements.
  const std::vector<QueryParam>& query_parameter_pairs() const {
    return query_parameter_pairs_;
  }
  const TString& fragment() const { return fragment_; }

  TString ToString() const;

 private:
  URI(TString scheme, TString authority, TString path,
      std::vector<QueryParam> query_parameter_pairs, TString fragment);

  TString scheme_;
  TString authority_;
  TString path_;
  std::map<y_absl::string_view, y_absl::string_view> query_parameter_map_;
  std::vector<QueryParam> query_parameter_pairs_;
  TString fragment_;
};
}  // namespace grpc_core

#endif  // GRPC_SRC_CORE_LIB_URI_URI_PARSER_H
