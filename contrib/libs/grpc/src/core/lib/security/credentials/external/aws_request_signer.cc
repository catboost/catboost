//
// Copyright 2020 gRPC authors.
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

#include "src/core/lib/security/credentials/external/aws_request_signer.h"

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/sha.h>

#include "y_absl/status/statusor.h"
#include "y_absl/strings/ascii.h"
#include "y_absl/strings/escaping.h"
#include "y_absl/strings/str_cat.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/strings/str_join.h"
#include "y_absl/strings/str_split.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/time/clock.h"
#include "y_absl/time/time.h"

namespace grpc_core {

namespace {

const char kAlgorithm[] = "AWS4-HMAC-SHA256";
const char kDateFormat[] = "%a, %d %b %E4Y %H:%M:%S %Z";
const char kXAmzDateFormat[] = "%Y%m%dT%H%M%SZ";

void SHA256(const TString& str, unsigned char out[SHA256_DIGEST_LENGTH]) {
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, str.c_str(), str.size());
  SHA256_Final(out, &sha256);
}

TString SHA256Hex(const TString& str) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(str, hash);
  TString hash_str(reinterpret_cast<char const*>(hash),
                       SHA256_DIGEST_LENGTH);
  return y_absl::BytesToHexString(hash_str);
}

TString HMAC(const TString& key, const TString& msg) {
  unsigned int len;
  unsigned char digest[EVP_MAX_MD_SIZE];
  HMAC(EVP_sha256(), key.c_str(), key.length(),
       reinterpret_cast<const unsigned char*>(msg.c_str()), msg.length(),
       digest, &len);
  return TString(reinterpret_cast<const char*>(digest), reinterpret_cast<const char*>(digest + len));
}

}  // namespace

AwsRequestSigner::AwsRequestSigner(
    TString access_key_id, TString secret_access_key, TString token,
    TString method, TString url, TString region,
    TString request_payload,
    std::map<TString, TString> additional_headers,
    grpc_error_handle* error)
    : access_key_id_(std::move(access_key_id)),
      secret_access_key_(std::move(secret_access_key)),
      token_(std::move(token)),
      method_(std::move(method)),
      region_(std::move(region)),
      request_payload_(std::move(request_payload)),
      additional_headers_(std::move(additional_headers)) {
  auto amz_date_it = additional_headers_.find("x-amz-date");
  auto date_it = additional_headers_.find("date");
  if (amz_date_it != additional_headers_.end() &&
      date_it != additional_headers_.end()) {
    *error = GRPC_ERROR_CREATE(
        "Only one of {date, x-amz-date} can be specified, not both.");
    return;
  }
  if (amz_date_it != additional_headers_.end()) {
    static_request_date_ = amz_date_it->second;
  } else if (date_it != additional_headers_.end()) {
    y_absl::Time request_date;
    TString err_str;
    if (!y_absl::ParseTime(kDateFormat, date_it->second, &request_date,
                         &err_str)) {
      *error = GRPC_ERROR_CREATE(err_str.c_str());
      return;
    }
    static_request_date_ =
        y_absl::FormatTime(kXAmzDateFormat, request_date, y_absl::UTCTimeZone());
  }
  y_absl::StatusOr<URI> tmp_url = URI::Parse(url);
  if (!tmp_url.ok()) {
    *error = GRPC_ERROR_CREATE("Invalid Aws request url.");
    return;
  }
  url_ = tmp_url.value();
}

std::map<TString, TString> AwsRequestSigner::GetSignedRequestHeaders() {
  TString request_date_full;
  if (!static_request_date_.empty()) {
    if (!request_headers_.empty()) {
      return request_headers_;
    }
    request_date_full = static_request_date_;
  } else {
    y_absl::Time request_date = y_absl::Now();
    request_date_full =
        y_absl::FormatTime(kXAmzDateFormat, request_date, y_absl::UTCTimeZone());
  }
  TString request_date_short = request_date_full.substr(0, 8);
  // TASK 1: Create a canonical request for Signature Version 4
  // https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
  std::vector<y_absl::string_view> canonical_request_vector;
  // 1. HTTPRequestMethod
  canonical_request_vector.emplace_back(method_);
  canonical_request_vector.emplace_back("\n");
  // 2. CanonicalURI
  canonical_request_vector.emplace_back(
      url_.path().empty() ? "/" : y_absl::string_view(url_.path()));
  canonical_request_vector.emplace_back("\n");
  // 3. CanonicalQueryString
  std::vector<TString> query_vector;
  for (const URI::QueryParam& query_kv : url_.query_parameter_pairs()) {
    query_vector.emplace_back(y_absl::StrCat(query_kv.key, "=", query_kv.value));
  }
  TString query = y_absl::StrJoin(query_vector, "&");
  canonical_request_vector.emplace_back(query);
  canonical_request_vector.emplace_back("\n");
  // 4. CanonicalHeaders
  if (request_headers_.empty()) {
    request_headers_.insert({"host", url_.authority()});
    if (!token_.empty()) {
      request_headers_.insert({"x-amz-security-token", token_});
    }
    for (const auto& header : additional_headers_) {
      request_headers_.insert(
          {y_absl::AsciiStrToLower(header.first), header.second});
    }
  }
  if (additional_headers_.find("date") == additional_headers_.end()) {
    request_headers_["x-amz-date"] = request_date_full;
  }
  std::vector<y_absl::string_view> canonical_headers_vector;
  for (const auto& header : request_headers_) {
    canonical_headers_vector.emplace_back(header.first);
    canonical_headers_vector.emplace_back(":");
    canonical_headers_vector.emplace_back(header.second);
    canonical_headers_vector.emplace_back("\n");
  }
  TString canonical_headers = y_absl::StrJoin(canonical_headers_vector, "");
  canonical_request_vector.emplace_back(canonical_headers);
  canonical_request_vector.emplace_back("\n");
  // 5. SignedHeaders
  std::vector<y_absl::string_view> signed_headers_vector;
  signed_headers_vector.reserve(request_headers_.size());
  for (const auto& header : request_headers_) {
    signed_headers_vector.emplace_back(header.first);
  }
  TString signed_headers = y_absl::StrJoin(signed_headers_vector, ";");
  canonical_request_vector.emplace_back(signed_headers);
  canonical_request_vector.emplace_back("\n");
  // 6. RequestPayload
  TString hashed_request_payload = SHA256Hex(request_payload_);
  canonical_request_vector.emplace_back(hashed_request_payload);
  TString canonical_request = y_absl::StrJoin(canonical_request_vector, "");
  // TASK 2: Create a string to sign for Signature Version 4
  // https://docs.aws.amazon.com/general/latest/gr/sigv4-create-string-to-sign.html
  std::vector<y_absl::string_view> string_to_sign_vector;
  // 1. Algorithm
  string_to_sign_vector.emplace_back("AWS4-HMAC-SHA256");
  string_to_sign_vector.emplace_back("\n");
  // 2. RequestDateTime
  string_to_sign_vector.emplace_back(request_date_full);
  string_to_sign_vector.emplace_back("\n");
  // 3. CredentialScope
  std::pair<y_absl::string_view, y_absl::string_view> host_parts =
      y_absl::StrSplit(url_.authority(), y_absl::MaxSplits('.', 1));
  TString service_name(host_parts.first);
  TString credential_scope = y_absl::StrFormat(
      "%s/%s/%s/aws4_request", request_date_short, region_, service_name);
  string_to_sign_vector.emplace_back(credential_scope);
  string_to_sign_vector.emplace_back("\n");
  // 4. HashedCanonicalRequest
  TString hashed_canonical_request = SHA256Hex(canonical_request);
  string_to_sign_vector.emplace_back(hashed_canonical_request);
  TString string_to_sign = y_absl::StrJoin(string_to_sign_vector, "");
  // TASK 3: Task 3: Calculate the signature for AWS Signature Version 4
  // https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
  // 1. Derive your signing key.
  TString date = HMAC("AWS4" + secret_access_key_, request_date_short);
  TString region = HMAC(date, region_);
  TString service = HMAC(region, service_name);
  TString signing = HMAC(service, "aws4_request");
  // 2. Calculate the signature.
  TString signature_str = HMAC(signing, string_to_sign);
  TString signature = y_absl::BytesToHexString(signature_str);
  // TASK 4: Add the signature to the HTTP request
  // https://docs.aws.amazon.com/general/latest/gr/sigv4-add-signature-to-request.html
  TString authorization_header = y_absl::StrFormat(
      "%s Credential=%s/%s, SignedHeaders=%s, Signature=%s", kAlgorithm,
      access_key_id_, credential_scope, signed_headers, signature);
  request_headers_["Authorization"] = authorization_header;
  return request_headers_;
}

}  // namespace grpc_core
