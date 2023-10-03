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

#ifndef GRPC_SRC_CORE_LIB_SECURITY_CREDENTIALS_EXTERNAL_AWS_REQUEST_SIGNER_H
#define GRPC_SRC_CORE_LIB_SECURITY_CREDENTIALS_EXTERNAL_AWS_REQUEST_SIGNER_H

#include <grpc/support/port_platform.h>

#include <map>
#include <util/generic/string.h>
#include <util/string/cast.h>

#include "src/core/lib/iomgr/error.h"
#include "src/core/lib/uri/uri_parser.h"

namespace grpc_core {

// Implements an AWS API request signer based on the AWS Signature Version 4
// signing process.
// https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
// To retrieve the subject token in AwsExternalAccountCredentials, we need to
// sign an AWS request server and use the signed request as the subject token.
// This class is a utility to sign an AWS request.
class AwsRequestSigner {
 public:
  // Construct a signer with the necessary information to sign a request.
  // `access_key_id`, `secret_access_key` and `token` are the AWS credentials
  // required for signing. `method` and `url` are the HTTP method and url of the
  // request. `region` is the region of the AWS environment. `request_payload`
  // is the payload of the HTTP request. `additional_headers` are additional
  // headers to be inject into the request.
  AwsRequestSigner(TString access_key_id, TString secret_access_key,
                   TString token, TString method, TString url,
                   TString region, TString request_payload,
                   std::map<TString, TString> additional_headers,
                   grpc_error_handle* error);

  // This method triggers the signing process then returns the headers of the
  // signed request as a map. In case there is an error, the input `error`
  // parameter will be updated and an empty map will be returned if there is
  // error.
  std::map<TString, TString> GetSignedRequestHeaders();

 private:
  TString access_key_id_;
  TString secret_access_key_;
  TString token_;
  TString method_;
  URI url_;
  TString region_;
  TString request_payload_;
  std::map<TString, TString> additional_headers_;

  TString static_request_date_;
  std::map<TString, TString> request_headers_;
};

}  // namespace grpc_core

#endif  // GRPC_SRC_CORE_LIB_SECURITY_CREDENTIALS_EXTERNAL_AWS_REQUEST_SIGNER_H
