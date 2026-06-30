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

#ifndef GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_TEXT_FORMAT_DECODE_DATA_H__
#define GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_TEXT_FORMAT_DECODE_DATA_H__

#include <string>
#include <utility>
#include <vector>

#include "util/generic/string.h"

// Must be included last
#include "google/protobuf/port_def.inc"

using TProtoStringType = TString;

namespace google {
namespace protobuf {
namespace compiler {
namespace objectivec {

// TODO(b/250947994): PROTOC_EXPORT is only used to allow the CMake build to
// find/link these in the unittest, this is not public api.

// Generate decode data needed for ObjC's GPBDecodeTextFormatName() to transform
// the input into the expected output.
class PROTOC_EXPORT TextFormatDecodeData {
 public:
  TextFormatDecodeData() = default;
  ~TextFormatDecodeData() = default;

  TextFormatDecodeData(const TextFormatDecodeData&) = delete;
  TextFormatDecodeData& operator=(const TextFormatDecodeData&) = delete;

  void AddString(arc_i32 key, const TProtoStringType& input_for_decode,
                 const TProtoStringType& desired_output);
  size_t num_entries() const { return entries_.size(); }
  TProtoStringType Data() const;

  static TProtoStringType DecodeDataForString(const TProtoStringType& input_for_decode,
                                         const TProtoStringType& desired_output);

 private:
  typedef std::pair<arc_i32, TProtoStringType> DataEntry;
  std::vector<DataEntry> entries_;
};

}  // namespace objectivec
}  // namespace compiler
}  // namespace protobuf
}  // namespace google

#include "google/protobuf/port_undef.inc"

#endif  // GOOGLE_PROTOBUF_COMPILER_OBJECTIVEC_TEXT_FORMAT_DECODE_DATA_H__
