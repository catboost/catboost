// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef Y_ABSL_STRINGS_INTERNAL_OSTRINGSTREAM_H_
#define Y_ABSL_STRINGS_INTERNAL_OSTRINGSTREAM_H_

#include <cassert>
#include <ios>
#include <ostream>
#include <streambuf>
#include <util/generic/string.h>
#include <utility>

#include "y_absl/base/config.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace strings_internal {

// The same as std::ostringstream but appends to a user-specified TString,
// and is faster. It is ~70% faster to create, ~50% faster to write to, and
// completely free to extract the result TString.
//
//   TString s;
//   OStringStream strm(&s);
//   strm << 42 << ' ' << 3.14;  // appends to `s`
//
// The stream object doesn't have to be named. Starting from C++11 operator<<
// works with rvalues of std::ostream.
//
//   TString s;
//   OStringStream(&s) << 42 << ' ' << 3.14;  // appends to `s`
//
// OStringStream is faster to create than std::ostringstream but it's still
// relatively slow. Avoid creating multiple streams where a single stream will
// do.
//
// Creates unnecessary instances of OStringStream: slow.
//
//   TString s;
//   OStringStream(&s) << 42;
//   OStringStream(&s) << ' ';
//   OStringStream(&s) << 3.14;
//
// Creates a single instance of OStringStream and reuses it: fast.
//
//   TString s;
//   OStringStream strm(&s);
//   strm << 42;
//   strm << ' ';
//   strm << 3.14;
//
// Note: flush() has no effect. No reason to call it.
class OStringStream final : public std::ostream {
 public:
  // The argument can be null, in which case you'll need to call str(p) with a
  // non-null argument before you can write to the stream.
  //
  // The destructor of OStringStream doesn't use the TString. It's OK to
  // destroy the TString before the stream.
  explicit OStringStream(TString* str)
      : std::ostream(&buf_), buf_(str) {}
  OStringStream(OStringStream&& that)
      : std::ostream(std::move(static_cast<std::ostream&>(that))),
        buf_(that.buf_) {
    rdbuf(&buf_);
  }
  OStringStream& operator=(OStringStream&& that) {
    std::ostream::operator=(std::move(static_cast<std::ostream&>(that)));
    buf_ = that.buf_;
    rdbuf(&buf_);
    return *this;
  }

  TString* str() { return buf_.str(); }
  const TString* str() const { return buf_.str(); }
  void str(TString* str) { buf_.str(str); }

 private:
  class Streambuf final : public std::streambuf {
   public:
    explicit Streambuf(TString* str) : str_(str) {}
    Streambuf(const Streambuf&) = default;
    Streambuf& operator=(const Streambuf&) = default;

    TString* str() { return str_; }
    const TString* str() const { return str_; }
    void str(TString* str) { str_ = str; }

   protected:
    int_type overflow(int c) override;
    std::streamsize xsputn(const char* s, std::streamsize n) override;

   private:
    TString* str_;
  } buf_;
};

}  // namespace strings_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STRINGS_INTERNAL_OSTRINGSTREAM_H_
