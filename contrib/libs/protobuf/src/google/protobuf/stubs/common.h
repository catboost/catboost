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

// Author: kenton@google.com (Kenton Varda) and others
//
// Contains basic types and utilities used by the rest of the library.

#ifndef GOOGLE_PROTOBUF_COMMON_H__
#define GOOGLE_PROTOBUF_COMMON_H__

// Yandex-specific
#include <util/generic/string.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
using TProtoStringType = TString;
// End of Yandex-specific

#include "stubs/port.h"
#include "stubs/macros.h"
#include "stubs/platform_macros.h"

// TODO(liujisi): Remove the following includes after the include clean-up.
#include <contrib/libs/protobuf/stubs/logging.h>
#include "stubs/scoped_ptr.h"
#include <contrib/libs/protobuf/stubs/mutex.h>
#include "stubs/callback.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>  // for TARGET_OS_IPHONE
#endif

#if defined(__ANDROID__) || defined(GOOGLE_PROTOBUF_OS_ANDROID) || (defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE) || defined(GOOGLE_PROTOBUF_OS_IPHONE)
#include <pthread.h>
#endif

#if defined(_WIN32) && defined(GetMessage)
// Allow GetMessage to be used as a valid method name in protobuf classes.
// windows.h defines GetMessage() as a macro.  Let's re-define it as an inline
// function.  The inline function should be equivalent for C++ users.
inline BOOL GetMessage_Win32(
    LPMSG lpMsg, HWND hWnd,
    UINT wMsgFilterMin, UINT wMsgFilterMax) {
  return GetMessage(lpMsg, hWnd, wMsgFilterMin, wMsgFilterMax);
}
#undef GetMessage
inline BOOL GetMessage(
    LPMSG lpMsg, HWND hWnd,
    UINT wMsgFilterMin, UINT wMsgFilterMax) {
  return GetMessage_Win32(lpMsg, hWnd, wMsgFilterMin, wMsgFilterMax);
}
#endif

namespace std {}

namespace google {
namespace protobuf {

using string = TProtoStringType;

namespace internal {

// Some of these constants are macros rather than const ints so that they can
// be used in #if directives.

// The current version, represented as a single integer to make comparison
// easier:  major * 10^6 + minor * 10^3 + micro
#define GOOGLE_PROTOBUF_VERSION 3004000

// A suffix string for alpha, beta or rc releases. Empty for stable releases.
#define GOOGLE_PROTOBUF_VERSION_SUFFIX ""

// The minimum library version which works with the current version of the
// headers.
#define GOOGLE_PROTOBUF_MIN_LIBRARY_VERSION 3004000

// The minimum header version which works with the current version of
// the library.  This constant should only be used by protoc's C++ code
// generator.
static const int kMinHeaderVersionForLibrary = 3004000;

// The minimum protoc version which works with the current version of the
// headers.
#define GOOGLE_PROTOBUF_MIN_PROTOC_VERSION 3004000

// The minimum header version which works with the current version of
// protoc.  This constant should only be used in VerifyVersion().
static const int kMinHeaderVersionForProtoc = 3004000;

// Verifies that the headers and libraries are compatible.  Use the macro
// below to call this.
void LIBPROTOBUF_EXPORT VerifyVersion(int headerVersion, int minLibraryVersion,
                                      const char* filename);

// Converts a numeric version number to a string.
TProtoStringType LIBPROTOBUF_EXPORT VersionString(int version);

}  // namespace internal

// Place this macro in your main() function (or somewhere before you attempt
// to use the protobuf library) to verify that the version you link against
// matches the headers you compiled against.  If a version mismatch is
// detected, the process will abort.
#define GOOGLE_PROTOBUF_VERIFY_VERSION                                    \
  ::google::protobuf::internal::VerifyVersion(                            \
    GOOGLE_PROTOBUF_VERSION, GOOGLE_PROTOBUF_MIN_LIBRARY_VERSION,         \
    __FILE__)


// ===================================================================
// from google3/util/utf8/public/unilib.h

class StringPiece;
namespace internal {

// Checks if the buffer contains structurally-valid UTF-8.  Implemented in
// structurally_valid.cc.
LIBPROTOBUF_EXPORT bool IsStructurallyValidUTF8(const char* buf, int len);

inline bool IsStructurallyValidUTF8(const TProtoStringType& str) {
  return IsStructurallyValidUTF8(str.data(), static_cast<int>(str.length()));
}

// Returns initial number of bytes of structually valid UTF-8.
LIBPROTOBUF_EXPORT int UTF8SpnStructurallyValid(const StringPiece& str);

// Coerce UTF-8 byte string in src_str to be
// a structurally-valid equal-length string by selectively
// overwriting illegal bytes with replace_char (typically ' ' or '?').
// replace_char must be legal printable 7-bit Ascii 0x20..0x7e.
// src_str is read-only.
//
// Returns pointer to output buffer, src_str.data() if no changes were made,
//  or idst if some bytes were changed. idst is allocated by the caller
//  and must be at least as big as src_str
//
// Optimized for: all structurally valid and no byte copying is done.
//
LIBPROTOBUF_EXPORT char* UTF8CoerceToStructurallyValid(
    const StringPiece& str, char* dst, char replace_char);

}  // namespace internal


// ===================================================================
// Shutdown support.

// Shut down the entire protocol buffers library, deleting all static-duration
// objects allocated by the library or by generated .pb.cc files.
//
// There are two reasons you might want to call this:
// * You use a draconian definition of "memory leak" in which you expect
//   every single malloc() to have a corresponding free(), even for objects
//   which live until program exit.
// * You are writing a dynamically-loaded library which needs to clean up
//   after itself when the library is unloaded.
//
// It is safe to call this multiple times.  However, it is not safe to use
// any other part of the protocol buffers library after
// ShutdownProtobufLibrary() has been called.
LIBPROTOBUF_EXPORT void ShutdownProtobufLibrary();

namespace internal {

// Register a function to be called when ShutdownProtocolBuffers() is called.
LIBPROTOBUF_EXPORT void OnShutdown(void (*func)());
// Destroy the string (call string destructor)
LIBPROTOBUF_EXPORT void OnShutdownDestroyString(const TProtoStringType* ptr);
// Destroy (not delete) the message
LIBPROTOBUF_EXPORT void OnShutdownDestroyMessage(const void* ptr);

}  // namespace internal
}  // namespace protobuf
}  // namespace google

namespace NProtoBuf {
  using namespace google;
  using namespace google::protobuf;
};

#endif  // GOOGLE_PROTOBUF_COMMON_H__
