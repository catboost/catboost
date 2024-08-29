// Copyright 2019 The Abseil Authors.
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
//
// -----------------------------------------------------------------------------
// File: status.h
// -----------------------------------------------------------------------------
//
// This header file defines the Abseil `status` library, consisting of:
//
//   * An `y_absl::Status` class for holding error handling information
//   * A set of canonical `y_absl::StatusCode` error codes, and associated
//     utilities for generating and propagating status codes.
//   * A set of helper functions for creating status codes and checking their
//     values
//
// Within Google, `y_absl::Status` is the primary mechanism for communicating
// errors in C++, and is used to represent error state in both in-process
// library calls as well as RPC calls. Some of these errors may be recoverable,
// but others may not. Most functions that can produce a recoverable error
// should be designed to return an `y_absl::Status` (or `y_absl::StatusOr`).
//
// Example:
//
// y_absl::Status myFunction(y_absl::string_view fname, ...) {
//   ...
//   // encounter error
//   if (error condition) {
//     return y_absl::InvalidArgumentError("bad mode");
//   }
//   // else, return OK
//   return y_absl::OkStatus();
// }
//
// An `y_absl::Status` is designed to either return "OK" or one of a number of
// different error codes, corresponding to typical error conditions.
// In almost all cases, when using `y_absl::Status` you should use the canonical
// error codes (of type `y_absl::StatusCode`) enumerated in this header file.
// These canonical codes are understood across the codebase and will be
// accepted across all API and RPC boundaries.
#ifndef Y_ABSL_STATUS_STATUS_H_
#define Y_ABSL_STATUS_STATUS_H_

#include <cassert>
#include <cstdint>
#include <ostream>
#include <util/generic/string.h>
#include <utility>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/macros.h"
#include "y_absl/base/nullability.h"
#include "y_absl/base/optimization.h"
#include "y_absl/functional/function_ref.h"
#include "y_absl/status/internal/status_internal.h"
#include "y_absl/strings/cord.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/types/optional.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

// y_absl::StatusCode
//
// An `y_absl::StatusCode` is an enumerated type indicating either no error ("OK")
// or an error condition. In most cases, an `y_absl::Status` indicates a
// recoverable error, and the purpose of signalling an error is to indicate what
// action to take in response to that error. These error codes map to the proto
// RPC error codes indicated in https://cloud.google.com/apis/design/errors.
//
// The errors listed below are the canonical errors associated with
// `y_absl::Status` and are used throughout the codebase. As a result, these
// error codes are somewhat generic.
//
// In general, try to return the most specific error that applies if more than
// one error may pertain. For example, prefer `kOutOfRange` over
// `kFailedPrecondition` if both codes apply. Similarly prefer `kNotFound` or
// `kAlreadyExists` over `kFailedPrecondition`.
//
// Because these errors may cross RPC boundaries, these codes are tied to the
// `google.rpc.Code` definitions within
// https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
// The string value of these RPC codes is denoted within each enum below.
//
// If your error handling code requires more context, you can attach payloads
// to your status. See `y_absl::Status::SetPayload()` and
// `y_absl::Status::GetPayload()` below.
enum class StatusCode : int {
  // StatusCode::kOk
  //
  // kOK (gRPC code "OK") does not indicate an error; this value is returned on
  // success. It is typical to check for this value before proceeding on any
  // given call across an API or RPC boundary. To check this value, use the
  // `y_absl::Status::ok()` member function rather than inspecting the raw code.
  kOk = 0,

  // StatusCode::kCancelled
  //
  // kCancelled (gRPC code "CANCELLED") indicates the operation was cancelled,
  // typically by the caller.
  kCancelled = 1,

  // StatusCode::kUnknown
  //
  // kUnknown (gRPC code "UNKNOWN") indicates an unknown error occurred. In
  // general, more specific errors should be raised, if possible. Errors raised
  // by APIs that do not return enough error information may be converted to
  // this error.
  kUnknown = 2,

  // StatusCode::kInvalidArgument
  //
  // kInvalidArgument (gRPC code "INVALID_ARGUMENT") indicates the caller
  // specified an invalid argument, such as a malformed filename. Note that use
  // of such errors should be narrowly limited to indicate the invalid nature of
  // the arguments themselves. Errors with validly formed arguments that may
  // cause errors with the state of the receiving system should be denoted with
  // `kFailedPrecondition` instead.
  kInvalidArgument = 3,

  // StatusCode::kDeadlineExceeded
  //
  // kDeadlineExceeded (gRPC code "DEADLINE_EXCEEDED") indicates a deadline
  // expired before the operation could complete. For operations that may change
  // state within a system, this error may be returned even if the operation has
  // completed successfully. For example, a successful response from a server
  // could have been delayed long enough for the deadline to expire.
  kDeadlineExceeded = 4,

  // StatusCode::kNotFound
  //
  // kNotFound (gRPC code "NOT_FOUND") indicates some requested entity (such as
  // a file or directory) was not found.
  //
  // `kNotFound` is useful if a request should be denied for an entire class of
  // users, such as during a gradual feature rollout or undocumented allow list.
  // If a request should be denied for specific sets of users, such as through
  // user-based access control, use `kPermissionDenied` instead.
  kNotFound = 5,

  // StatusCode::kAlreadyExists
  //
  // kAlreadyExists (gRPC code "ALREADY_EXISTS") indicates that the entity a
  // caller attempted to create (such as a file or directory) is already
  // present.
  kAlreadyExists = 6,

  // StatusCode::kPermissionDenied
  //
  // kPermissionDenied (gRPC code "PERMISSION_DENIED") indicates that the caller
  // does not have permission to execute the specified operation. Note that this
  // error is different than an error due to an *un*authenticated user. This
  // error code does not imply the request is valid or the requested entity
  // exists or satisfies any other pre-conditions.
  //
  // `kPermissionDenied` must not be used for rejections caused by exhausting
  // some resource. Instead, use `kResourceExhausted` for those errors.
  // `kPermissionDenied` must not be used if the caller cannot be identified.
  // Instead, use `kUnauthenticated` for those errors.
  kPermissionDenied = 7,

  // StatusCode::kResourceExhausted
  //
  // kResourceExhausted (gRPC code "RESOURCE_EXHAUSTED") indicates some resource
  // has been exhausted, perhaps a per-user quota, or perhaps the entire file
  // system is out of space.
  kResourceExhausted = 8,

  // StatusCode::kFailedPrecondition
  //
  // kFailedPrecondition (gRPC code "FAILED_PRECONDITION") indicates that the
  // operation was rejected because the system is not in a state required for
  // the operation's execution. For example, a directory to be deleted may be
  // non-empty, an "rmdir" operation is applied to a non-directory, etc.
  //
  // Some guidelines that may help a service implementer in deciding between
  // `kFailedPrecondition`, `kAborted`, and `kUnavailable`:
  //
  //  (a) Use `kUnavailable` if the client can retry just the failing call.
  //  (b) Use `kAborted` if the client should retry at a higher transaction
  //      level (such as when a client-specified test-and-set fails, indicating
  //      the client should restart a read-modify-write sequence).
  //  (c) Use `kFailedPrecondition` if the client should not retry until
  //      the system state has been explicitly fixed. For example, if a "rmdir"
  //      fails because the directory is non-empty, `kFailedPrecondition`
  //      should be returned since the client should not retry unless
  //      the files are deleted from the directory.
  kFailedPrecondition = 9,

  // StatusCode::kAborted
  //
  // kAborted (gRPC code "ABORTED") indicates the operation was aborted,
  // typically due to a concurrency issue such as a sequencer check failure or a
  // failed transaction.
  //
  // See the guidelines above for deciding between `kFailedPrecondition`,
  // `kAborted`, and `kUnavailable`.
  kAborted = 10,

  // StatusCode::kOutOfRange
  //
  // kOutOfRange (gRPC code "OUT_OF_RANGE") indicates the operation was
  // attempted past the valid range, such as seeking or reading past an
  // end-of-file.
  //
  // Unlike `kInvalidArgument`, this error indicates a problem that may
  // be fixed if the system state changes. For example, a 32-bit file
  // system will generate `kInvalidArgument` if asked to read at an
  // offset that is not in the range [0,2^32-1], but it will generate
  // `kOutOfRange` if asked to read from an offset past the current
  // file size.
  //
  // There is a fair bit of overlap between `kFailedPrecondition` and
  // `kOutOfRange`.  We recommend using `kOutOfRange` (the more specific
  // error) when it applies so that callers who are iterating through
  // a space can easily look for an `kOutOfRange` error to detect when
  // they are done.
  kOutOfRange = 11,

  // StatusCode::kUnimplemented
  //
  // kUnimplemented (gRPC code "UNIMPLEMENTED") indicates the operation is not
  // implemented or supported in this service. In this case, the operation
  // should not be re-attempted.
  kUnimplemented = 12,

  // StatusCode::kInternal
  //
  // kInternal (gRPC code "INTERNAL") indicates an internal error has occurred
  // and some invariants expected by the underlying system have not been
  // satisfied. This error code is reserved for serious errors.
  kInternal = 13,

  // StatusCode::kUnavailable
  //
  // kUnavailable (gRPC code "UNAVAILABLE") indicates the service is currently
  // unavailable and that this is most likely a transient condition. An error
  // such as this can be corrected by retrying with a backoff scheme. Note that
  // it is not always safe to retry non-idempotent operations.
  //
  // See the guidelines above for deciding between `kFailedPrecondition`,
  // `kAborted`, and `kUnavailable`.
  kUnavailable = 14,

  // StatusCode::kDataLoss
  //
  // kDataLoss (gRPC code "DATA_LOSS") indicates that unrecoverable data loss or
  // corruption has occurred. As this error is serious, proper alerting should
  // be attached to errors such as this.
  kDataLoss = 15,

  // StatusCode::kUnauthenticated
  //
  // kUnauthenticated (gRPC code "UNAUTHENTICATED") indicates that the request
  // does not have valid authentication credentials for the operation. Correct
  // the authentication and try again.
  kUnauthenticated = 16,

  // StatusCode::DoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_
  //
  // NOTE: this error code entry should not be used and you should not rely on
  // its value, which may change.
  //
  // The purpose of this enumerated value is to force people who handle status
  // codes with `switch()` statements to *not* simply enumerate all possible
  // values, but instead provide a "default:" case. Providing such a default
  // case ensures that code will compile when new codes are added.
  kDoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

// StatusCodeToString()
//
// Returns the name for the status code, or "" if it is an unknown value.
TString StatusCodeToString(StatusCode code);

// operator<<
//
// Streams StatusCodeToString(code) to `os`.
std::ostream& operator<<(std::ostream& os, StatusCode code);

// y_absl::StatusToStringMode
//
// An `y_absl::StatusToStringMode` is an enumerated type indicating how
// `y_absl::Status::ToString()` should construct the output string for a non-ok
// status.
enum class StatusToStringMode : int {
  // ToString will not contain any extra data (such as payloads). It will only
  // contain the error code and message, if any.
  kWithNoExtraData = 0,
  // ToString will contain the payloads.
  kWithPayload = 1 << 0,
  // ToString will include all the extra data this Status has.
  kWithEverything = ~kWithNoExtraData,
  // Default mode used by ToString. Its exact value might change in the future.
  kDefault = kWithPayload,
};

// y_absl::StatusToStringMode is specified as a bitmask type, which means the
// following operations must be provided:
inline constexpr StatusToStringMode operator&(StatusToStringMode lhs,
                                              StatusToStringMode rhs) {
  return static_cast<StatusToStringMode>(static_cast<int>(lhs) &
                                         static_cast<int>(rhs));
}
inline constexpr StatusToStringMode operator|(StatusToStringMode lhs,
                                              StatusToStringMode rhs) {
  return static_cast<StatusToStringMode>(static_cast<int>(lhs) |
                                         static_cast<int>(rhs));
}
inline constexpr StatusToStringMode operator^(StatusToStringMode lhs,
                                              StatusToStringMode rhs) {
  return static_cast<StatusToStringMode>(static_cast<int>(lhs) ^
                                         static_cast<int>(rhs));
}
inline constexpr StatusToStringMode operator~(StatusToStringMode arg) {
  return static_cast<StatusToStringMode>(~static_cast<int>(arg));
}
inline StatusToStringMode& operator&=(StatusToStringMode& lhs,
                                      StatusToStringMode rhs) {
  lhs = lhs & rhs;
  return lhs;
}
inline StatusToStringMode& operator|=(StatusToStringMode& lhs,
                                      StatusToStringMode rhs) {
  lhs = lhs | rhs;
  return lhs;
}
inline StatusToStringMode& operator^=(StatusToStringMode& lhs,
                                      StatusToStringMode rhs) {
  lhs = lhs ^ rhs;
  return lhs;
}

// y_absl::Status
//
// The `y_absl::Status` class is generally used to gracefully handle errors
// across API boundaries (and in particular across RPC boundaries). Some of
// these errors may be recoverable, but others may not. Most
// functions which can produce a recoverable error should be designed to return
// either an `y_absl::Status` (or the similar `y_absl::StatusOr<T>`, which holds
// either an object of type `T` or an error).
//
// API developers should construct their functions to return `y_absl::OkStatus()`
// upon success, or an `y_absl::StatusCode` upon another type of error (e.g
// an `y_absl::StatusCode::kInvalidArgument` error). The API provides convenience
// functions to construct each status code.
//
// Example:
//
// y_absl::Status myFunction(y_absl::string_view fname, ...) {
//   ...
//   // encounter error
//   if (error condition) {
//     // Construct an y_absl::StatusCode::kInvalidArgument error
//     return y_absl::InvalidArgumentError("bad mode");
//   }
//   // else, return OK
//   return y_absl::OkStatus();
// }
//
// Users handling status error codes should prefer checking for an OK status
// using the `ok()` member function. Handling multiple error codes may justify
// use of switch statement, but only check for error codes you know how to
// handle; do not try to exhaustively match against all canonical error codes.
// Errors that cannot be handled should be logged and/or propagated for higher
// levels to deal with. If you do use a switch statement, make sure that you
// also provide a `default:` switch case, so that code does not break as other
// canonical codes are added to the API.
//
// Example:
//
//   y_absl::Status result = DoSomething();
//   if (!result.ok()) {
//     LOG(ERROR) << result;
//   }
//
//   // Provide a default if switching on multiple error codes
//   switch (result.code()) {
//     // The user hasn't authenticated. Ask them to reauth
//     case y_absl::StatusCode::kUnauthenticated:
//       DoReAuth();
//       break;
//     // The user does not have permission. Log an error.
//     case y_absl::StatusCode::kPermissionDenied:
//       LOG(ERROR) << result;
//       break;
//     // Propagate the error otherwise.
//     default:
//       return true;
//   }
//
// An `y_absl::Status` can optionally include a payload with more information
// about the error. Typically, this payload serves one of several purposes:
//
//   * It may provide more fine-grained semantic information about the error to
//     facilitate actionable remedies.
//   * It may provide human-readable contextual information that is more
//     appropriate to display to an end user.
//
// Example:
//
//   y_absl::Status result = DoSomething();
//   // Inform user to retry after 30 seconds
//   // See more error details in googleapis/google/rpc/error_details.proto
//   if (y_absl::IsResourceExhausted(result)) {
//     google::rpc::RetryInfo info;
//     info.retry_delay().seconds() = 30;
//     // Payloads require a unique key (a URL to ensure no collisions with
//     // other payloads), and an `y_absl::Cord` to hold the encoded data.
//     y_absl::string_view url = "type.googleapis.com/google.rpc.RetryInfo";
//     result.SetPayload(url, info.SerializeAsCord());
//     return result;
//   }
//
// For documentation see https://abseil.io/docs/cpp/guides/status.
//
// Returned Status objects may not be ignored. status_internal.h has a forward
// declaration of the form
// class Y_ABSL_MUST_USE_RESULT Status;
class Y_ABSL_ATTRIBUTE_TRIVIAL_ABI Status final {
 public:
  // Constructors

  // This default constructor creates an OK status with no message or payload.
  // Avoid this constructor and prefer explicit construction of an OK status
  // with `y_absl::OkStatus()`.
  Status();

  // Creates a status in the canonical error space with the specified
  // `y_absl::StatusCode` and error message.  If `code == y_absl::StatusCode::kOk`,  // NOLINT
  // `msg` is ignored and an object identical to an OK status is constructed.
  //
  // The `msg` string must be in UTF-8. The implementation may complain (e.g.,  // NOLINT
  // by printing a warning) if it is not.
  Status(y_absl::StatusCode code, y_absl::string_view msg);

  Status(const Status&);
  Status& operator=(const Status& x);

  // Move operators

  // The moved-from state is valid but unspecified.
  Status(Status&&) noexcept;
  Status& operator=(Status&&) noexcept;

  ~Status();

  // Status::Update()
  //
  // Updates the existing status with `new_status` provided that `this->ok()`.
  // If the existing status already contains a non-OK error, this update has no
  // effect and preserves the current data. Note that this behavior may change
  // in the future to augment a current non-ok status with additional
  // information about `new_status`.
  //
  // `Update()` provides a convenient way of keeping track of the first error
  // encountered.
  //
  // Example:
  //   // Instead of "if (overall_status.ok()) overall_status = new_status"
  //   overall_status.Update(new_status);
  //
  void Update(const Status& new_status);
  void Update(Status&& new_status);

  // Status::ok()
  //
  // Returns `true` if `this->code()` == `y_absl::StatusCode::kOk`,
  // indicating the absence of an error.
  // Prefer checking for an OK status using this member function.
  Y_ABSL_MUST_USE_RESULT bool ok() const;

  // Status::code()
  //
  // Returns the canonical error code of type `y_absl::StatusCode` of this status.
  y_absl::StatusCode code() const;

  // Status::raw_code()
  //
  // Returns a raw (canonical) error code corresponding to the enum value of
  // `google.rpc.Code` definitions within
  // https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto.
  // These values could be out of the range of canonical `y_absl::StatusCode`
  // enum values.
  //
  // NOTE: This function should only be called when converting to an associated
  // wire format. Use `Status::code()` for error handling.
  int raw_code() const;

  // Status::message()
  //
  // Returns the error message associated with this error code, if available.
  // Note that this message rarely describes the error code.  It is not unusual
  // for the error message to be the empty string. As a result, prefer
  // `operator<<` or `Status::ToString()` for debug logging.
  y_absl::string_view message() const;

  friend bool operator==(const Status&, const Status&);
  friend bool operator!=(const Status&, const Status&);

  // Status::ToString()
  //
  // Returns a string based on the `mode`. By default, it returns combination of
  // the error code name, the message and any associated payload messages. This
  // string is designed simply to be human readable and its exact format should
  // not be load bearing. Do not depend on the exact format of the result of
  // `ToString()` which is subject to change.
  //
  // The printed code name and the message are generally substrings of the
  // result, and the payloads to be printed use the status payload printer
  // mechanism (which is internal).
  TString ToString(
      StatusToStringMode mode = StatusToStringMode::kDefault) const;

  // Support `y_absl::StrCat`, `y_absl::StrFormat`, etc.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Status& status) {
    sink.Append(status.ToString(StatusToStringMode::kWithEverything));
  }

  // Status::IgnoreError()
  //
  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  // swap()
  //
  // Swap the contents of one status with another.
  friend void swap(Status& a, Status& b) noexcept;

  //----------------------------------------------------------------------------
  // Payload Management APIs
  //----------------------------------------------------------------------------

  // A payload may be attached to a status to provide additional context to an
  // error that may not be satisfied by an existing `y_absl::StatusCode`.
  // Typically, this payload serves one of several purposes:
  //
  //   * It may provide more fine-grained semantic information about the error
  //     to facilitate actionable remedies.
  //   * It may provide human-readable contextual information that is more
  //     appropriate to display to an end user.
  //
  // A payload consists of a [key,value] pair, where the key is a string
  // referring to a unique "type URL" and the value is an object of type
  // `y_absl::Cord` to hold the contextual data.
  //
  // The "type URL" should be unique and follow the format of a URL
  // (https://en.wikipedia.org/wiki/URL) and, ideally, provide some
  // documentation or schema on how to interpret its associated data. For
  // example, the default type URL for a protobuf message type is
  // "type.googleapis.com/packagename.messagename". Other custom wire formats
  // should define the format of type URL in a similar practice so as to
  // minimize the chance of conflict between type URLs.
  // Users should ensure that the type URL can be mapped to a concrete
  // C++ type if they want to deserialize the payload and read it effectively.
  //
  // To attach a payload to a status object, call `Status::SetPayload()`,
  // passing it the type URL and an `y_absl::Cord` of associated data. Similarly,
  // to extract the payload from a status, call `Status::GetPayload()`. You
  // may attach multiple payloads (with differing type URLs) to any given
  // status object, provided that the status is currently exhibiting an error
  // code (i.e. is not OK).

  // Status::GetPayload()
  //
  // Gets the payload of a status given its unique `type_url` key, if present.
  y_absl::optional<y_absl::Cord> GetPayload(y_absl::string_view type_url) const;

  // Status::SetPayload()
  //
  // Sets the payload for a non-ok status using a `type_url` key, overwriting
  // any existing payload for that `type_url`.
  //
  // NOTE: This function does nothing if the Status is ok.
  void SetPayload(y_absl::string_view type_url, y_absl::Cord payload);

  // Status::ErasePayload()
  //
  // Erases the payload corresponding to the `type_url` key.  Returns `true` if
  // the payload was present.
  bool ErasePayload(y_absl::string_view type_url);

  // Status::ForEachPayload()
  //
  // Iterates over the stored payloads and calls the
  // `visitor(type_key, payload)` callable for each one.
  //
  // NOTE: The order of calls to `visitor()` is not specified and may change at
  // any time.
  //
  // NOTE: Any mutation on the same 'y_absl::Status' object during visitation is
  // forbidden and could result in undefined behavior.
  void ForEachPayload(
      y_absl::FunctionRef<void(y_absl::string_view, const y_absl::Cord&)> visitor)
      const;

 private:
  friend Status CancelledError();

  // Creates a status in the canonical error space with the specified
  // code, and an empty error message.
  explicit Status(y_absl::StatusCode code);

  // Underlying constructor for status from a rep_.
  explicit Status(uintptr_t rep) : rep_(rep) {}

  static void Ref(uintptr_t rep);
  static void Unref(uintptr_t rep);

  // REQUIRES: !ok()
  // Ensures rep is not inlined or shared with any other Status.
  static y_absl::Nonnull<status_internal::StatusRep*> PrepareToModify(
      uintptr_t rep);

  // MSVC 14.0 limitation requires the const.
  static constexpr const char kMovedFromString[] =
      "Status accessed after move.";

  static y_absl::Nonnull<const TString*> EmptyString();
  static y_absl::Nonnull<const TString*> MovedFromString();

  // Returns whether rep contains an inlined representation.
  // See rep_ for details.
  static constexpr bool IsInlined(uintptr_t rep);

  // Indicates whether this Status was the rhs of a move operation. See rep_
  // for details.
  static constexpr bool IsMovedFrom(uintptr_t rep);
  static constexpr uintptr_t MovedFromRep();

  // Convert between error::Code and the inlined uintptr_t representation used
  // by rep_. See rep_ for details.
  static constexpr uintptr_t CodeToInlinedRep(y_absl::StatusCode code);
  static constexpr y_absl::StatusCode InlinedRepToCode(uintptr_t rep);

  // Converts between StatusRep* and the external uintptr_t representation used
  // by rep_. See rep_ for details.
  static uintptr_t PointerToRep(status_internal::StatusRep* r);
  static y_absl::Nonnull<const status_internal::StatusRep*> RepToPointer(
      uintptr_t r);

  static TString ToStringSlow(uintptr_t rep, StatusToStringMode mode);

  // Status supports two different representations.
  //  - When the low bit is set it is an inlined representation.
  //    It uses the canonical error space, no message or payload.
  //    The error code is (rep_ >> 2).
  //    The (rep_ & 2) bit is the "moved from" indicator, used in IsMovedFrom().
  //  - When the low bit is off it is an external representation.
  //    In this case all the data comes from a heap allocated Rep object.
  //    rep_ is a status_internal::StatusRep* pointer to that structure.
  uintptr_t rep_;

  friend class status_internal::StatusRep;
};

// OkStatus()
//
// Returns an OK status, equivalent to a default constructed instance. Prefer
// usage of `y_absl::OkStatus()` when constructing such an OK status.
Status OkStatus();

// operator<<()
//
// Prints a human-readable representation of `x` to `os`.
std::ostream& operator<<(std::ostream& os, const Status& x);

// IsAborted()
// IsAlreadyExists()
// IsCancelled()
// IsDataLoss()
// IsDeadlineExceeded()
// IsFailedPrecondition()
// IsInternal()
// IsInvalidArgument()
// IsNotFound()
// IsOutOfRange()
// IsPermissionDenied()
// IsResourceExhausted()
// IsUnauthenticated()
// IsUnavailable()
// IsUnimplemented()
// IsUnknown()
//
// These convenience functions return `true` if a given status matches the
// `y_absl::StatusCode` error code of its associated function.
Y_ABSL_MUST_USE_RESULT bool IsAborted(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsAlreadyExists(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsCancelled(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsDataLoss(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsDeadlineExceeded(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsFailedPrecondition(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsInternal(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsInvalidArgument(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsNotFound(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsOutOfRange(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsPermissionDenied(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsResourceExhausted(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsUnauthenticated(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsUnavailable(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsUnimplemented(const Status& status);
Y_ABSL_MUST_USE_RESULT bool IsUnknown(const Status& status);

// AbortedError()
// AlreadyExistsError()
// CancelledError()
// DataLossError()
// DeadlineExceededError()
// FailedPreconditionError()
// InternalError()
// InvalidArgumentError()
// NotFoundError()
// OutOfRangeError()
// PermissionDeniedError()
// ResourceExhaustedError()
// UnauthenticatedError()
// UnavailableError()
// UnimplementedError()
// UnknownError()
//
// These convenience functions create an `y_absl::Status` object with an error
// code as indicated by the associated function name, using the error message
// passed in `message`.
Status AbortedError(y_absl::string_view message);
Status AlreadyExistsError(y_absl::string_view message);
Status CancelledError(y_absl::string_view message);
Status DataLossError(y_absl::string_view message);
Status DeadlineExceededError(y_absl::string_view message);
Status FailedPreconditionError(y_absl::string_view message);
Status InternalError(y_absl::string_view message);
Status InvalidArgumentError(y_absl::string_view message);
Status NotFoundError(y_absl::string_view message);
Status OutOfRangeError(y_absl::string_view message);
Status PermissionDeniedError(y_absl::string_view message);
Status ResourceExhaustedError(y_absl::string_view message);
Status UnauthenticatedError(y_absl::string_view message);
Status UnavailableError(y_absl::string_view message);
Status UnimplementedError(y_absl::string_view message);
Status UnknownError(y_absl::string_view message);

// ErrnoToStatusCode()
//
// Returns the StatusCode for `error_number`, which should be an `errno` value.
// See https://en.cppreference.com/w/cpp/error/errno_macros and similar
// references.
y_absl::StatusCode ErrnoToStatusCode(int error_number);

// ErrnoToStatus()
//
// Convenience function that creates a `y_absl::Status` using an `error_number`,
// which should be an `errno` value.
Status ErrnoToStatus(int error_number, y_absl::string_view message);

//------------------------------------------------------------------------------
// Implementation details follow
//------------------------------------------------------------------------------

inline Status::Status() : Status(y_absl::StatusCode::kOk) {}

inline Status::Status(y_absl::StatusCode code) : Status(CodeToInlinedRep(code)) {}

inline Status::Status(const Status& x) : Status(x.rep_) { Ref(rep_); }

inline Status& Status::operator=(const Status& x) {
  uintptr_t old_rep = rep_;
  if (x.rep_ != old_rep) {
    Ref(x.rep_);
    rep_ = x.rep_;
    Unref(old_rep);
  }
  return *this;
}

inline Status::Status(Status&& x) noexcept : Status(x.rep_) {
  x.rep_ = MovedFromRep();
}

inline Status& Status::operator=(Status&& x) noexcept {
  uintptr_t old_rep = rep_;
  if (x.rep_ != old_rep) {
    rep_ = x.rep_;
    x.rep_ = MovedFromRep();
    Unref(old_rep);
  }
  return *this;
}

inline void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

inline void Status::Update(Status&& new_status) {
  if (ok()) {
    *this = std::move(new_status);
  }
}

inline Status::~Status() { Unref(rep_); }

inline bool Status::ok() const {
  return rep_ == CodeToInlinedRep(y_absl::StatusCode::kOk);
}

inline y_absl::StatusCode Status::code() const {
  return status_internal::MapToLocalCode(raw_code());
}

inline int Status::raw_code() const {
  if (IsInlined(rep_)) return static_cast<int>(InlinedRepToCode(rep_));
  return static_cast<int>(RepToPointer(rep_)->code());
}

inline y_absl::string_view Status::message() const {
  return !IsInlined(rep_)
             ? RepToPointer(rep_)->message()
             : (IsMovedFrom(rep_) ? y_absl::string_view(kMovedFromString)
                                  : y_absl::string_view());
}

inline bool operator==(const Status& lhs, const Status& rhs) {
  if (lhs.rep_ == rhs.rep_) return true;
  if (Status::IsInlined(lhs.rep_)) return false;
  if (Status::IsInlined(rhs.rep_)) return false;
  return *Status::RepToPointer(lhs.rep_) == *Status::RepToPointer(rhs.rep_);
}

inline bool operator!=(const Status& lhs, const Status& rhs) {
  return !(lhs == rhs);
}

inline TString Status::ToString(StatusToStringMode mode) const {
  return ok() ? "OK" : ToStringSlow(rep_, mode);
}

inline void Status::IgnoreError() const {
  // no-op
}

inline void swap(y_absl::Status& a, y_absl::Status& b) noexcept {
  using std::swap;
  swap(a.rep_, b.rep_);
}

inline y_absl::optional<y_absl::Cord> Status::GetPayload(
    y_absl::string_view type_url) const {
  if (IsInlined(rep_)) return y_absl::nullopt;
  return RepToPointer(rep_)->GetPayload(type_url);
}

inline void Status::SetPayload(y_absl::string_view type_url, y_absl::Cord payload) {
  if (ok()) return;
  status_internal::StatusRep* rep = PrepareToModify(rep_);
  rep->SetPayload(type_url, std::move(payload));
  rep_ = PointerToRep(rep);
}

inline bool Status::ErasePayload(y_absl::string_view type_url) {
  if (IsInlined(rep_)) return false;
  status_internal::StatusRep* rep = PrepareToModify(rep_);
  auto res = rep->ErasePayload(type_url);
  rep_ = res.new_rep;
  return res.erased;
}

inline void Status::ForEachPayload(
    y_absl::FunctionRef<void(y_absl::string_view, const y_absl::Cord&)> visitor)
    const {
  if (IsInlined(rep_)) return;
  RepToPointer(rep_)->ForEachPayload(visitor);
}

constexpr bool Status::IsInlined(uintptr_t rep) { return (rep & 1) != 0; }

constexpr bool Status::IsMovedFrom(uintptr_t rep) { return (rep & 2) != 0; }

constexpr uintptr_t Status::CodeToInlinedRep(y_absl::StatusCode code) {
  return (static_cast<uintptr_t>(code) << 2) + 1;
}

constexpr y_absl::StatusCode Status::InlinedRepToCode(uintptr_t rep) {
  Y_ABSL_ASSERT(IsInlined(rep));
  return static_cast<y_absl::StatusCode>(rep >> 2);
}

constexpr uintptr_t Status::MovedFromRep() {
  return CodeToInlinedRep(y_absl::StatusCode::kInternal) | 2;
}

inline y_absl::Nonnull<const status_internal::StatusRep*> Status::RepToPointer(
    uintptr_t rep) {
  assert(!IsInlined(rep));
  return reinterpret_cast<const status_internal::StatusRep*>(rep);
}

inline uintptr_t Status::PointerToRep(
    y_absl::Nonnull<status_internal::StatusRep*> rep) {
  return reinterpret_cast<uintptr_t>(rep);
}

inline void Status::Ref(uintptr_t rep) {
  if (!IsInlined(rep)) RepToPointer(rep)->Ref();
}

inline void Status::Unref(uintptr_t rep) {
  if (!IsInlined(rep)) RepToPointer(rep)->Unref();
}

inline Status OkStatus() { return Status(); }

// Creates a `Status` object with the `y_absl::StatusCode::kCancelled` error code
// and an empty message. It is provided only for efficiency, given that
// message-less kCancelled errors are common in the infrastructure.
inline Status CancelledError() { return Status(y_absl::StatusCode::kCancelled); }

// Retrieves a message's status as a null terminated C string. The lifetime of
// this string is tied to the lifetime of the status object itself.
//
// If the status's message is empty, the empty string is returned.
//
// StatusMessageAsCStr exists for C support. Use `status.message()` in C++.
y_absl::Nonnull<const char*> StatusMessageAsCStr(
    const Status& status Y_ABSL_ATTRIBUTE_LIFETIME_BOUND);

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STATUS_STATUS_H_
