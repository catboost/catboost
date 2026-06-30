// Copyright 2020 The Abseil Authors.
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
// File: statusor.h
// -----------------------------------------------------------------------------
//
// An `y_absl::StatusOr<T>` represents a union of an `y_absl::Status` object
// and an object of type `T`. The `y_absl::StatusOr<T>` will either contain an
// object of type `T` (indicating a successful operation), or an error (of type
// `y_absl::Status`) explaining why such a value is not present.
//
// In general, check the success of an operation returning an
// `y_absl::StatusOr<T>` like you would an `y_absl::Status` by using the `ok()`
// member function.
//
// Example:
//
//   StatusOr<Foo> result = Calculation();
//   if (result.ok()) {
//     result->DoSomethingCool();
//   } else {
//     LOG(ERROR) << result.status();
//   }
#ifndef Y_ABSL_STATUS_STATUSOR_H_
#define Y_ABSL_STATUS_STATUSOR_H_

#include <exception>
#include <initializer_list>
#include <new>
#include <ostream>
#include <util/generic/string.h>
#include <type_traits>
#include <utility>

#include "y_absl/base/attributes.h"
#include "y_absl/base/nullability.h"
#include "y_absl/base/call_once.h"
#include "y_absl/meta/type_traits.h"
#include "y_absl/status/internal/statusor_internal.h"
#include "y_absl/status/status.h"
#include "y_absl/strings/has_absl_stringify.h"
#include "y_absl/strings/has_ostream_operator.h"
#include "y_absl/strings/str_format.h"
#include "y_absl/types/variant.h"
#include "y_absl/utility/utility.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

// BadStatusOrAccess
//
// This class defines the type of object to throw (if exceptions are enabled),
// when accessing the value of an `y_absl::StatusOr<T>` object that does not
// contain a value. This behavior is analogous to that of
// `std::bad_optional_access` in the case of accessing an invalid
// `std::optional` value.
//
// Example:
//
// try {
//   y_absl::StatusOr<int> v = FetchInt();
//   DoWork(v.value());  // Accessing value() when not "OK" may throw
// } catch (y_absl::BadStatusOrAccess& ex) {
//   LOG(ERROR) << ex.status();
// }
class BadStatusOrAccess : public std::exception {
 public:
  explicit BadStatusOrAccess(y_absl::Status status);
  ~BadStatusOrAccess() override = default;

  BadStatusOrAccess(const BadStatusOrAccess& other);
  BadStatusOrAccess& operator=(const BadStatusOrAccess& other);
  BadStatusOrAccess(BadStatusOrAccess&& other);
  BadStatusOrAccess& operator=(BadStatusOrAccess&& other);

  // BadStatusOrAccess::what()
  //
  // Returns the associated explanatory string of the `y_absl::StatusOr<T>`
  // object's error code. This function contains information about the failing
  // status, but its exact formatting may change and should not be depended on.
  //
  // The pointer of this string is guaranteed to be valid until any non-const
  // function is invoked on the exception object.
  y_absl::Nonnull<const char*> what() const noexcept override;

  // BadStatusOrAccess::status()
  //
  // Returns the associated `y_absl::Status` of the `y_absl::StatusOr<T>` object's
  // error.
  const y_absl::Status& status() const;

 private:
  void InitWhat() const;

  y_absl::Status status_;
  mutable y_absl::once_flag init_what_;
  mutable TString what_;
};

// Returned StatusOr objects may not be ignored.
template <typename T>
#if Y_ABSL_HAVE_CPP_ATTRIBUTE(nodiscard)
// TODO(b/176172494): Y_ABSL_MUST_USE_RESULT should expand to the more strict
// [[nodiscard]]. For now, just use [[nodiscard]] directly when it is available.
class [[nodiscard]] StatusOr;
#else
class Y_ABSL_MUST_USE_RESULT StatusOr;
#endif  // Y_ABSL_HAVE_CPP_ATTRIBUTE(nodiscard)

// y_absl::StatusOr<T>
//
// The `y_absl::StatusOr<T>` class template is a union of an `y_absl::Status` object
// and an object of type `T`. The `y_absl::StatusOr<T>` models an object that is
// either a usable object, or an error (of type `y_absl::Status`) explaining why
// such an object is not present. An `y_absl::StatusOr<T>` is typically the return
// value of a function which may fail.
//
// An `y_absl::StatusOr<T>` can never hold an "OK" status (an
// `y_absl::StatusCode::kOk` value); instead, the presence of an object of type
// `T` indicates success. Instead of checking for a `kOk` value, use the
// `y_absl::StatusOr<T>::ok()` member function. (It is for this reason, and code
// readability, that using the `ok()` function is preferred for `y_absl::Status`
// as well.)
//
// Example:
//
//   StatusOr<Foo> result = DoBigCalculationThatCouldFail();
//   if (result.ok()) {
//     result->DoSomethingCool();
//   } else {
//     LOG(ERROR) << result.status();
//   }
//
// Accessing the object held by an `y_absl::StatusOr<T>` should be performed via
// `operator*` or `operator->`, after a call to `ok()` confirms that the
// `y_absl::StatusOr<T>` holds an object of type `T`:
//
// Example:
//
//   y_absl::StatusOr<int> i = GetCount();
//   if (i.ok()) {
//     updated_total += *i;
//   }
//
// NOTE: using `y_absl::StatusOr<T>::value()` when no valid value is present will
// throw an exception if exceptions are enabled or terminate the process when
// exceptions are not enabled.
//
// Example:
//
//   StatusOr<Foo> result = DoBigCalculationThatCouldFail();
//   const Foo& foo = result.value();    // Crash/exception if no value present
//   foo.DoSomethingCool();
//
// A `y_absl::StatusOr<T*>` can be constructed from a null pointer like any other
// pointer value, and the result will be that `ok()` returns `true` and
// `value()` returns `nullptr`. Checking the value of pointer in an
// `y_absl::StatusOr<T*>` generally requires a bit more care, to ensure both that
// a value is present and that value is not null:
//
//  StatusOr<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (!result.ok()) {
//    LOG(ERROR) << result.status();
//  } else if (*result == nullptr) {
//    LOG(ERROR) << "Unexpected null pointer";
//  } else {
//    (*result)->DoSomethingCool();
//  }
//
// Example factory implementation returning StatusOr<T>:
//
//  StatusOr<Foo> FooFactory::MakeFoo(int arg) {
//    if (arg <= 0) {
//      return y_absl::Status(y_absl::StatusCode::kInvalidArgument,
//                          "Arg must be positive");
//    }
//    return Foo(arg);
//  }
template <typename T>
class StatusOr : private internal_statusor::StatusOrData<T>,
                 private internal_statusor::CopyCtorBase<T>,
                 private internal_statusor::MoveCtorBase<T>,
                 private internal_statusor::CopyAssignBase<T>,
                 private internal_statusor::MoveAssignBase<T> {
  template <typename U>
  friend class StatusOr;

  typedef internal_statusor::StatusOrData<T> Base;

 public:
  // StatusOr<T>::value_type
  //
  // This instance data provides a generic `value_type` member for use within
  // generic programming. This usage is analogous to that of
  // `optional::value_type` in the case of `std::optional`.
  typedef T value_type;

  // Constructors

  // Constructs a new `y_absl::StatusOr` with an `y_absl::StatusCode::kUnknown`
  // status. This constructor is marked 'explicit' to prevent usages in return
  // values such as 'return {};', under the misconception that
  // `y_absl::StatusOr<std::vector<int>>` will be initialized with an empty
  // vector, instead of an `y_absl::StatusCode::kUnknown` error code.
  explicit StatusOr();

  // `StatusOr<T>` is copy constructible if `T` is copy constructible.
  StatusOr(const StatusOr&) = default;
  // `StatusOr<T>` is copy assignable if `T` is copy constructible and copy
  // assignable.
  StatusOr& operator=(const StatusOr&) = default;

  // `StatusOr<T>` is move constructible if `T` is move constructible.
  StatusOr(StatusOr&&) = default;
  // `StatusOr<T>` is moveAssignable if `T` is move constructible and move
  // assignable.
  StatusOr& operator=(StatusOr&&) = default;

  // Converting Constructors

  // Constructs a new `y_absl::StatusOr<T>` from an `y_absl::StatusOr<U>`, when `T`
  // is constructible from `U`. To avoid ambiguity, these constructors are
  // disabled if `T` is also constructible from `StatusOr<U>.`. This constructor
  // is explicit if and only if the corresponding construction of `T` from `U`
  // is explicit. (This constructor inherits its explicitness from the
  // underlying constructor.)
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                false, T, U, false, const U&>::value,
                            int> = 0>
  StatusOr(const StatusOr<U>& other)  // NOLINT
      : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                false, T, U, true, const U&>::value,
                            int> = 0>
  StatusOr(const StatusOr<U>& other Y_ABSL_ATTRIBUTE_LIFETIME_BOUND)  // NOLINT
      : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                true, T, U, false, const U&>::value,
                            int> = 0>
  explicit StatusOr(const StatusOr<U>& other)
      : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                true, T, U, true, const U&>::value,
                            int> = 0>
  explicit StatusOr(const StatusOr<U>& other Y_ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : Base(static_cast<const typename StatusOr<U>::Base&>(other)) {}

  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                false, T, U, false, U&&>::value,
                            int> = 0>
  StatusOr(StatusOr<U>&& other)  // NOLINT
      : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                false, T, U, true, U&&>::value,
                            int> = 0>
  StatusOr(StatusOr<U>&& other Y_ABSL_ATTRIBUTE_LIFETIME_BOUND)  // NOLINT
      : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                true, T, U, false, U&&>::value,
                            int> = 0>
  explicit StatusOr(StatusOr<U>&& other)
      : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}
  template <typename U, y_absl::enable_if_t<
                            internal_statusor::IsConstructionFromStatusOrValid<
                                true, T, U, true, U&&>::value,
                            int> = 0>
  explicit StatusOr(StatusOr<U>&& other Y_ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : Base(static_cast<typename StatusOr<U>::Base&&>(other)) {}

  // Converting Assignment Operators

  // Creates an `y_absl::StatusOr<T>` through assignment from an
  // `y_absl::StatusOr<U>` when:
  //
  //   * Both `y_absl::StatusOr<T>` and `y_absl::StatusOr<U>` are OK by assigning
  //     `U` to `T` directly.
  //   * `y_absl::StatusOr<T>` is OK and `y_absl::StatusOr<U>` contains an error
  //      code by destroying `y_absl::StatusOr<T>`'s value and assigning from
  //      `y_absl::StatusOr<U>'
  //   * `y_absl::StatusOr<T>` contains an error code and `y_absl::StatusOr<U>` is
  //      OK by directly initializing `T` from `U`.
  //   * Both `y_absl::StatusOr<T>` and `y_absl::StatusOr<U>` contain an error
  //     code by assigning the `Status` in `y_absl::StatusOr<U>` to
  //     `y_absl::StatusOr<T>`
  //
  // These overloads only apply if `y_absl::StatusOr<T>` is constructible and
  // assignable from `y_absl::StatusOr<U>` and `StatusOr<T>` cannot be directly
  // assigned from `StatusOr<U>`.
  template <typename U,
            y_absl::enable_if_t<internal_statusor::IsStatusOrAssignmentValid<
                                  T, const U&, false>::value,
                              int> = 0>
  StatusOr& operator=(const StatusOr<U>& other) {
    this->Assign(other);
    return *this;
  }
  template <typename U,
            y_absl::enable_if_t<internal_statusor::IsStatusOrAssignmentValid<
                                  T, const U&, true>::value,
                              int> = 0>
  StatusOr& operator=(const StatusOr<U>& other Y_ABSL_ATTRIBUTE_LIFETIME_BOUND) {
    this->Assign(other);
    return *this;
  }
  template <typename U,
            y_absl::enable_if_t<internal_statusor::IsStatusOrAssignmentValid<
                                  T, U&&, false>::value,
                              int> = 0>
  StatusOr& operator=(StatusOr<U>&& other) {
    this->Assign(std::move(other));
    return *this;
  }
  template <typename U,
            y_absl::enable_if_t<internal_statusor::IsStatusOrAssignmentValid<
                                  T, U&&, true>::value,
                              int> = 0>
  StatusOr& operator=(StatusOr<U>&& other Y_ABSL_ATTRIBUTE_LIFETIME_BOUND) {
    this->Assign(std::move(other));
    return *this;
  }

  // Constructs a new `y_absl::StatusOr<T>` with a non-ok status. After calling
  // this constructor, `this->ok()` will be `false` and calls to `value()` will
  // crash, or produce an exception if exceptions are enabled.
  //
  // The constructor also takes any type `U` that is convertible to
  // `y_absl::Status`. This constructor is explicit if an only if `U` is not of
  // type `y_absl::Status` and the conversion from `U` to `Status` is explicit.
  //
  // REQUIRES: !Status(std::forward<U>(v)).ok(). This requirement is DCHECKed.
  // In optimized builds, passing y_absl::OkStatus() here will have the effect
  // of passing y_absl::StatusCode::kInternal as a fallback.
  template <typename U = y_absl::Status,
            y_absl::enable_if_t<internal_statusor::IsConstructionFromStatusValid<
                                  false, T, U>::value,
                              int> = 0>
  StatusOr(U&& v) : Base(std::forward<U>(v)) {}

  template <typename U = y_absl::Status,
            y_absl::enable_if_t<internal_statusor::IsConstructionFromStatusValid<
                                  true, T, U>::value,
                              int> = 0>
  explicit StatusOr(U&& v) : Base(std::forward<U>(v)) {}
  template <typename U = y_absl::Status,
            y_absl::enable_if_t<internal_statusor::IsConstructionFromStatusValid<
                                  false, T, U>::value,
                              int> = 0>
  StatusOr& operator=(U&& v) {
    this->AssignStatus(std::forward<U>(v));
    return *this;
  }

  // Perfect-forwarding value assignment operator.

  // If `*this` contains a `T` value before the call, the contained value is
  // assigned from `std::forward<U>(v)`; Otherwise, it is directly-initialized
  // from `std::forward<U>(v)`.
  // This function does not participate in overload unless:
  // 1. `std::is_constructible_v<T, U>` is true,
  // 2. `std::is_assignable_v<T&, U>` is true.
  // 3. `std::is_same_v<StatusOr<T>, std::remove_cvref_t<U>>` is false.
  // 4. Assigning `U` to `T` is not ambiguous:
  //  If `U` is `StatusOr<V>` and `T` is constructible and assignable from
  //  both `StatusOr<V>` and `V`, the assignment is considered bug-prone and
  //  ambiguous thus will fail to compile. For example:
  //    StatusOr<bool> s1 = true;  // s1.ok() && *s1 == true
  //    StatusOr<bool> s2 = false;  // s2.ok() && *s2 == false
  //    s1 = s2;  // ambiguous, `s1 = *s2` or `s1 = bool(s2)`?
  template <typename U = T,
            typename std::enable_if<
                internal_statusor::IsAssignmentValid<T, U, false>::value,
                int>::type = 0>
  StatusOr& operator=(U&& v) {
    this->Assign(std::forward<U>(v));
    return *this;
  }
  template <typename U = T,
            typename std::enable_if<
                internal_statusor::IsAssignmentValid<T, U, true>::value,
                int>::type = 0>
  StatusOr& operator=(U&& v Y_ABSL_ATTRIBUTE_LIFETIME_BOUND) {
    this->Assign(std::forward<U>(v));
    return *this;
  }

  // Constructs the inner value `T` in-place using the provided args, using the
  // `T(args...)` constructor.
  template <typename... Args>
  explicit StatusOr(y_absl::in_place_t, Args&&... args);
  template <typename U, typename... Args>
  explicit StatusOr(y_absl::in_place_t, std::initializer_list<U> ilist,
                    Args&&... args);

  // Constructs the inner value `T` in-place using the provided args, using the
  // `T(U)` (direct-initialization) constructor. This constructor is only valid
  // if `T` can be constructed from a `U`. Can accept move or copy constructors.
  //
  // This constructor is explicit if `U` is not convertible to `T`. To avoid
  // ambiguity, this constructor is disabled if `U` is a `StatusOr<J>`, where
  // `J` is convertible to `T`.
  template <typename U = T,
            y_absl::enable_if_t<internal_statusor::IsConstructionValid<
                                  false, T, U, false>::value,
                              int> = 0>
  StatusOr(U&& u)  // NOLINT
      : StatusOr(y_absl::in_place, std::forward<U>(u)) {}
  template <typename U = T,
            y_absl::enable_if_t<internal_statusor::IsConstructionValid<
                                  false, T, U, true>::value,
                              int> = 0>
  StatusOr(U&& u Y_ABSL_ATTRIBUTE_LIFETIME_BOUND)  // NOLINT
      : StatusOr(y_absl::in_place, std::forward<U>(u)) {}

  template <typename U = T,
            y_absl::enable_if_t<internal_statusor::IsConstructionValid<
                                  true, T, U, false>::value,
                              int> = 0>
  explicit StatusOr(U&& u)  // NOLINT
      : StatusOr(y_absl::in_place, std::forward<U>(u)) {}
  template <typename U = T,
            y_absl::enable_if_t<
                internal_statusor::IsConstructionValid<true, T, U, true>::value,
                int> = 0>
  explicit StatusOr(U&& u Y_ABSL_ATTRIBUTE_LIFETIME_BOUND)  // NOLINT
      : StatusOr(y_absl::in_place, std::forward<U>(u)) {}

  // StatusOr<T>::ok()
  //
  // Returns whether or not this `y_absl::StatusOr<T>` holds a `T` value. This
  // member function is analogous to `y_absl::Status::ok()` and should be used
  // similarly to check the status of return values.
  //
  // Example:
  //
  // StatusOr<Foo> result = DoBigCalculationThatCouldFail();
  // if (result.ok()) {
  //    // Handle result
  // else {
  //    // Handle error
  // }
  Y_ABSL_MUST_USE_RESULT bool ok() const { return this->status_.ok(); }

  // StatusOr<T>::status()
  //
  // Returns a reference to the current `y_absl::Status` contained within the
  // `y_absl::StatusOr<T>`. If `y_absl::StatusOr<T>` contains a `T`, then this
  // function returns `y_absl::OkStatus()`.
  const Status& status() const&;
  Status status() &&;

  // StatusOr<T>::value()
  //
  // Returns a reference to the held value if `this->ok()`. Otherwise, throws
  // `y_absl::BadStatusOrAccess` if exceptions are enabled, or is guaranteed to
  // terminate the process if exceptions are disabled.
  //
  // If you have already checked the status using `this->ok()`, you probably
  // want to use `operator*()` or `operator->()` to access the value instead of
  // `value`.
  //
  // Note: for value types that are cheap to copy, prefer simple code:
  //
  //   T value = statusor.value();
  //
  // Otherwise, if the value type is expensive to copy, but can be left
  // in the StatusOr, simply assign to a reference:
  //
  //   T& value = statusor.value();  // or `const T&`
  //
  // Otherwise, if the value type supports an efficient move, it can be
  // used as follows:
  //
  //   T value = std::move(statusor).value();
  //
  // The `std::move` on statusor instead of on the whole expression enables
  // warnings about possible uses of the statusor object after the move.
  const T& value() const& Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  T& value() & Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  const T&& value() const&& Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  T&& value() && Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // StatusOr<T>:: operator*()
  //
  // Returns a reference to the current value.
  //
  // REQUIRES: `this->ok() == true`, otherwise the behavior is undefined.
  //
  // Use `this->ok()` to verify that there is a current value within the
  // `y_absl::StatusOr<T>`. Alternatively, see the `value()` member function for a
  // similar API that guarantees crashing or throwing an exception if there is
  // no current value.
  const T& operator*() const& Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  T& operator*() & Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  const T&& operator*() const&& Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  T&& operator*() && Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // StatusOr<T>::operator->()
  //
  // Returns a pointer to the current value.
  //
  // REQUIRES: `this->ok() == true`, otherwise the behavior is undefined.
  //
  // Use `this->ok()` to verify that there is a current value.
  const T* operator->() const Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;
  T* operator->() Y_ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // StatusOr<T>::value_or()
  //
  // Returns the current value if `this->ok() == true`. Otherwise constructs a
  // value using the provided `default_value`.
  //
  // Unlike `value`, this function returns by value, copying the current value
  // if necessary. If the value type supports an efficient move, it can be used
  // as follows:
  //
  //   T value = std::move(statusor).value_or(def);
  //
  // Unlike with `value`, calling `std::move()` on the result of `value_or` will
  // still trigger a copy.
  template <typename U>
  T value_or(U&& default_value) const&;
  template <typename U>
  T value_or(U&& default_value) &&;

  // StatusOr<T>::IgnoreError()
  //
  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

  // StatusOr<T>::emplace()
  //
  // Reconstructs the inner value T in-place using the provided args, using the
  // T(args...) constructor. Returns reference to the reconstructed `T`.
  template <typename... Args>
  T& emplace(Args&&... args) Y_ABSL_ATTRIBUTE_LIFETIME_BOUND {
    if (ok()) {
      this->Clear();
      this->MakeValue(std::forward<Args>(args)...);
    } else {
      this->MakeValue(std::forward<Args>(args)...);
      this->status_ = y_absl::OkStatus();
    }
    return this->data_;
  }

  template <
      typename U, typename... Args,
      y_absl::enable_if_t<
          std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value,
          int> = 0>
  T& emplace(std::initializer_list<U> ilist,
             Args&&... args) Y_ABSL_ATTRIBUTE_LIFETIME_BOUND {
    if (ok()) {
      this->Clear();
      this->MakeValue(ilist, std::forward<Args>(args)...);
    } else {
      this->MakeValue(ilist, std::forward<Args>(args)...);
      this->status_ = y_absl::OkStatus();
    }
    return this->data_;
  }

  // StatusOr<T>::AssignStatus()
  //
  // Sets the status of `y_absl::StatusOr<T>` to the given non-ok status value.
  //
  // NOTE: We recommend using the constructor and `operator=` where possible.
  // This method is intended for use in generic programming, to enable setting
  // the status of a `StatusOr<T>` when `T` may be `Status`. In that case, the
  // constructor and `operator=` would assign into the inner value of type
  // `Status`, rather than status of the `StatusOr` (b/280392796).
  //
  // REQUIRES: !Status(std::forward<U>(v)).ok(). This requirement is DCHECKed.
  // In optimized builds, passing y_absl::OkStatus() here will have the effect
  // of passing y_absl::StatusCode::kInternal as a fallback.
  using internal_statusor::StatusOrData<T>::AssignStatus;

 private:
  using internal_statusor::StatusOrData<T>::Assign;
  template <typename U>
  void Assign(const y_absl::StatusOr<U>& other);
  template <typename U>
  void Assign(y_absl::StatusOr<U>&& other);
};

// operator==()
//
// This operator checks the equality of two `y_absl::StatusOr<T>` objects.
template <typename T>
bool operator==(const StatusOr<T>& lhs, const StatusOr<T>& rhs) {
  if (lhs.ok() && rhs.ok()) return *lhs == *rhs;
  return lhs.status() == rhs.status();
}

// operator!=()
//
// This operator checks the inequality of two `y_absl::StatusOr<T>` objects.
template <typename T>
bool operator!=(const StatusOr<T>& lhs, const StatusOr<T>& rhs) {
  return !(lhs == rhs);
}

// Prints the `value` or the status in brackets to `os`.
//
// Requires `T` supports `operator<<`.  Do not rely on the output format which
// may change without notice.
template <typename T, typename std::enable_if<
                          y_absl::HasOstreamOperator<T>::value, int>::type = 0>
std::ostream& operator<<(std::ostream& os, const StatusOr<T>& status_or) {
  if (status_or.ok()) {
    os << status_or.value();
  } else {
    os << internal_statusor::StringifyRandom::OpenBrackets()
       << status_or.status()
       << internal_statusor::StringifyRandom::CloseBrackets();
  }
  return os;
}

// As above, but supports `StrCat`, `StrFormat`, etc.
//
// Requires `T` has `AbslStringify`.  Do not rely on the output format which
// may change without notice.
template <
    typename Sink, typename T,
    typename std::enable_if<y_absl::HasAbslStringify<T>::value, int>::type = 0>
void AbslStringify(Sink& sink, const StatusOr<T>& status_or) {
  if (status_or.ok()) {
    y_absl::Format(&sink, "%v", status_or.value());
  } else {
    y_absl::Format(&sink, "%s%v%s",
                 internal_statusor::StringifyRandom::OpenBrackets(),
                 status_or.status(),
                 internal_statusor::StringifyRandom::CloseBrackets());
  }
}

//------------------------------------------------------------------------------
// Implementation details for StatusOr<T>
//------------------------------------------------------------------------------

// TODO(sbenza): avoid the string here completely.
template <typename T>
StatusOr<T>::StatusOr() : Base(Status(y_absl::StatusCode::kUnknown, "")) {}

template <typename T>
template <typename U>
inline void StatusOr<T>::Assign(const StatusOr<U>& other) {
  if (other.ok()) {
    this->Assign(*other);
  } else {
    this->AssignStatus(other.status());
  }
}

template <typename T>
template <typename U>
inline void StatusOr<T>::Assign(StatusOr<U>&& other) {
  if (other.ok()) {
    this->Assign(*std::move(other));
  } else {
    this->AssignStatus(std::move(other).status());
  }
}
template <typename T>
template <typename... Args>
StatusOr<T>::StatusOr(y_absl::in_place_t, Args&&... args)
    : Base(y_absl::in_place, std::forward<Args>(args)...) {}

template <typename T>
template <typename U, typename... Args>
StatusOr<T>::StatusOr(y_absl::in_place_t, std::initializer_list<U> ilist,
                      Args&&... args)
    : Base(y_absl::in_place, ilist, std::forward<Args>(args)...) {}

template <typename T>
const Status& StatusOr<T>::status() const& {
  return this->status_;
}
template <typename T>
Status StatusOr<T>::status() && {
  return ok() ? OkStatus() : std::move(this->status_);
}

template <typename T>
const T& StatusOr<T>::value() const& {
  if (!this->ok()) internal_statusor::ThrowBadStatusOrAccess(this->status_);
  return this->data_;
}

template <typename T>
T& StatusOr<T>::value() & {
  if (!this->ok()) internal_statusor::ThrowBadStatusOrAccess(this->status_);
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::value() const&& {
  if (!this->ok()) {
    internal_statusor::ThrowBadStatusOrAccess(std::move(this->status_));
  }
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::value() && {
  if (!this->ok()) {
    internal_statusor::ThrowBadStatusOrAccess(std::move(this->status_));
  }
  return std::move(this->data_);
}

template <typename T>
const T& StatusOr<T>::operator*() const& {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
T& StatusOr<T>::operator*() & {
  this->EnsureOk();
  return this->data_;
}

template <typename T>
const T&& StatusOr<T>::operator*() const&& {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
T&& StatusOr<T>::operator*() && {
  this->EnsureOk();
  return std::move(this->data_);
}

template <typename T>
y_absl::Nonnull<const T*> StatusOr<T>::operator->() const {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
y_absl::Nonnull<T*> StatusOr<T>::operator->() {
  this->EnsureOk();
  return &this->data_;
}

template <typename T>
template <typename U>
T StatusOr<T>::value_or(U&& default_value) const& {
  if (ok()) {
    return this->data_;
  }
  return std::forward<U>(default_value);
}

template <typename T>
template <typename U>
T StatusOr<T>::value_or(U&& default_value) && {
  if (ok()) {
    return std::move(this->data_);
  }
  return std::forward<U>(default_value);
}

template <typename T>
void StatusOr<T>::IgnoreError() const {
  // no-op
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STATUS_STATUSOR_H_
