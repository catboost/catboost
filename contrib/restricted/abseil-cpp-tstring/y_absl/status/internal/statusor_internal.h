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
#ifndef Y_ABSL_STATUS_INTERNAL_STATUSOR_INTERNAL_H_
#define Y_ABSL_STATUS_INTERNAL_STATUSOR_INTERNAL_H_

#include <cstdint>
#include <type_traits>
#include <utility>

#include "y_absl/base/attributes.h"
#include "y_absl/base/nullability.h"
#include "y_absl/meta/type_traits.h"
#include "y_absl/status/status.h"
#include "y_absl/strings/string_view.h"
#include "y_absl/utility/utility.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

template <typename T>
class Y_ABSL_MUST_USE_RESULT StatusOr;

namespace internal_statusor {

// Detects whether `U` has conversion operator to `StatusOr<T>`, i.e. `operator
// StatusOr<T>()`.
template <typename T, typename U, typename = void>
struct HasConversionOperatorToStatusOr : std::false_type {};

template <typename T, typename U>
void test(char (*)[sizeof(std::declval<U>().operator y_absl::StatusOr<T>())]);

template <typename T, typename U>
struct HasConversionOperatorToStatusOr<T, U, decltype(test<T, U>(0))>
    : std::true_type {};

// Detects whether `T` is constructible or convertible from `StatusOr<U>`.
template <typename T, typename U>
using IsConstructibleOrConvertibleFromStatusOr =
    y_absl::disjunction<std::is_constructible<T, StatusOr<U>&>,
                      std::is_constructible<T, const StatusOr<U>&>,
                      std::is_constructible<T, StatusOr<U>&&>,
                      std::is_constructible<T, const StatusOr<U>&&>,
                      std::is_convertible<StatusOr<U>&, T>,
                      std::is_convertible<const StatusOr<U>&, T>,
                      std::is_convertible<StatusOr<U>&&, T>,
                      std::is_convertible<const StatusOr<U>&&, T>>;

// Detects whether `T` is constructible or convertible or assignable from
// `StatusOr<U>`.
template <typename T, typename U>
using IsConstructibleOrConvertibleOrAssignableFromStatusOr =
    y_absl::disjunction<IsConstructibleOrConvertibleFromStatusOr<T, U>,
                      std::is_assignable<T&, StatusOr<U>&>,
                      std::is_assignable<T&, const StatusOr<U>&>,
                      std::is_assignable<T&, StatusOr<U>&&>,
                      std::is_assignable<T&, const StatusOr<U>&&>>;

// Detects whether direct initializing `StatusOr<T>` from `U` is ambiguous, i.e.
// when `U` is `StatusOr<V>` and `T` is constructible or convertible from `V`.
template <typename T, typename U>
struct IsDirectInitializationAmbiguous
    : public y_absl::conditional_t<
          std::is_same<y_absl::remove_cvref_t<U>, U>::value, std::false_type,
          IsDirectInitializationAmbiguous<T, y_absl::remove_cvref_t<U>>> {};

template <typename T, typename V>
struct IsDirectInitializationAmbiguous<T, y_absl::StatusOr<V>>
    : public IsConstructibleOrConvertibleFromStatusOr<T, V> {};

// Checks against the constraints of the direction initialization, i.e. when
// `StatusOr<T>::StatusOr(U&&)` should participate in overload resolution.
template <typename T, typename U>
using IsDirectInitializationValid = y_absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, y_absl::remove_cvref_t<U>>,
    y_absl::negation<y_absl::disjunction<
        std::is_same<y_absl::StatusOr<T>, y_absl::remove_cvref_t<U>>,
        std::is_same<y_absl::Status, y_absl::remove_cvref_t<U>>,
        std::is_same<y_absl::in_place_t, y_absl::remove_cvref_t<U>>,
        IsDirectInitializationAmbiguous<T, U>>>>;

// This trait detects whether `StatusOr<T>::operator=(U&&)` is ambiguous, which
// is equivalent to whether all the following conditions are met:
// 1. `U` is `StatusOr<V>`.
// 2. `T` is constructible and assignable from `V`.
// 3. `T` is constructible and assignable from `U` (i.e. `StatusOr<V>`).
// For example, the following code is considered ambiguous:
// (`T` is `bool`, `U` is `StatusOr<bool>`, `V` is `bool`)
//   StatusOr<bool> s1 = true;  // s1.ok() && s1.ValueOrDie() == true
//   StatusOr<bool> s2 = false;  // s2.ok() && s2.ValueOrDie() == false
//   s1 = s2;  // ambiguous, `s1 = s2.ValueOrDie()` or `s1 = bool(s2)`?
template <typename T, typename U>
struct IsForwardingAssignmentAmbiguous
    : public y_absl::conditional_t<
          std::is_same<y_absl::remove_cvref_t<U>, U>::value, std::false_type,
          IsForwardingAssignmentAmbiguous<T, y_absl::remove_cvref_t<U>>> {};

template <typename T, typename U>
struct IsForwardingAssignmentAmbiguous<T, y_absl::StatusOr<U>>
    : public IsConstructibleOrConvertibleOrAssignableFromStatusOr<T, U> {};

// Checks against the constraints of the forwarding assignment, i.e. whether
// `StatusOr<T>::operator(U&&)` should participate in overload resolution.
template <typename T, typename U>
using IsForwardingAssignmentValid = y_absl::disjunction<
    // Short circuits if T is basically U.
    std::is_same<T, y_absl::remove_cvref_t<U>>,
    y_absl::negation<y_absl::disjunction<
        std::is_same<y_absl::StatusOr<T>, y_absl::remove_cvref_t<U>>,
        std::is_same<y_absl::Status, y_absl::remove_cvref_t<U>>,
        std::is_same<y_absl::in_place_t, y_absl::remove_cvref_t<U>>,
        IsForwardingAssignmentAmbiguous<T, U>>>>;

template <bool Value, typename T>
using Equality = std::conditional_t<Value, T, y_absl::negation<T>>;

template <bool Explicit, typename T, typename U, bool Lifetimebound>
using IsConstructionValid = y_absl::conjunction<
    Equality<Lifetimebound,
             type_traits_internal::IsLifetimeBoundAssignment<T, U>>,
    IsDirectInitializationValid<T, U&&>, std::is_constructible<T, U&&>,
    Equality<!Explicit, std::is_convertible<U&&, T>>,
    y_absl::disjunction<
        std::is_same<T, y_absl::remove_cvref_t<U>>,
        y_absl::conjunction<
            std::conditional_t<
                Explicit,
                y_absl::negation<std::is_constructible<y_absl::Status, U&&>>,
                y_absl::negation<std::is_convertible<U&&, y_absl::Status>>>,
            y_absl::negation<
                internal_statusor::HasConversionOperatorToStatusOr<T, U&&>>>>>;

template <typename T, typename U, bool Lifetimebound>
using IsAssignmentValid = y_absl::conjunction<
    Equality<Lifetimebound,
             type_traits_internal::IsLifetimeBoundAssignment<T, U>>,
    std::is_constructible<T, U&&>, std::is_assignable<T&, U&&>,
    y_absl::disjunction<
        std::is_same<T, y_absl::remove_cvref_t<U>>,
        y_absl::conjunction<
            y_absl::negation<std::is_convertible<U&&, y_absl::Status>>,
            y_absl::negation<HasConversionOperatorToStatusOr<T, U&&>>>>,
    IsForwardingAssignmentValid<T, U&&>>;

template <bool Explicit, typename T, typename U>
using IsConstructionFromStatusValid = y_absl::conjunction<
    y_absl::negation<std::is_same<y_absl::StatusOr<T>, y_absl::remove_cvref_t<U>>>,
    y_absl::negation<std::is_same<T, y_absl::remove_cvref_t<U>>>,
    y_absl::negation<std::is_same<y_absl::in_place_t, y_absl::remove_cvref_t<U>>>,
    Equality<!Explicit, std::is_convertible<U, y_absl::Status>>,
    std::is_constructible<y_absl::Status, U>,
    y_absl::negation<HasConversionOperatorToStatusOr<T, U>>>;

template <bool Explicit, typename T, typename U, bool Lifetimebound,
          typename UQ>
using IsConstructionFromStatusOrValid = y_absl::conjunction<
    y_absl::negation<std::is_same<T, U>>,
    Equality<Lifetimebound,
             type_traits_internal::IsLifetimeBoundAssignment<T, U>>,
    std::is_constructible<T, UQ>,
    Equality<!Explicit, std::is_convertible<UQ, T>>,
    y_absl::negation<IsConstructibleOrConvertibleFromStatusOr<T, U>>>;

template <typename T, typename U, bool Lifetimebound>
using IsStatusOrAssignmentValid = y_absl::conjunction<
    y_absl::negation<std::is_same<T, y_absl::remove_cvref_t<U>>>,
    Equality<Lifetimebound,
             type_traits_internal::IsLifetimeBoundAssignment<T, U>>,
    std::is_constructible<T, U>, std::is_assignable<T, U>,
    y_absl::negation<IsConstructibleOrConvertibleOrAssignableFromStatusOr<
        T, y_absl::remove_cvref_t<U>>>>;

class Helper {
 public:
  // Move type-agnostic error handling to the .cc.
  static void HandleInvalidStatusCtorArg(y_absl::Nonnull<Status*>);
  [[noreturn]] static void Crash(const y_absl::Status& status);
};

// Construct an instance of T in `p` through placement new, passing Args... to
// the constructor.
// This abstraction is here mostly for the gcc performance fix.
template <typename T, typename... Args>
Y_ABSL_ATTRIBUTE_NONNULL(1)
void PlacementNew(y_absl::Nonnull<void*> p, Args&&... args) {
  new (p) T(std::forward<Args>(args)...);
}

// Helper base class to hold the data and all operations.
// We move all this to a base class to allow mixing with the appropriate
// TraitsBase specialization.
template <typename T>
class StatusOrData {
  template <typename U>
  friend class StatusOrData;

 public:
  StatusOrData() = delete;

  StatusOrData(const StatusOrData& other) {
    if (other.ok()) {
      MakeValue(other.data_);
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  StatusOrData(StatusOrData&& other) noexcept {
    if (other.ok()) {
      MakeValue(std::move(other.data_));
      MakeStatus();
    } else {
      MakeStatus(std::move(other.status_));
    }
  }

  template <typename U>
  explicit StatusOrData(const StatusOrData<U>& other) {
    if (other.ok()) {
      MakeValue(other.data_);
      MakeStatus();
    } else {
      MakeStatus(other.status_);
    }
  }

  template <typename U>
  explicit StatusOrData(StatusOrData<U>&& other) {
    if (other.ok()) {
      MakeValue(std::move(other.data_));
      MakeStatus();
    } else {
      MakeStatus(std::move(other.status_));
    }
  }

  template <typename... Args>
  explicit StatusOrData(y_absl::in_place_t, Args&&... args)
      : data_(std::forward<Args>(args)...) {
    MakeStatus();
  }

  explicit StatusOrData(const T& value) : data_(value) {
    MakeStatus();
  }
  explicit StatusOrData(T&& value) : data_(std::move(value)) {
    MakeStatus();
  }

  template <typename U,
            y_absl::enable_if_t<std::is_constructible<y_absl::Status, U&&>::value,
                              int> = 0>
  explicit StatusOrData(U&& v) : status_(std::forward<U>(v)) {
    EnsureNotOk();
  }

  StatusOrData& operator=(const StatusOrData& other) {
    if (this == &other) return *this;
    if (other.ok())
      Assign(other.data_);
    else
      AssignStatus(other.status_);
    return *this;
  }

  StatusOrData& operator=(StatusOrData&& other) {
    if (this == &other) return *this;
    if (other.ok())
      Assign(std::move(other.data_));
    else
      AssignStatus(std::move(other.status_));
    return *this;
  }

  ~StatusOrData() {
    if (ok()) {
      status_.~Status();
      data_.~T();
    } else {
      status_.~Status();
    }
  }

  template <typename U>
  void Assign(U&& value) {
    if (ok()) {
      data_ = std::forward<U>(value);
    } else {
      MakeValue(std::forward<U>(value));
      status_ = OkStatus();
    }
  }

  template <typename U>
  void AssignStatus(U&& v) {
    Clear();
    status_ = static_cast<y_absl::Status>(std::forward<U>(v));
    EnsureNotOk();
  }

  bool ok() const { return status_.ok(); }

 protected:
  // status_ will always be active after the constructor.
  // We make it a union to be able to initialize exactly how we need without
  // waste.
  // Eg. in the copy constructor we use the default constructor of Status in
  // the ok() path to avoid an extra Ref call.
  union {
    Status status_;
  };

  // data_ is active iff status_.ok()==true
  struct Dummy {};
  union {
    // When T is const, we need some non-const object we can cast to void* for
    // the placement new. dummy_ is that object.
    Dummy dummy_;
    T data_;
  };

  void Clear() {
    if (ok()) data_.~T();
  }

  void EnsureOk() const {
    if (Y_ABSL_PREDICT_FALSE(!ok())) Helper::Crash(status_);
  }

  void EnsureNotOk() {
    if (Y_ABSL_PREDICT_FALSE(ok())) Helper::HandleInvalidStatusCtorArg(&status_);
  }

  // Construct the value (ie. data_) through placement new with the passed
  // argument.
  template <typename... Arg>
  void MakeValue(Arg&&... arg) {
    internal_statusor::PlacementNew<T>(&dummy_, std::forward<Arg>(arg)...);
  }

  // Construct the status (ie. status_) through placement new with the passed
  // argument.
  template <typename... Args>
  void MakeStatus(Args&&... args) {
    internal_statusor::PlacementNew<Status>(&status_,
                                            std::forward<Args>(args)...);
  }
};

// Helper base classes to allow implicitly deleted constructors and assignment
// operators in `StatusOr`. For example, `CopyCtorBase` will explicitly delete
// the copy constructor when T is not copy constructible and `StatusOr` will
// inherit that behavior implicitly.
template <typename T, bool = std::is_copy_constructible<T>::value>
struct CopyCtorBase {
  CopyCtorBase() = default;
  CopyCtorBase(const CopyCtorBase&) = default;
  CopyCtorBase(CopyCtorBase&&) = default;
  CopyCtorBase& operator=(const CopyCtorBase&) = default;
  CopyCtorBase& operator=(CopyCtorBase&&) = default;
};

template <typename T>
struct CopyCtorBase<T, false> {
  CopyCtorBase() = default;
  CopyCtorBase(const CopyCtorBase&) = delete;
  CopyCtorBase(CopyCtorBase&&) = default;
  CopyCtorBase& operator=(const CopyCtorBase&) = default;
  CopyCtorBase& operator=(CopyCtorBase&&) = default;
};

template <typename T, bool = std::is_move_constructible<T>::value>
struct MoveCtorBase {
  MoveCtorBase() = default;
  MoveCtorBase(const MoveCtorBase&) = default;
  MoveCtorBase(MoveCtorBase&&) = default;
  MoveCtorBase& operator=(const MoveCtorBase&) = default;
  MoveCtorBase& operator=(MoveCtorBase&&) = default;
};

template <typename T>
struct MoveCtorBase<T, false> {
  MoveCtorBase() = default;
  MoveCtorBase(const MoveCtorBase&) = default;
  MoveCtorBase(MoveCtorBase&&) = delete;
  MoveCtorBase& operator=(const MoveCtorBase&) = default;
  MoveCtorBase& operator=(MoveCtorBase&&) = default;
};

template <typename T, bool = std::is_copy_constructible<T>::value&&
                          std::is_copy_assignable<T>::value>
struct CopyAssignBase {
  CopyAssignBase() = default;
  CopyAssignBase(const CopyAssignBase&) = default;
  CopyAssignBase(CopyAssignBase&&) = default;
  CopyAssignBase& operator=(const CopyAssignBase&) = default;
  CopyAssignBase& operator=(CopyAssignBase&&) = default;
};

template <typename T>
struct CopyAssignBase<T, false> {
  CopyAssignBase() = default;
  CopyAssignBase(const CopyAssignBase&) = default;
  CopyAssignBase(CopyAssignBase&&) = default;
  CopyAssignBase& operator=(const CopyAssignBase&) = delete;
  CopyAssignBase& operator=(CopyAssignBase&&) = default;
};

template <typename T, bool = std::is_move_constructible<T>::value&&
                          std::is_move_assignable<T>::value>
struct MoveAssignBase {
  MoveAssignBase() = default;
  MoveAssignBase(const MoveAssignBase&) = default;
  MoveAssignBase(MoveAssignBase&&) = default;
  MoveAssignBase& operator=(const MoveAssignBase&) = default;
  MoveAssignBase& operator=(MoveAssignBase&&) = default;
};

template <typename T>
struct MoveAssignBase<T, false> {
  MoveAssignBase() = default;
  MoveAssignBase(const MoveAssignBase&) = default;
  MoveAssignBase(MoveAssignBase&&) = default;
  MoveAssignBase& operator=(const MoveAssignBase&) = default;
  MoveAssignBase& operator=(MoveAssignBase&&) = delete;
};

[[noreturn]] void ThrowBadStatusOrAccess(y_absl::Status status);

// Used to introduce jitter into the output of printing functions for
// `StatusOr` (i.e. `AbslStringify` and `operator<<`).
class StringifyRandom {
  enum BracesType {
    kBareParens = 0,
    kSpaceParens,
    kBareBrackets,
    kSpaceBrackets,
  };

  // Returns a random `BracesType` determined once per binary load.
  static BracesType RandomBraces() {
    static const BracesType kRandomBraces = static_cast<BracesType>(
        (reinterpret_cast<uintptr_t>(&kRandomBraces) >> 4) % 4);
    return kRandomBraces;
  }

 public:
  static inline y_absl::string_view OpenBrackets() {
    switch (RandomBraces()) {
      case kBareParens:
        return "(";
      case kSpaceParens:
        return "( ";
      case kBareBrackets:
        return "[";
      case kSpaceBrackets:
        return "[ ";
    }
    return "(";
  }

  static inline y_absl::string_view CloseBrackets() {
    switch (RandomBraces()) {
      case kBareParens:
        return ")";
      case kSpaceParens:
        return " )";
      case kBareBrackets:
        return "]";
      case kSpaceBrackets:
        return " ]";
    }
    return ")";
  }
};

}  // namespace internal_statusor
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_STATUS_INTERNAL_STATUSOR_INTERNAL_H_
