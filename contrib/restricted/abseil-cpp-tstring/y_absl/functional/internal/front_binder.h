// Copyright 2018 The Abseil Authors.
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

// Implementation details for `y_absl::bind_front()`.

#ifndef Y_ABSL_FUNCTIONAL_INTERNAL_FRONT_BINDER_H_
#define Y_ABSL_FUNCTIONAL_INTERNAL_FRONT_BINDER_H_

#include <cstddef>
#include <type_traits>
#include <utility>

#include "y_absl/base/internal/invoke.h"
#include "y_absl/container/internal/compressed_tuple.h"
#include "y_absl/meta/type_traits.h"
#include "y_absl/utility/utility.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN
namespace functional_internal {

// Invoke the method, expanding the tuple of bound arguments.
template <class R, class Tuple, size_t... Idx, class... Args>
R Apply(Tuple&& bound, y_absl::index_sequence<Idx...>, Args&&... free) {
  return base_internal::invoke(
      y_absl::forward<Tuple>(bound).template get<Idx>()...,
      y_absl::forward<Args>(free)...);
}

template <class F, class... BoundArgs>
class FrontBinder {
  using BoundArgsT = y_absl::container_internal::CompressedTuple<F, BoundArgs...>;
  using Idx = y_absl::make_index_sequence<sizeof...(BoundArgs) + 1>;

  BoundArgsT bound_args_;

 public:
  template <class... Ts>
  constexpr explicit FrontBinder(y_absl::in_place_t, Ts&&... ts)
      : bound_args_(y_absl::forward<Ts>(ts)...) {}

  template <class... FreeArgs, class R = base_internal::invoke_result_t<
                                   F&, BoundArgs&..., FreeArgs&&...>>
  R operator()(FreeArgs&&... free_args) & {
    return functional_internal::Apply<R>(bound_args_, Idx(),
                                         y_absl::forward<FreeArgs>(free_args)...);
  }

  template <class... FreeArgs,
            class R = base_internal::invoke_result_t<
                const F&, const BoundArgs&..., FreeArgs&&...>>
  R operator()(FreeArgs&&... free_args) const& {
    return functional_internal::Apply<R>(bound_args_, Idx(),
                                         y_absl::forward<FreeArgs>(free_args)...);
  }

  template <class... FreeArgs, class R = base_internal::invoke_result_t<
                                   F&&, BoundArgs&&..., FreeArgs&&...>>
  R operator()(FreeArgs&&... free_args) && {
    // This overload is called when *this is an rvalue. If some of the bound
    // arguments are stored by value or rvalue reference, we move them.
    return functional_internal::Apply<R>(y_absl::move(bound_args_), Idx(),
                                         y_absl::forward<FreeArgs>(free_args)...);
  }

  template <class... FreeArgs,
            class R = base_internal::invoke_result_t<
                const F&&, const BoundArgs&&..., FreeArgs&&...>>
  R operator()(FreeArgs&&... free_args) const&& {
    // This overload is called when *this is an rvalue. If some of the bound
    // arguments are stored by value or rvalue reference, we move them.
    return functional_internal::Apply<R>(y_absl::move(bound_args_), Idx(),
                                         y_absl::forward<FreeArgs>(free_args)...);
  }
};

template <class F, class... BoundArgs>
using bind_front_t = FrontBinder<decay_t<F>, y_absl::decay_t<BoundArgs>...>;

}  // namespace functional_internal
Y_ABSL_NAMESPACE_END
}  // namespace y_absl

#endif  // Y_ABSL_FUNCTIONAL_INTERNAL_FRONT_BINDER_H_
