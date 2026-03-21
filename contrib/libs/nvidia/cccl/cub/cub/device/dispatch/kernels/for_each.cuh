// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_for.cuh>
#include <cub/detail/fast_modulo_division.cuh> // fast_div_mod
#include <cub/detail/mdspan_utils.cuh> // is_sub_size_static
#include <cub/detail/type_traits.cuh> // implicit_prom_t

#include <cuda/std/cstddef> // size_t
#include <cuda/std/mdspan> // dynamic_extent
#include <cuda/std/type_traits>
#include <cuda/std/utility> // make_index_sequence

CUB_NAMESPACE_BEGIN

namespace detail::for_each
{

template <class Fn>
struct first_parameter
{
  using type = void;
};

template <class C, class R, class A>
struct first_parameter<R (C::*)(A)>
{
  using type = A;
};

template <class C, class R, class A>
struct first_parameter<R (C::*)(A) const>
{
  using type = A;
};

template <class Fn>
using first_parameter_t = typename first_parameter<decltype(&Fn::operator())>::type;

template <class Value, class Fn, class = void>
struct has_unique_value_overload : _CUDA_VSTD::false_type
{};

// clang-format off
template <class Value, class Fn>
struct has_unique_value_overload<
  Value,
  Fn,
  _CUDA_VSTD::enable_if_t<
              !_CUDA_VSTD::is_reference_v<first_parameter_t<Fn>> &&
              _CUDA_VSTD::is_convertible_v<Value, first_parameter_t<Fn>
             >>>
    : _CUDA_VSTD::true_type
{};

// For trivial types, foreach is not allowed to copy values, even if those are trivially copyable.
// This can be observable if the unary operator takes parameter by reference and modifies it or uses address.
// The trait below checks if the freedom to copy trivial types can be regained.
template <typename Value, typename Fn>
using can_regain_copy_freedom =
  _CUDA_VSTD::integral_constant<
    bool,
    _CUDA_VSTD::is_trivially_constructible_v<Value> &&
    _CUDA_VSTD::is_trivially_copy_assignable_v<Value> &&
    _CUDA_VSTD::is_trivially_move_assignable_v<Value> &&
    _CUDA_VSTD::is_trivially_destructible_v<Value> &&
    has_unique_value_overload<Value, Fn>::value>;
// clang-format on

// This kernel is used when the block size is not known at compile time
template <class ChainedPolicyT, class OffsetT, class OpT>
CUB_DETAIL_KERNEL_ATTRIBUTES void dynamic_kernel(OffsetT num_items, OpT op)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using agent_t         = agent_block_striped_t<active_policy_t, OffsetT, OpT>;

  const auto block_threads  = static_cast<OffsetT>(blockDim.x);
  const auto items_per_tile = active_policy_t::items_per_thread * block_threads;
  const auto tile_base      = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining  = num_items - tile_base;
  const auto items_in_tile  = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {
    agent_t{tile_base, op}.template consume_tile<true>(items_per_tile, block_threads);
  }
  else
  {
    agent_t{tile_base, op}.template consume_tile<false>(items_in_tile, block_threads);
  }
}

// This kernel is used when the block size is known at compile time
template <class ChainedPolicyT, class OffsetT, class OpT>
CUB_DETAIL_KERNEL_ATTRIBUTES //
__launch_bounds__(ChainedPolicyT::ActivePolicy::for_policy_t::block_threads) //
  void static_kernel(OffsetT num_items, OpT op)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy::for_policy_t;
  using agent_t         = agent_block_striped_t<active_policy_t, OffsetT, OpT>;

  constexpr auto block_threads  = active_policy_t::block_threads;
  constexpr auto items_per_tile = active_policy_t::items_per_thread * block_threads;

  const auto tile_base     = static_cast<OffsetT>(blockIdx.x) * items_per_tile;
  const auto num_remaining = num_items - tile_base;
  const auto items_in_tile = static_cast<OffsetT>(num_remaining < items_per_tile ? num_remaining : items_per_tile);

  if (items_in_tile == items_per_tile)
  {
    agent_t{tile_base, op}.template consume_tile<true>(items_per_tile, block_threads);
  }
  else
  {
    agent_t{tile_base, op}.template consume_tile<false>(items_in_tile, block_threads);
  }
}

/***********************************************************************************************************************
 * ForEachInExtents
 **********************************************************************************************************************/

// Returns the extent at the given rank. If the extents is static, returns it, otherwise returns the precomputed value
template <int Rank, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto extent_at(ExtentType extents, FastDivModType dynamic_extent)
{
  if constexpr (ExtentType::static_extent(Rank) != _CUDA_VSTD::dynamic_extent)
  {
    using extent_index_type   = typename ExtentType::index_type;
    using index_type          = implicit_prom_t<extent_index_type>;
    using unsigned_index_type = _CUDA_VSTD::make_unsigned_t<index_type>;
    return static_cast<unsigned_index_type>(extents.static_extent(Rank));
  }
  else
  {
    return dynamic_extent;
  }
}

// Returns the product of all extents from position Rank. If the result is static, returns it, otherwise returns the
// precomputed value
template <int Rank, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto get_extents_sub_size(ExtentType extents, FastDivModType extent_sub_size)
{
  if constexpr (cub::detail::is_sub_size_static<Rank + 1, ExtentType>())
  {
    using extent_index_type   = typename ExtentType::index_type;
    using index_type          = implicit_prom_t<extent_index_type>;
    using unsigned_index_type = _CUDA_VSTD::make_unsigned_t<index_type>;
    return static_cast<unsigned_index_type>(cub::detail::sub_size<Rank + 1>(extents));
  }
  else
  {
    return extent_sub_size;
  }
}

template <int Rank, typename IndexType, typename ExtentType, typename FastDivModType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto
coordinate_at(IndexType index, ExtentType extents, FastDivModType extent_sub_size, FastDivModType dynamic_extent)
{
  using cub::detail::for_each::extent_at;
  using cub::detail::for_each::get_extents_sub_size;
  using extent_index_type = typename ExtentType::index_type;
  return static_cast<extent_index_type>(
    (index / get_extents_sub_size<Rank>(extents, extent_sub_size)) % extent_at<Rank>(extents, dynamic_extent));
}

template <typename OpT, typename ExtentsT, typename FastDivModArrayT>
struct op_wrapper_extents_t
{
  OpT op;
  ExtentsT extents;
  FastDivModArrayT sub_sizes_div_array;
  FastDivModArrayT extents_mod_array;

  template <typename OffsetT, size_t... Ranks>
  _CCCL_DEVICE _CCCL_FORCEINLINE void impl(OffsetT i, _CUDA_VSTD::index_sequence<Ranks...>)
  {
    using cub::detail::for_each::coordinate_at;
    op(i, coordinate_at<Ranks>(i, extents, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }

  template <typename OffsetT, size_t... Ranks>
  _CCCL_DEVICE _CCCL_FORCEINLINE void impl(OffsetT i, _CUDA_VSTD::index_sequence<Ranks...>) const
  {
    using cub::detail::for_each::coordinate_at;
    op(i, coordinate_at<Ranks>(i, extents, sub_sizes_div_array[Ranks], extents_mod_array[Ranks])...);
  }

  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(OffsetT i)
  {
    impl(i, _CUDA_VSTD::make_index_sequence<ExtentsT::rank()>{});
  }

  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(OffsetT i) const
  {
    impl(i, _CUDA_VSTD::make_index_sequence<ExtentsT::rank()>{});
  }
};

} // namespace detail::for_each

CUB_NAMESPACE_END
