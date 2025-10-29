/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/climits>
#include <cuda/std/cmath>

CUB_NAMESPACE_BEGIN

namespace detail::rfa
{

// jump table for indexing into data
inline constexpr int cub_rfa_max_jump = 5;
static_assert(cub_rfa_max_jump <= 5, "cub_rfa_max_jump must be less than or equal to 5");

template <typename FType, int Len>
static _CCCL_DEVICE FType* get_shared_bin_array()
{
  static __shared__ FType bin_computed_array[Len];
  return bin_computed_array;
}

//! Class to hold a reproducible summation of the numbers passed to it
//!
//! @param FType Floating-point data type; either `float` or `double
//! @param Fold  Number of collectors in the binned number (K-fold), used for reproducible summation. Defaults to 3.
template <class FType, int Fold = 3, ::cuda::std::enable_if_t<::cuda::std::is_floating_point_v<FType>>* = nullptr>
class alignas(2 * sizeof(FType)) ReproducibleFloatingAccumulator
{
public:
  using ftype = FType;

private:
  ::cuda::std::array<ftype, 2 * Fold> data{};

  /// Floating-point precision bin width
  static constexpr int bin_width = ::cuda::std::is_same_v<ftype, double> ? 40 : 13;
  static constexpr int min_exp   = ::cuda::std::numeric_limits<ftype>::min_exponent;
  static constexpr int max_exp   = ::cuda::std::numeric_limits<ftype>::max_exponent;
  static constexpr int mant_dig  = ::cuda::std::numeric_limits<ftype>::digits;

public:
  /// Binned floating-point maximum index
  static constexpr int max_index = ((max_exp - min_exp + mant_dig - 1) / bin_width) - 1;

  // The maximum floating-point fold supported by the library
  static constexpr auto max_fold = max_index + 1;

  _CCCL_DEVICE static ftype initialize_bin(int index) noexcept
  {
    if (index == 0)
    {
      if constexpr (::cuda::std::is_same_v<ftype, float>)
      {
        return ::cuda::std::ldexp(0.75, max_exp);
      }
      else
      {
        return 2.0 * ::cuda::std::ldexp(0.75, max_exp - 1);
      }
    }

    if (index > 0 && index <= max_index)
    {
      return ::cuda::std::ldexp(0.75, max_exp + mant_dig - bin_width + 1 - index * bin_width);
    }
    else
    {
      return ::cuda::std::ldexp(0.75, max_exp + mant_dig - bin_width + 1 - max_index * bin_width);
    }
  }

private:
  /// Binned floating-point compression factor
  /// This factor is used to scale down inputs before deposition into the bin of
  /// highest index
  static constexpr auto compression = 1.0 / (1 << (mant_dig - bin_width + 1));
  /// Binned double precision expansion factor
  /// This factor is used to scale up inputs after deposition into the bin of
  /// highest index
  static constexpr auto expansion = 1.0 * (1 << (mant_dig - bin_width + 1));
  static constexpr auto exp_bias  = max_exp - 2;

  /// Return a binned floating-point bin
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static ftype binned_bins(int index)
  {
    ftype* bins = get_shared_bin_array<ftype, max_index + max_fold>();
    return bins[index];
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static uint32_t& get_bit_representation(float& x) noexcept
  {
    return *reinterpret_cast<uint32_t*>(&x);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static uint64_t& get_bit_representation(double& x) noexcept
  {
    return *reinterpret_cast<uint64_t*>(&x);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static uint32_t get_bit_representation(const float& x) noexcept
  {
    return ::cuda::std::bit_cast<uint32_t>(x);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static uint64_t get_bit_representation(const double& x) noexcept
  {
    return ::cuda::std::bit_cast<uint64_t>(x);
  }

  /// Return primary vector value const ref
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE const ftype& primary(int i) const noexcept
  {
    if constexpr (Fold <= cub_rfa_max_jump)
    {
      switch (i)
      {
        case 0:
          if constexpr (Fold >= 1)
          {
            return data[0];
          }
          [[fallthrough]];
        case 1:
          if constexpr (Fold >= 2)
          {
            return data[1];
          }
          [[fallthrough]];
        case 2:
          if constexpr (Fold >= 3)
          {
            return data[2];
          }
          [[fallthrough]];
        case 3:
          if constexpr (Fold >= 4)
          {
            return data[3];
          }
          [[fallthrough]];
        case 4:
          if constexpr (Fold >= 5)
          {
            return data[4];
          }
          [[fallthrough]];
        default:
          return data[Fold - 1];
      }
    }
    else
    {
      return data[i];
    }
  }

  /// Return carry vector value const ref
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE const ftype& carry(int i) const noexcept
  {
    if (Fold <= cub_rfa_max_jump)
    {
      switch (i)
      {
        case 0:
          if (Fold >= 1)
          {
            return data[Fold + 0];
          }
          [[fallthrough]];
        case 1:
          if (Fold >= 2)
          {
            return data[Fold + 1];
          }
          [[fallthrough]];
        case 2:
          if (Fold >= 3)
          {
            return data[Fold + 2];
          }
          [[fallthrough]];
        case 3:
          if (Fold >= 4)
          {
            return data[Fold + 3];
          }
          [[fallthrough]];
        case 4:
          if (Fold >= 5)
          {
            return data[Fold + 4];
          }
          [[fallthrough]];
        default:
          return data[2 * Fold - 1];
      }
    }
    else
    {
      return data[Fold + i];
    }
  }

  /// Return primary vector value ref
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ftype& primary(int i) noexcept
  {
    const auto& c = *this;
    return const_cast<ftype&>(c.primary(i));
  }

  /// Return carry vector value ref
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ftype& carry(int i) noexcept
  {
    const auto& c = *this;
    return const_cast<ftype&>(c.carry(i));
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static int exp_val(const ftype x) noexcept
  {
    const auto bits = get_bit_representation(x);
    return (bits >> (mant_dig - 1)) & (2 * max_exp - 1);
  }

  /// Get index of float-point precision
  /// The index of a non-binned type is the smallest index a binned type would
  /// need to have to sum it reproducibly. Higher indices correspond to smaller
  /// bins.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static int binned_dindex(const ftype x)
  {
    int exp = exp_val(x);

    if (exp != 0)
    {
      return ((max_exp + exp_bias) - exp) / bin_width;
    }
    if (x == 0.0)
    {
      return max_index;
    }
    else
    {
      (void) ::cuda::std::frexpf(x, &exp);
      return (::cuda::std::min) ((max_exp - exp) / bin_width, +max_index);
    }
  }

  /// Get index of manually specified binned double precision
  /// The index of a binned type is the bin that it corresponds to. Higher
  /// indices correspond to smaller bins.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE int binned_index() const
  {
    return ((max_exp + mant_dig - bin_width + 1 + exp_bias) - exp_val(primary(0))) / bin_width;
  }

  /// Check if index of manually specified binned floating-point is 0
  /// A quick check to determine if the index is 0
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool is_binned_index_zero() const
  {
    return exp_val(primary(0)) == max_exp + exp_bias;
  }

  //! Update manually specified binned fp with a scalar (X -> Y)
  //!
  //! This method updates the binned fp to an index suitable for adding numbers
  //! with absolute value less than @p max_abs_val
  _CCCL_DEVICE void binned_update(const ftype max_abs_val)
  {
    int X_index = binned_dindex(max_abs_val);
    int shift   = binned_index() - X_index;
    if (shift > 0)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = Fold - 1; i >= 1; i--)
      {
        if (i < shift)
        {
          break;
        }
        primary(i) = primary((i - shift));
        carry(i)   = carry((i - shift));
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < Fold; j++)
      {
        if (j >= shift)
        {
          break;
        }
        primary(j) = binned_bins(j + X_index);
        carry(j)   = 0.0;
      }
    }
  }

  //! Add scalar @p X to suitably binned manually specified binned fp (Y += X)
  //!
  //! Performs the operation Y += X on an binned type Y where the index of Y is
  //! larger than the index of @p X
  _CCCL_DEVICE void binned_deposit(const ftype X)
  {
    ftype M;
    ftype x = X;

    if (is_binned_index_zero())
    {
      M        = primary(0);
      ftype qd = x * compression;
      auto& ql = get_bit_representation(qd);
      ql |= 1;
      qd += M;
      primary(0) = qd;
      M -= qd;
      M *= expansion * 0.5;
      x += M;
      x += M;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 1; i < Fold - 1; i++)
      {
        M  = primary(i);
        qd = x;
        ql |= 1;
        qd += M;
        primary(i) = qd;
        M -= qd;
        x += M;
      }
      qd = x;
      ql |= 1;
      primary((Fold - 1)) += qd;
    }
    else
    {
      ftype qd = x;
      auto& ql = get_bit_representation(qd);
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Fold - 1; i++)
      {
        M  = primary(i);
        qd = x;
        ql |= 1;
        qd += M;
        primary(i) = qd;
        M -= qd;
        x += M;
      }
      qd = x;
      ql |= 1;
      primary((Fold - 1)) += qd;
    }
  }

  //! Renormalize manually specified binned double precision
  //!
  //! Renormalization keeps the primary vector within the necessary bins by
  //! shifting over to the carry vector
  _CCCL_DEVICE _CCCL_FORCEINLINE void binned_renorm()
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < Fold; i++)
    {
      auto tmp_renormd  = primary(i);
      auto& tmp_renorml = get_bit_representation(tmp_renormd);

      carry(i) += static_cast<int>((tmp_renorml >> (mant_dig - 3)) & 3) - 2;

      tmp_renorml &= ~(1ull << (mant_dig - 3));
      tmp_renorml |= 1ull << (mant_dig - 2);
      primary(i) = tmp_renormd;
    }
  }

  //! Add scalar to manually specified binned fp (Y += X)
  //!
  //! Performs the operation Y += X on an binned type Y
  _CCCL_DEVICE _CCCL_FORCEINLINE void binned_add(const ftype x)
  {
    binned_update(x);
    binned_deposit(x);
    binned_renorm();
  }

  //! Add two manually specified binned fp (Y += X)
  //! Performs the operation Y += X
  //!
  //! @param x   Another binned fp of the same type
  _CCCL_DEVICE void binned_add(const ReproducibleFloatingAccumulator& x)
  {
    const auto X_index = x.binned_index();
    const auto Y_index = this->binned_index();
    const auto shift   = Y_index - X_index;
    if (shift > 0)
    {
      // shift Y upwards and add X to Y
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = Fold - 1; i >= 1; i--)
      {
        if (i < shift)
        {
          break;
        }
        primary(i) = x.primary(i) + (primary((i - shift)) - binned_bins(i - shift + Y_index));
        carry(i)   = x.carry(i) + carry((i - shift));
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Fold; i++)
      {
        if (i == shift)
        {
          break;
        }
        primary(i) = x.primary(i);
        carry(i)   = x.carry(i);
      }
    }
    else if (shift < 0)
    {
      // shift X upwards and add X to Y
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Fold; i++)
      {
        if (i < -shift)
        {
          continue;
        }
        primary(i) += x.primary((i + shift)) - binned_bins(X_index + i + shift);
        carry(i) += x.carry((i + shift));
      }
    }
    else if (shift == 0)
    {
      // add X to Y
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Fold; i++)
      {
        primary(i) += x.primary(i) - binned_bins(i + X_index);
        carry(i) += x.carry(i);
      }
    }

    binned_renorm();
  }

  [[nodiscard]] _CCCL_DEVICE double conv_binned_to_double() const
  {
    int i              = 0;
    double Y           = 0.0;
    const auto X_index = binned_index();
    if (X_index <= (3 * mant_dig) / bin_width)
    {
      double scale_down = ::cuda::std::ldexpf(0.5f, 1 - (2 * mant_dig - bin_width));
      double scale_up   = ::cuda::std::ldexpf(0.5f, 1 - (2 * mant_dig - bin_width));
      int scaled        = ::cuda::std::max(::cuda::std::min(Fold, (3 * mant_dig) / bin_width - X_index), 0);
      if (X_index == 0)
      {
        Y += carry(0) * ((binned_bins(0 + X_index) / 6.0) * scale_down * expansion);
        Y += carry(1) * ((binned_bins(1 + X_index) / 6.0) * scale_down);
        Y += (primary(0) - binned_bins(0 + X_index)) * scale_down * expansion;
        i = 2;
      }
      else
      {
        Y += carry(0) * ((binned_bins(0 + X_index) / 6.0) * scale_down);
        i = 1;
      }
      for (; i < scaled; i++)
      {
        Y += carry(i) * ((binned_bins(i + X_index) / 6.0) * scale_down);
        Y += (primary((i - 1)) - binned_bins(i - 1 + X_index)) * scale_down;
      }
      if (i == Fold)
      {
        Y += (primary((Fold - 1)) - binned_bins(Fold - 1 + X_index)) * scale_down;
        return Y * scale_up;
      }
      if (::cuda::std::isinf(Y * scale_up))
      {
        return Y * scale_up;
      }
      Y *= scale_up;
      for (; i < Fold; i++)
      {
        Y += carry(i) * (binned_bins(i + X_index) / 6.0);
        Y += primary((i - 1)) - binned_bins(i - 1 + X_index);
      }
      Y += primary((Fold - 1)) - binned_bins(Fold - 1 + X_index);
    }
    else
    {
      Y += carry(0) * (binned_bins(0 + X_index) / 6.0);
      for (i = 1; i < Fold; i++)
      {
        Y += carry(i) * (binned_bins(i + X_index) / 6.0);
        Y += (primary((i - 1)) - binned_bins(i - 1 + X_index));
      }
      Y += (primary((Fold - 1)) - binned_bins(Fold - 1 + X_index));
    }
    return Y;
  }

  [[nodiscard]] _CCCL_DEVICE float conv_binned_to_float() const
  {
    int i    = 0;
    double Y = 0.0;

    // Note that the following order of summation is in order of decreasing
    // exponent. The following code is specific to SBWIDTH=13, FLT_MANT_DIG=24, and
    // the number of carries equal to 1.
    const auto X_index = binned_index();
    if (X_index == 0)
    {
      Y += static_cast<double>(carry(0)) * static_cast<double>(binned_bins(0 + X_index) / 6.0)
         * static_cast<double>(expansion);
      Y += static_cast<double>(carry(1)) * static_cast<double>(binned_bins(1 + X_index) / 6.0);
      Y += static_cast<double>(primary(0) - binned_bins(0 + X_index)) * static_cast<double>(expansion);
      i = 2;
    }
    else
    {
      Y += static_cast<double>(carry(0)) * static_cast<double>((binned_bins(0 + X_index) / 6.0));
      i = 1;
    }
    for (; i < Fold; i++)
    {
      Y += static_cast<double>(carry(i)) * static_cast<double>(binned_bins(i + X_index) / 6.0);
      Y += static_cast<double>(primary(i - 1) - binned_bins(i - 1 + X_index));
    }
    Y += static_cast<double>(primary(Fold - 1) - binned_bins(Fold - 1 + X_index));
    return static_cast<float>(Y);
  }

public:
  ReproducibleFloatingAccumulator() = default;

  /// Set the binned fp to zero
  _CCCL_DEVICE void zero() noexcept
  {
    data = {};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE constexpr int endurance() const noexcept
  {
    return 1 << (mant_dig - bin_width - 2);
  }

  //! Accumulate an arithmetic @p x into the binned fp.
  //! NOTE: Casts @p x to the type of the binned fp
  _CCCL_TEMPLATE(typename U)
  _CCCL_REQUIRES(::cuda::std::is_arithmetic_v<U>)
  _CCCL_DEVICE ReproducibleFloatingAccumulator& operator+=(const U x)
  {
    binned_add(static_cast<ftype>(x));
    return *this;
  }

  //! Accumulate-subtract an arithmetic @p x into the binned fp.
  //! NOTE: Casts @p x to the type of the binned fp
  _CCCL_TEMPLATE(typename U)
  _CCCL_REQUIRES(::cuda::std::is_arithmetic_v<U>)
  _CCCL_DEVICE ReproducibleFloatingAccumulator& operator-=(const U x)
  {
    binned_add(-static_cast<ftype>(x));
    return *this;
  }

  /// Accumulate a binned fp @p x into the binned fp.
  _CCCL_DEVICE ReproducibleFloatingAccumulator& operator+=(const ReproducibleFloatingAccumulator& other)
  {
    binned_add(other);
    return *this;
  }

  //! Accumulate-subtract a binned fp @p other into the binned fp.
  //! NOTE: Makes a copy and performs arithmetic; slow.
  _CCCL_DEVICE ReproducibleFloatingAccumulator& operator-=(const ReproducibleFloatingAccumulator& other)
  {
    const auto temp = -other;
    binned_add(temp);
  }

  _CCCL_DEVICE friend bool operator==(const ReproducibleFloatingAccumulator& a, const ReproducibleFloatingAccumulator& b)
  {
    return a.data == b.data;
  }

  _CCCL_DEVICE friend bool operator!=(const ReproducibleFloatingAccumulator& a, const ReproducibleFloatingAccumulator& b)
  {
    return !(a == b);
  }

  //! Sets this binned fp equal to the arithmetic value @p x
  //! NOTE: Casts @p x to the type of the binned fp
  _CCCL_TEMPLATE(typename U)
  _CCCL_REQUIRES(::cuda::std::is_arithmetic_v<U>)
  _CCCL_DEVICE ReproducibleFloatingAccumulator& operator=(const U x)
  {
    zero();
    binned_add(static_cast<ftype>(x));
    return *this;
  }

  //! Returns the negative of this binned fp
  //! NOTE: Makes a copy and performs arithmetic; slow.
  [[nodiscard]] _CCCL_DEVICE ReproducibleFloatingAccumulator operator-() const
  {
    ReproducibleFloatingAccumulator temp = *this;
    if (primary(0) != 0.0)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < Fold; i++)
      {
        temp.primary(i) = binned_bins(i + binned_index()) - (primary(i) - binned_bins(i + binned_index()));
        temp.carry(i)   = -carry(i);
      }
    }
    return temp;
  }

  /// Convert this binned fp into its native floating-point representation
  [[nodiscard]] _CCCL_DEVICE ftype conv_to_fp() const
  {
    if (::cuda::std::is_same_v<ftype, float>)
    {
      return conv_binned_to_float();
    }
    else
    {
      return conv_binned_to_double();
    }
  }

  /// Add @p x to the binned fp
  _CCCL_DEVICE void add(const ftype x)
  {
    binned_add(x);
  }

  //////////////////////////////////////
  // MANUAL OPERATIONS; USE WISELY
  //////////////////////////////////////

  //! Rebins for repeated accumulation of scalars with magnitude <= @p mav
  //!
  //! Once rebinned, `endurance` values <= @p mav can be added to the accumulator
  //! with `unsafe_add` after which `renorm()` must be called. See the source of
  //!`add()` for an example
  _CCCL_DEVICE void set_max_val(const ftype mav)
  {
    binned_update(mav);
  }

  //! Add @p x to the binned fp
  //!
  //! This is intended to be used after a call to `set_max_abs_val()`
  _CCCL_DEVICE void unsafe_add(const ftype x)
  {
    binned_deposit(x);
  }

  //! Renormalizes the binned fp
  //!
  //! This is intended to be used after a call to `set_max_abs_val()` and one or
  //! more calls to `unsafe_add()`
  _CCCL_DEVICE void renorm()
  {
    binned_renorm();
  }
};

_CCCL_TEMPLATE(typename FType = float)
_CCCL_REQUIRES(::cuda::std::is_floating_point_v<FType>)
struct rfa_float_transform_t
{
  _CCCL_DEVICE FType operator()(FType accum) const
  {
    return accum;
  }

  _CCCL_DEVICE FType operator()(ReproducibleFloatingAccumulator<FType> accum) const
  {
    return accum.conv_to_fp();
  }
};
} // namespace detail::rfa

CUB_NAMESPACE_END
