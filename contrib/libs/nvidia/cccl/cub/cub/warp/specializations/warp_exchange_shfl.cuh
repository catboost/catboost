/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN

namespace detail
{

template <typename InputT, int ITEMS_PER_THREAD, int LOGICAL_WARP_THREADS = warp_threads>
class WarpExchangeShfl
{
  static_assert(PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE, "LOGICAL_WARP_THREADS must be a power of two");

  static_assert(ITEMS_PER_THREAD == LOGICAL_WARP_THREADS,
                "WARP_EXCHANGE_SHUFFLE currently only works when ITEMS_PER_THREAD == "
                "LOGICAL_WARP_THREADS");

  static constexpr bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == warp_threads;

  // concrete recursion class
  template <typename OutputT, int IDX, int SIZE>
  class CompileTimeArray : protected CompileTimeArray<OutputT, IDX + 1, SIZE>
  {
  protected:
    InputT val;

    template <int NUM_ENTRIES>
    _CCCL_DEVICE void Foreach(const bool xor_bit_set, const unsigned mask)
    {
      // The implementation here is a recursive divide-and-conquer approach
      // that takes inspiration from:
      // https://forums.developer.nvidia.com/t/transposing-register-held-matrices-with-warp-shuffles-need-help/38652/2
      //
      // At its core, the problem can be boiled down to transposing the matrix
      //
      //   A B
      //   C D
      //
      // by swapping the off-diagonal elements/sub-matrices B and C recursively.
      //
      // This implementation requires power-of-two matrices. In order to avoid
      // the use of local or shared memory, all index computation has to occur
      // at compile-time, since registers cannot be indexed dynamically.
      // Furthermore, using recursive templates reduces the mental load on the
      // optimizer, since lowering for-loops into registers oftentimes requires
      // finagling them with #pragma unroll, which leads to brittle code.
      //
      // To illustrate this algorithm, let's pretend we have warpSize = 8,
      // where t0, ..., t7 denote the 8 threads, and thread i has an array of
      // size 8 with data = [Ai, Bi, ..., Hi] (the columns in the schematics).
      //
      // In the first round, we exchange the largest 4x4 off-diagonal
      // submatrix. Boxes illustrate the submatrices to be exchanged.
      //
      //       ROUND 1
      //       =======
      //  t0  t1  t2  t3  t4  t5  t6  t7
      //                 ┌──────────────┐
      //  A0  A1  A2  A3 │A4  A5  A6  A7│    NUM_ENTRIES == 4 tells us how many
      //                 │              │       entries we have in a submatrix,
      //                 │              │       in this case 4 and the size of
      //  B0  B1  B2  B3 │B4  B5  B6  B7│       the jumps between submatrices.
      //                 │              │
      //                 │              │  1. t[0,1,2,3] data[4] swap with t[4,5,6,7]'s data[0]
      //  C0  C1  C2  C3 │C4  C5  C6  C7│  2. t[0,1,2,3] data[5] swap with t[4,5,6,7]'s data[1]
      //                 │              │  3. t[0,1,2,3] data[6] swap with t[4,5,6,7]'s data[2]
      //                 │              │  4. t[0,1,2,3] data[7] swap with t[4,5,6,7]'s data[3]
      //  D0  D1  D2  D3 │D4  D5  D6  D7│
      //                 └──────────────┘
      // ┌──────────────┐
      // │E0  E1  E2  E3│ E4  E5  E6  E7
      // │              │
      // │              │
      // │F0  F1  F2  F3│ F4  F5  F6  F7
      // │              │
      // │              │
      // │G0  G1  G2  G3│ G4  G5  G6  G7
      // │              │
      // │              │
      // │H0  H1  H2  H3│ H4  H5  H6  H7
      // └──────────────┘
      //
      //       ROUND 2
      //       =======
      //  t0  t1  t2  t3  t4  t5  t6  t7
      //         ┌──────┐        ┌──────┐
      //  A0  A1 │A2  A3│ E0  E1 │E2  E3│    NUM_ENTRIES == 2 so we have 2
      //         │      │        │      │       submatrices per thread and there
      //         │      │        │      │       are 2 elements between these
      //  B0  B1 │B2  B3│ F0  F1 │F2  F3│       submatrices.
      //         └──────┘        └──────┘
      // ┌──────┐        ┌──────┐          1. t[0,1,4,5] data[2] swap with t[2,3,6,7]'s data[0]
      // │C0  C1│ C2  C3 │G0  G1│ G2  G3   2. t[0,1,4,5] data[3] swap with t[2,3,6,7]'s data[1]
      // │      │        │      │          3. t[0,1,4,5] data[6] swap with t[2,3,6,7]'s data[4]
      // │      │        │      │          4. t[0,1,4,5] data[7] swap with t[2,3,6,7]'s data[5]
      // │D0  D1│ D2  D3 │H0  H1│ H2  H3
      // └──────┘        └──────┘
      //         ┌──────┐        ┌──────┐
      //  A4  A5 │A6  A7│ E4  E5 │E6  E7│
      //         │      │        │      │
      //         │      │        │      │
      //  B4  B5 │B6  B7│ F4  F5 │F6  F7│
      //         └──────┘        └──────┘
      // ┌──────┐        ┌──────┐
      // │C4  C5│ C6  C7 │G4  G5│ G6  G7
      // │      │        │      │
      // │      │        │      │
      // │D4  D5│ D6  D7 │H4  H5│ H6  H7
      // └──────┘        └──────┘
      //
      //       ROUND 3
      //       =======
      //  t0  t1  t2  t3  t4  t5  t6  t7
      //     ┌──┐    ┌──┐    ┌──┐    ┌──┐
      //  A0 │A1│ C0 │C1│ E0 │E1│ G0 │G1│    NUM_ENTRIES == 1 so we have 4
      //     └──┘    └──┘    └──┘    └──┘       submatrices per thread and there
      // ┌──┐    ┌──┐    ┌──┐    ┌──┐           is 1 element between these
      // │B0│ B1 │D0│ D1 │F0│ F1 │H0│ H1        submatrices.
      // └──┘    └──┘    └──┘    └──┘
      //     ┌──┐    ┌──┐    ┌──┐    ┌──┐  1. t[0,2,4,6] data[1] swap with t[1,3,5,7]'s data[0]
      //  A2 │A3│ C2 │C3│ E2 │E3│ G2 │G3│  2. t[0,2,4,6] data[3] swap with t[1,3,5,7]'s data[2]
      //     └──┘    └──┘    └──┘    └──┘  3. t[0,2,4,6] data[5] swap with t[1,3,5,7]'s data[4]
      // ┌──┐    ┌──┐    ┌──┐    ┌──┐      4. t[0,2,4,6] data[7] swap with t[1,3,5,7]'s data[6]
      // │B2│ B3 │D2│ D3 │F2│ F3 │H2│ H3
      // └──┘    └──┘    └──┘    └──┘
      //     ┌──┐    ┌──┐    ┌──┐    ┌──┐
      //  A4 │A5│ C4 │C5│ E4 │E5│ G4 │G5│
      //     └──┘    └──┘    └──┘    └──┘
      // ┌──┐    ┌──┐    ┌──┐    ┌──┐
      // │B4│ B5 │D4│ D5 │F4│ F5 │H4│ H5
      // └──┘    └──┘    └──┘    └──┘
      //     ┌──┐    ┌──┐    ┌──┐    ┌──┐
      //  A6 │A7│ C6 │C7│ E6 │E7│ G6 │G7│
      //     └──┘    └──┘    └──┘    └──┘
      // ┌──┐    ┌──┐    ┌──┐    ┌──┐
      // │B6│ B7 │D6│ D7 │F6│ F7 │H6│ H7
      // └──┘    └──┘    └──┘    └──┘
      //
      //       RESULT
      //       ======
      //  t0  t1  t2  t3  t4  t5  t6  t7
      //
      //  A0  B0  C0  D0  E0  F0  G0  H0
      //
      //
      //  A1  B1  C1  D1  E1  F1  G1  H1
      //
      //
      //  A2  B2  C2  D2  E2  F2  G2  H2
      //
      //
      //  A3  B3  C3  D3  E3  F3  G3  H3
      //
      //
      //  A4  B4  C4  D4  E4  F4  G4  H4
      //
      //
      //  A5  B5  C5  D5  E5  F5  G5  H5
      //
      //
      //  A6  B6  C6  D6  E6  F6  G6  H6
      //
      //
      //  A7  B7  C7  D7  E7  F7  G7  H7
      //

      // NOTE: Do *NOT* try to refactor this code to use a reference, since nvcc
      //       tends to choke on it and then drop everything into local memory.
      const InputT send_val = (xor_bit_set ? CompileTimeArray<OutputT, IDX, SIZE>::val
                                           : CompileTimeArray<OutputT, IDX + NUM_ENTRIES, SIZE>::val);
      const InputT recv_val = __shfl_xor_sync(mask, send_val, NUM_ENTRIES, LOGICAL_WARP_THREADS);
      (xor_bit_set ? CompileTimeArray<OutputT, IDX, SIZE>::val
                   : CompileTimeArray<OutputT, IDX + NUM_ENTRIES, SIZE>::val) = recv_val;

      constexpr int next_idx = IDX + 1 + ((IDX + 1) % NUM_ENTRIES == 0) * NUM_ENTRIES;
      CompileTimeArray<OutputT, next_idx, SIZE>::template Foreach<NUM_ENTRIES>(xor_bit_set, mask);
    }

    // terminate recursion
    _CCCL_DEVICE void TransposeImpl(unsigned int, unsigned int, constant_t<0>) {}

    template <int NUM_ENTRIES>
    _CCCL_DEVICE void TransposeImpl(const unsigned int lane_id, const unsigned int mask, constant_t<NUM_ENTRIES>)
    {
      const bool xor_bit_set = lane_id & NUM_ENTRIES;
      Foreach<NUM_ENTRIES>(xor_bit_set, mask);

      TransposeImpl(lane_id, mask, constant_v<NUM_ENTRIES / 2>);
    }

  public:
    _CCCL_DEVICE
    CompileTimeArray(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
        : CompileTimeArray<OutputT, IDX + 1, SIZE>{input_items, output_items}
        , val{input_items[IDX]}
    {}

    _CCCL_DEVICE ~CompileTimeArray()
    {
      this->output_items[IDX] = val;
    }

    _CCCL_DEVICE void Transpose(const unsigned int lane_id, const unsigned int mask)
    {
      TransposeImpl(lane_id, mask, constant_v<ITEMS_PER_THREAD / 2>);
    }
  };

  // terminating partial specialization
  template <typename OutputT, int SIZE>
  class CompileTimeArray<OutputT, SIZE, SIZE>
  {
  protected:
    // used for dumping back the individual values after transposing
    InputT (&output_items)[ITEMS_PER_THREAD];

    template <int>
    _CCCL_DEVICE void Foreach(bool, unsigned)
    {}

  public:
    _CCCL_DEVICE CompileTimeArray(const InputT (&)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
        : output_items{output_items}
    {}
  };

  const unsigned int lane_id;
  const unsigned int warp_id;
  const unsigned int member_mask;

public:
  using TempStorage = NullType;

  WarpExchangeShfl() = delete;

  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpExchangeShfl(TempStorage&)
      : lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : (::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {}

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  BlockedToStriped(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    CompileTimeArray<OutputT, 0, ITEMS_PER_THREAD> arr{input_items, output_items};
    arr.Transpose(lane_id, member_mask);
  }

  template <typename OutputT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  StripedToBlocked(const InputT (&input_items)[ITEMS_PER_THREAD], OutputT (&output_items)[ITEMS_PER_THREAD])
  {
    BlockedToStriped(input_items, output_items);
  }

  // Trick to keep the compiler from inferring that the
  // condition in the static_assert is always false.
  template <typename T>
  struct dependent_false
  {
    static constexpr bool value = false;
  };

  template <typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(InputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])
  {
    static_assert(dependent_false<OffsetT>::value,
                  "Shuffle specialization of warp exchange does not support\n"
                  "ScatterToStriped(InputT (&items)[ITEMS_PER_THREAD],\n"
                  "                 OffsetT (&ranks)[ITEMS_PER_THREAD])");
  }

  template <typename OutputT, typename OffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScatterToStriped(const InputT (&)[ITEMS_PER_THREAD], OutputT (&)[ITEMS_PER_THREAD], OffsetT (&)[ITEMS_PER_THREAD])
  {
    static_assert(dependent_false<OffsetT>::value,
                  "Shuffle specialization of warp exchange does not support\n"
                  "ScatterToStriped(const InputT (&input_items)[ITEMS_PER_THREAD],\n"
                  "                 OutputT (&output_items)[ITEMS_PER_THREAD],\n"
                  "                 OffsetT (&ranks)[ITEMS_PER_THREAD])");
  }
};

} // namespace detail

CUB_NAMESPACE_END
