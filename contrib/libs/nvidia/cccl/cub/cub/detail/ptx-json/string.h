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

namespace ptx_json
{
template <int N>
struct string
{
  static const constexpr auto Length = N;

  __device__ constexpr string(const char (&c)[N])
  {
    for (int i = 0; i < N; ++i)
    {
      str[i] = c[i];
    }
    (void) Length;
  }

  char str[N];
};

__forceinline__ __device__ void comma()
{
  asm volatile("," ::: "memory");
}

#pragma nv_diag_suppress 177
template <char... Cs>
struct storage_helper
{
  // This, and the dance to invoke this through value_traits elsewhere, is necessary because the "C" inline assembly
  // constraint supported by NVCC requires that its argument is a pointer to a constant array of type char; NVCC also
  // doesn't allow passing raw character literals as pointer template arguments; and *also* it seems to look at the type
  // of a containing object, not a subobject it is given, when passed in a pointer to an array inside a literal type.
  // All of this means that we can't just pass strings, and *also* we can't just use the string<N>::array member above
  // as the string literal; therefore, using the fact that the length of the string is a core constant expression in the
  // definition of value_traits, we can generate a variadic pack that allows us to expand the contents of
  // string<N>::array into a comma separated list of N chars. We can then plug that in as template arguments to
  // storage_helper, which then can, as below, turn that into its own char array that NVCC accepts as an argument for a
  // "C" inline assembly constraint.
  static const constexpr char value[] = {Cs...};
};
#pragma nv_diag_default 177
} // namespace ptx_json
