//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_FP_H
#define _LIBCUDACXX___FLOATING_POINT_FP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/cast.h>
#include <cuda/std/__floating_point/cccl_fp.h>
#include <cuda/std/__floating_point/common_type.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__floating_point/conversion_rank_order.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/native_type.h>
#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__floating_point/overflow_handler.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__floating_point/traits.h>

#endif // _LIBCUDACXX___FLOATING_POINT_FP_H
