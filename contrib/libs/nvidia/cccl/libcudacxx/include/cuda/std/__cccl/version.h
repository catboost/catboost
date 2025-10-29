//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// This file is somewhat automatically generated. Disable clang-format.
// clang-format off


#ifndef __CCCL_VERSION_H
#define __CCCL_VERSION_H

#define CCCL_VERSION 3001000
#define CCCL_MAJOR_VERSION (CCCL_VERSION / 1000000)
#define CCCL_MINOR_VERSION (((CCCL_VERSION / 1000) % 1000))
#define CCCL_PATCH_VERSION (CCCL_VERSION % 1000)

#if CCCL_PATCH_VERSION > 99
#  error "CCCL patch version cannot be greater than 99 for compatibility with Thrust/CUB's MMMmmmpp format."
#endif

#endif // __CCCL_VERSION_H
