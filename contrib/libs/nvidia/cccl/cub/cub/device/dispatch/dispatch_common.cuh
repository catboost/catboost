// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

// Options for specifying memory aliasing
enum class MayAlias
{
  Yes,
  No
};

// Options for specifying sorting order.
enum class SortOrder
{
  Ascending,
  Descending
};

// Options for specifying the behavior of the stream compaction algorithm.
enum class SelectImpl
{
  // Stream compaction, discarding rejected items. It's required that memory of input and output are disjoint.
  Select,
  // Stream compaction, discarding rejected items. Memory of the input may be identical to the memory of the output.
  SelectPotentiallyInPlace,
  // Partition, keeping rejected items. It's required that memory of input and output are disjoint.
  Partition
};

CUB_NAMESPACE_END
