// Copyright 2010 Google Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Poor man's platform-independent implementation of aligned memory allocator.

#ifndef CRCUTIL_ALIGNED_ALLOC_H_
#define CRCUTIL_ALIGNED_ALLOC_H_

#include "std_headers.h"    // size_t, ptrdiff_t

namespace crcutil {

// Allocates a block of memory of "size" bytes so that a field
// at "field_offset" is aligned on "align" boundary.
//
// NB #1: "align" shall be exact power of two.
//
// NB #2: memory allocated by AlignedAlloc should be release by AlignedFree().
//
inline void *AlignedAlloc(size_t size,
                          size_t field_offset,
                          size_t align,
                          const void **allocated_mem) {
  if (align == 0 || (align & (align - 1)) != 0 || align < sizeof(char *)) {
    align = sizeof(*allocated_mem);
  }
  size += align - 1 + sizeof(*allocated_mem);
  char *allocated_memory = new char[size];
  char *aligned_memory = allocated_memory + sizeof(*allocated_mem);
  field_offset &= align - 1;
  size_t actual_alignment =
      reinterpret_cast<size_t>(aligned_memory + field_offset) & (align - 1);
  if (actual_alignment != 0) {
    aligned_memory += align - actual_alignment;
  }
  reinterpret_cast<char **>(aligned_memory)[-1] = allocated_memory;

  if (allocated_mem != NULL) {
    *allocated_mem = allocated_memory;
  }

  return aligned_memory;
}

// Frees memory allocated by AlignedAlloc().
inline void AlignedFree(void *aligned_memory) {
  if (aligned_memory != NULL) {
    char *allocated_memory = reinterpret_cast<char **>(aligned_memory)[-1];
    delete[] allocated_memory;
  }
}

}  // namespace crcutil

#endif  // CRCUTIL_ALIGNED_ALLOC_H_
