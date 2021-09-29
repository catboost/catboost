// Copyright 2020 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TCMALLOC_INTERNAL_OPTIMIZATION_H_
#define TCMALLOC_INTERNAL_OPTIMIZATION_H_

#include "tcmalloc/internal/logging.h"

// Our wrapper for __builtin_assume, allowing us to check the assumption on
// debug builds.
#ifndef NDEBUG
#ifdef __clang__
#define ASSUME(cond) CHECK_CONDITION(cond), __builtin_assume(cond)
#else
#define ASSUME(cond) \
  CHECK_CONDITION(cond), (!(cond) ? __builtin_unreachable() : (void)0)
#endif
#else
#ifdef __clang__
#define ASSUME(cond) __builtin_assume(cond)
#else
#define ASSUME(cond) (!(cond) ? __builtin_unreachable() : (void)0)
#endif
#endif

// Annotations for functions that are not affected by nor affect observable
// state of the program.
#if ABSL_HAVE_ATTRIBUTE(const)
#define TCMALLOC_ATTRIBUTE_CONST __attribute__((const))
#else
#define TCMALLOC_ATTRIBUTE_CONST
#endif

#endif  // TCMALLOC_INTERNAL_OPTIMIZATION_H_
