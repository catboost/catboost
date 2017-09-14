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

// Includes some standard C headers for size_t, memset, etc.
//
// Also, permanently disables a number of warnings produced
// by Microsoft's compiler when it includes standard headers
// (surprisingly, also by Microsoft).

#ifndef CRCUTIL_STD_HEADERS_H_
#define CRCUTIL_STD_HEADERS_H_

#if defined(_MSC_VER)
// '4' bytes padding added after data member ...
#pragma warning(disable:4820)

// unreferenced inline function has been removed ...
#pragma warning(disable:4514)

// conditional expression is constant
#pragma warning(disable: 4127)

// function ... not inlined
#pragma warning(disable: 4710)

// function ... selected for automatic inline expansion
#pragma warning(disable: 4711)

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#endif  // defined(_MSC_VER)

// #define _CSTDLIB_
#include <stdio.h>      // always handy
#include <string.h>     // memset
#include <stdlib.h>     // size_t, _rotl/_rotl64(MSC)
#include <stddef.h>     // ptrdiff_t (GNUC)
#include <stdarg.h>     // va_list

#endif  // CRCUTIL_STD_HEADERS_H_
