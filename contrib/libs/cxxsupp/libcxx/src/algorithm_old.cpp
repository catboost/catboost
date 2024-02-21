//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

namespace old_sort {

// TODO(varconst): this currently doesn't benefit `ranges::sort` because it uses `ranges::less` instead of `__less`.

template void __sort<std::__less<>&, char*>(char*, char*, std::__less<>&);
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template void __sort<std::__less<>&, wchar_t*>(wchar_t*, wchar_t*, std::__less<>&);
#endif
template void __sort<std::__less<>&, signed char*>(signed char*, signed char*, std::__less<>&);
template void __sort<std::__less<>&, unsigned char*>(unsigned char*, unsigned char*, std::__less<>&);
template void __sort<std::__less<>&, short*>(short*, short*, std::__less<>&);
template void __sort<std::__less<>&, unsigned short*>(unsigned short*, unsigned short*, std::__less<>&);
template void __sort<std::__less<>&, int*>(int*, int*, std::__less<>&);
template void __sort<std::__less<>&, unsigned*>(unsigned*, unsigned*, std::__less<>&);
template void __sort<std::__less<>&, long*>(long*, long*, std::__less<>&);
template void __sort<std::__less<>&, unsigned long*>(unsigned long*, unsigned long*, std::__less<>&);
template void __sort<std::__less<>&, long long*>(long long*, long long*, std::__less<>&);
template void __sort<std::__less<>&, unsigned long long*>(unsigned long long*, unsigned long long*, std::__less<>&);
template void __sort<std::__less<>&, float*>(float*, float*, std::__less<>&);
template void __sort<std::__less<>&, double*>(double*, double*, std::__less<>&);
template void __sort<std::__less<>&, long double*>(long double*, long double*, std::__less<>&);

template bool __insertion_sort_incomplete<std::__less<>&, char*>(char*, char*, std::__less<>&);
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
template bool __insertion_sort_incomplete<std::__less<>&, wchar_t*>(wchar_t*, wchar_t*, std::__less<>&);
#endif
template bool __insertion_sort_incomplete<std::__less<>&, signed char*>(signed char*, signed char*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, unsigned char*>(unsigned char*, unsigned char*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, short*>(short*, short*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, unsigned short*>(unsigned short*, unsigned short*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, int*>(int*, int*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, unsigned*>(unsigned*, unsigned*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, long*>(long*, long*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, unsigned long*>(unsigned long*, unsigned long*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, long long*>(long long*, long long*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, unsigned long long*>(unsigned long long*, unsigned long long*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, float*>(float*, float*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, double*>(double*, double*, std::__less<>&);
template bool __insertion_sort_incomplete<std::__less<>&, long double*>(long double*, long double*, std::__less<>&);

template unsigned __sort5<std::__less<>&, long double*>(long double*, long double*, long double*, long double*, long double*, std::__less<>&);

} // namespace old_sort
