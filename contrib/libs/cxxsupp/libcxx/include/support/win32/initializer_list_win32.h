#pragma once
#include <__config>

#pragma push_macro("std");
#undef std
#include _LIBCPP_MSVC_INCLUDE(yvals.h)
#include _LIBCPP_MSVC_INCLUDE(initializer_list)
#pragma pop_macro("std");

#ifdef std
namespace std
{
#pragma push_macro("std");
#undef std
    using std::initializer_list;
#pragma pop_macro("std")
}
#endif