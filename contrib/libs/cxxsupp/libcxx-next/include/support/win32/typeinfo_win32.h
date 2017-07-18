#pragma once
#include <__config>

#pragma push_macro("std");
#undef std
#include _LIBCPP_MSVC_INCLUDE(vcruntime_typeinfo.h)
#pragma pop_macro("std");

#ifdef std
namespace std
{
#pragma push_macro("std");
#undef std
    using std::type_info;
    using std::bad_cast;
    using std::bad_typeid;
#pragma pop_macro("std")
}
#endif