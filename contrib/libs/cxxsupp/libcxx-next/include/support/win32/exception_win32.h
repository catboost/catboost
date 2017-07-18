#pragma once
#include <__config>

#pragma push_macro("std");
#undef std
#include _LIBCPP_MSVC_INCLUDE(vcruntime_exception.h)
#pragma pop_macro("std");

#ifdef std
namespace std
{
#pragma push_macro("std");
#undef std
    using std::exception;
    using std::bad_exception;

    using std::bad_alloc;
    using std::bad_array_new_length;
#pragma pop_macro("std")
}

#endif
