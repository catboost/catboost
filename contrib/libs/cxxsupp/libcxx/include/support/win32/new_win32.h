#pragma once
#include <__config>

#pragma push_macro("std");
#undef std
#include <vcruntime_new.h>
#pragma pop_macro("std");

#ifdef std
namespace std
{
#pragma push_macro("std");
#undef std
    using std::nothrow_t;
    using std::nothrow;
#pragma pop_macro("std")
}
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

typedef void (*new_handler)();
_LIBCPP_FUNC_VIS new_handler set_new_handler(new_handler) _NOEXCEPT;
_LIBCPP_FUNC_VIS new_handler get_new_handler() _NOEXCEPT;

_LIBCPP_END_NAMESPACE_STD
