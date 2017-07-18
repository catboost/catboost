#ifndef _LIBCPP_SUPPORT_WIN32_NEW_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_NEW_WIN32_H

#if !defined(_LIBCPP_MSVCRT)
#error "This header complements Microsoft's C Runtime library, and should not be included otherwise."
#else

#if _LIBCXX_USE_NATIVE_STL
#error "This header is not supposed to be included from inside of native STL."
#endif

#undef std

#define _LIBCXX_USE_NATIVE_STL 1
#include _LIBCXX_NATIVE_HEADER(new)
#undef _LIBCXX_USE_NATIVE_STL

namespace msvc_std
{
using std::bad_alloc;
using std::bad_array_new_length;
using std::nothrow_t;
using std::nothrow;
using std::new_handler;
using std::set_new_handler;
using std::get_new_handler;
}
#define std Y_STD_NAMESPACE

#endif // _LIBCPP_MSVCRT

#endif // _LIBCPP_SUPPORT_WIN32_NEW_WIN32_H
