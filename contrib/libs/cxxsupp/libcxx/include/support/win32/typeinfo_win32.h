#ifndef _LIBCPP_SUPPORT_WIN32_TYPEINFO_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_TYPEINFO_WIN32_H

#if !defined(_LIBCPP_MSVCRT)
#error "This header complements Microsoft's C Runtime library, and should not be included otherwise."
#else

#if _LIBCXX_USE_NATIVE_STL
#error "This header is not supposed to be included from inside of native STL."
#endif

#undef std

#define _LIBCXX_USE_NATIVE_STL 1
#include _LIBCXX_NATIVE_HEADER(typeinfo)
#undef _LIBCXX_USE_NATIVE_STL

namespace msvc_std
{
using std::type_info;
using std::bad_cast;
using std::bad_typeid;
}
#define std Y_STD_NAMESPACE

#endif // _LIBCPP_MSVCRT

#endif // _LIBCPP_SUPPORT_WIN32_TYPEINFO_WIN32_H
