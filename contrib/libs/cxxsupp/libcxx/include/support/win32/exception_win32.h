#ifndef _LIBCPP_SUPPORT_WIN32_EXCEPTION_WIN32_H
#define _LIBCPP_SUPPORT_WIN32_EXCEPTION_WIN32_H

#if !defined(_LIBCPP_MSVCRT)
#error "This header complements Microsoft's C Runtime library, and should not be included otherwise."
#else

#if _LIBCXX_USE_NATIVE_STL
#error "This header is not supposed to be included from inside of native STL."
#endif

#undef std

#define _LIBCXX_USE_NATIVE_STL 1
#include _LIBCXX_NATIVE_HEADER(exception)
#undef _LIBCXX_USE_NATIVE_STL

namespace msvc_std
{
using std::exception;
using std::bad_exception;

using std::unexpected_handler;
using std::set_unexpected;
using std::get_unexpected;
using std::unexpected;

using std::terminate_handler;
using std::set_terminate;
using std::get_terminate;
using std::terminate;

using std::uncaught_exception;
using std::exception_ptr;
using std::current_exception;
using std::rethrow_exception;
using std::make_exception_ptr;
}
#define std Y_STD_NAMESPACE

#endif // _LIBCPP_MSVCRT

#endif // _LIBCPP_SUPPORT_WIN32_EXCEPTION_WIN32_H
