#pragma once

#  if defined(_LIBCPP_COMPILER_CLANG_BASED)
#    define _LIBCPP_REINITIALIZES_OBJECT [[clang::reinitializes]]
#  else
#    define _LIBCPP_REINITIALIZES_OBJECT
#  endif

#  if !__has_builtin(__builtin_is_constant_evaluated) || __CUDACC_VER_MAJOR__ == 10
#    define _LIBCPP_HAS_NO_BUILTIN_IS_CONSTANT_EVALUATED
#  endif

#if defined(__CUDACC__)
    #undef _LIBCPP_DECLSPEC_EMPTY_BASES
    #define _LIBCPP_DECLSPEC_EMPTY_BASES

    #undef _LIBCPP_NODEBUG
    #define _LIBCPP_NODEBUG
    
    #undef _LIBCPP_PACKED
    #define _LIBCPP_PACKED

    #undef _LIBCPP_USING_IF_EXISTS
    #define _LIBCPP_USING_IF_EXISTS
#endif 
