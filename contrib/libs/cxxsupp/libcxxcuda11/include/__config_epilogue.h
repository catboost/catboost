#pragma once

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
