// Copyright (c) 2017-2018 NVIDIA Corporation
// Copyright (c) 2014-2018 Bryce Adelstein Lelbach
// Copyright (c) 2001-2015 Housemarque Oy (housemarque.com)
// Copyright (c) 2007-2015 Hartmut Kaiser
// Copyright (c)      2002 Peter Dimov and Multi Media Ltd
//                         (`THRUST_CURRENT_FUNCTION`)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/preprocessor.h>

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_STRINGIZE(expr)
/// \brief Stringizes the expression \a expr.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_STRINGIZE(foo) << "\n";
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << "foo" << "\n";
/// }
/// \endcode
///
#define THRUST_PP_STRINGIZE(expr)       THRUST_PP_STRINGIZE_IMPL0(expr)
#define THRUST_PP_STRINGIZE_IMPL0(expr) #expr

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_CAT2(a, b)
/// \brief Concatenates the tokens \a a and \b b.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_CAT2(1, THRUST_PP_CAT2(2, 3)) << "\n";
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << 123 << "\n";
/// }
/// \endcode
///
#define THRUST_PP_CAT2(a, b) THRUST_PP_CAT2_IMPL0(a, b)

#if defined(_MSC_VER) && (defined(__EDG__) || defined(__EDG_VERSION__)) \
  && (defined(__INTELLISENSE__) || __EDG_VERSION__ >= 308)
#  define THRUST_PP_CAT2_IMPL0(a, b)   THRUST_PP_CAT2_IMPL1(~, a##b)
#  define THRUST_PP_CAT2_IMPL1(p, res) res
#else
#  define THRUST_PP_CAT2_IMPL0(a, b) a##b
#endif

#define THRUST_PP_CAT3(a, b, c)           \
  THRUST_PP_CAT2(a, THRUST_PP_CAT2(b, c)) \
  /**/

#define THRUST_PP_CAT4(a, b, c, d)                           \
  THRUST_PP_CAT2(a, THRUST_PP_CAT2(b, THRUST_PP_CAT2(c, d))) \
  /**/

#define THRUST_PP_CAT5(a, b, c, d, e)                                           \
  THRUST_PP_CAT2(a, THRUST_PP_CAT2(b, THRUST_PP_CAT2(c, THRUST_PP_CAT2(d, e)))) \
  /**/

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_EXPAND(x)
/// \brief Performs macro expansion on \a x.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// #define FOO_BAR() "foo_bar"
/// #define BUZZ()     THRUST_PP_EXPAND(THRUST_PP_CAT2(FOO_, BAR)())
///
/// int main()
/// {
///   std::cout << BUZZ() << "\n";
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << "foo_bar" << "\n";
/// }
/// \endcode
///
#define THRUST_PP_EXPAND(x)       THRUST_PP_EXPAND_IMPL0(x)
#define THRUST_PP_EXPAND_IMPL0(x) x

#define THRUST_PP_EXPAND_ARGS(...)       THRUST_PP_EXPAND_ARGS_IMPL0(__VA_ARGS__)
#define THRUST_PP_EXPAND_ARGS_IMPL0(...) __VA_ARGS__

#define THRUST_PP_HEAD(x, ...) x

#define THRUST_PP_TAIL(x, ...) __VA_ARGS__

///////////////////////////////////////////////////////////////////////////////

#define THRUST_PP_EMPTY()

#define THRUST_PP_COMMA() ,

#define THRUST_PP_INC _CCCL_PP_INC
#define THRUST_PP_DEC _CCCL_PP_DEC

#define THRUST_PP_BOOL(x) THRUST_PP_BOOL_IMPL0(x)

#define THRUST_PP_BOOL_IMPL0(x) THRUST_PP_CAT2(THRUST_PP_BOOL_IMPL_TAG, x)

#define THRUST_PP_BOOL_IMPL_TAG0   0
#define THRUST_PP_BOOL_IMPL_TAG1   1
#define THRUST_PP_BOOL_IMPL_TAG2   1
#define THRUST_PP_BOOL_IMPL_TAG3   1
#define THRUST_PP_BOOL_IMPL_TAG4   1
#define THRUST_PP_BOOL_IMPL_TAG5   1
#define THRUST_PP_BOOL_IMPL_TAG6   1
#define THRUST_PP_BOOL_IMPL_TAG7   1
#define THRUST_PP_BOOL_IMPL_TAG8   1
#define THRUST_PP_BOOL_IMPL_TAG9   1
#define THRUST_PP_BOOL_IMPL_TAG10  1
#define THRUST_PP_BOOL_IMPL_TAG11  1
#define THRUST_PP_BOOL_IMPL_TAG12  1
#define THRUST_PP_BOOL_IMPL_TAG13  1
#define THRUST_PP_BOOL_IMPL_TAG14  1
#define THRUST_PP_BOOL_IMPL_TAG15  1
#define THRUST_PP_BOOL_IMPL_TAG16  1
#define THRUST_PP_BOOL_IMPL_TAG17  1
#define THRUST_PP_BOOL_IMPL_TAG18  1
#define THRUST_PP_BOOL_IMPL_TAG19  1
#define THRUST_PP_BOOL_IMPL_TAG20  1
#define THRUST_PP_BOOL_IMPL_TAG21  1
#define THRUST_PP_BOOL_IMPL_TAG22  1
#define THRUST_PP_BOOL_IMPL_TAG23  1
#define THRUST_PP_BOOL_IMPL_TAG24  1
#define THRUST_PP_BOOL_IMPL_TAG25  1
#define THRUST_PP_BOOL_IMPL_TAG26  1
#define THRUST_PP_BOOL_IMPL_TAG27  1
#define THRUST_PP_BOOL_IMPL_TAG28  1
#define THRUST_PP_BOOL_IMPL_TAG29  1
#define THRUST_PP_BOOL_IMPL_TAG30  1
#define THRUST_PP_BOOL_IMPL_TAG31  1
#define THRUST_PP_BOOL_IMPL_TAG32  1
#define THRUST_PP_BOOL_IMPL_TAG33  1
#define THRUST_PP_BOOL_IMPL_TAG34  1
#define THRUST_PP_BOOL_IMPL_TAG35  1
#define THRUST_PP_BOOL_IMPL_TAG36  1
#define THRUST_PP_BOOL_IMPL_TAG37  1
#define THRUST_PP_BOOL_IMPL_TAG38  1
#define THRUST_PP_BOOL_IMPL_TAG39  1
#define THRUST_PP_BOOL_IMPL_TAG40  1
#define THRUST_PP_BOOL_IMPL_TAG41  1
#define THRUST_PP_BOOL_IMPL_TAG42  1
#define THRUST_PP_BOOL_IMPL_TAG43  1
#define THRUST_PP_BOOL_IMPL_TAG44  1
#define THRUST_PP_BOOL_IMPL_TAG45  1
#define THRUST_PP_BOOL_IMPL_TAG46  1
#define THRUST_PP_BOOL_IMPL_TAG47  1
#define THRUST_PP_BOOL_IMPL_TAG48  1
#define THRUST_PP_BOOL_IMPL_TAG49  1
#define THRUST_PP_BOOL_IMPL_TAG50  1
#define THRUST_PP_BOOL_IMPL_TAG51  1
#define THRUST_PP_BOOL_IMPL_TAG52  1
#define THRUST_PP_BOOL_IMPL_TAG53  1
#define THRUST_PP_BOOL_IMPL_TAG54  1
#define THRUST_PP_BOOL_IMPL_TAG55  1
#define THRUST_PP_BOOL_IMPL_TAG56  1
#define THRUST_PP_BOOL_IMPL_TAG57  1
#define THRUST_PP_BOOL_IMPL_TAG58  1
#define THRUST_PP_BOOL_IMPL_TAG59  1
#define THRUST_PP_BOOL_IMPL_TAG60  1
#define THRUST_PP_BOOL_IMPL_TAG61  1
#define THRUST_PP_BOOL_IMPL_TAG62  1
#define THRUST_PP_BOOL_IMPL_TAG63  1
#define THRUST_PP_BOOL_IMPL_TAG64  1
#define THRUST_PP_BOOL_IMPL_TAG65  1
#define THRUST_PP_BOOL_IMPL_TAG66  1
#define THRUST_PP_BOOL_IMPL_TAG67  1
#define THRUST_PP_BOOL_IMPL_TAG68  1
#define THRUST_PP_BOOL_IMPL_TAG69  1
#define THRUST_PP_BOOL_IMPL_TAG70  1
#define THRUST_PP_BOOL_IMPL_TAG71  1
#define THRUST_PP_BOOL_IMPL_TAG72  1
#define THRUST_PP_BOOL_IMPL_TAG73  1
#define THRUST_PP_BOOL_IMPL_TAG74  1
#define THRUST_PP_BOOL_IMPL_TAG75  1
#define THRUST_PP_BOOL_IMPL_TAG76  1
#define THRUST_PP_BOOL_IMPL_TAG77  1
#define THRUST_PP_BOOL_IMPL_TAG78  1
#define THRUST_PP_BOOL_IMPL_TAG79  1
#define THRUST_PP_BOOL_IMPL_TAG80  1
#define THRUST_PP_BOOL_IMPL_TAG81  1
#define THRUST_PP_BOOL_IMPL_TAG82  1
#define THRUST_PP_BOOL_IMPL_TAG83  1
#define THRUST_PP_BOOL_IMPL_TAG84  1
#define THRUST_PP_BOOL_IMPL_TAG85  1
#define THRUST_PP_BOOL_IMPL_TAG86  1
#define THRUST_PP_BOOL_IMPL_TAG87  1
#define THRUST_PP_BOOL_IMPL_TAG88  1
#define THRUST_PP_BOOL_IMPL_TAG89  1
#define THRUST_PP_BOOL_IMPL_TAG90  1
#define THRUST_PP_BOOL_IMPL_TAG91  1
#define THRUST_PP_BOOL_IMPL_TAG92  1
#define THRUST_PP_BOOL_IMPL_TAG93  1
#define THRUST_PP_BOOL_IMPL_TAG94  1
#define THRUST_PP_BOOL_IMPL_TAG95  1
#define THRUST_PP_BOOL_IMPL_TAG96  1
#define THRUST_PP_BOOL_IMPL_TAG97  1
#define THRUST_PP_BOOL_IMPL_TAG98  1
#define THRUST_PP_BOOL_IMPL_TAG99  1
#define THRUST_PP_BOOL_IMPL_TAG100 1
#define THRUST_PP_BOOL_IMPL_TAG101 1
#define THRUST_PP_BOOL_IMPL_TAG102 1
#define THRUST_PP_BOOL_IMPL_TAG103 1
#define THRUST_PP_BOOL_IMPL_TAG104 1
#define THRUST_PP_BOOL_IMPL_TAG105 1
#define THRUST_PP_BOOL_IMPL_TAG106 1
#define THRUST_PP_BOOL_IMPL_TAG107 1
#define THRUST_PP_BOOL_IMPL_TAG108 1
#define THRUST_PP_BOOL_IMPL_TAG109 1
#define THRUST_PP_BOOL_IMPL_TAG110 1
#define THRUST_PP_BOOL_IMPL_TAG111 1
#define THRUST_PP_BOOL_IMPL_TAG112 1
#define THRUST_PP_BOOL_IMPL_TAG113 1
#define THRUST_PP_BOOL_IMPL_TAG114 1
#define THRUST_PP_BOOL_IMPL_TAG115 1
#define THRUST_PP_BOOL_IMPL_TAG116 1
#define THRUST_PP_BOOL_IMPL_TAG117 1
#define THRUST_PP_BOOL_IMPL_TAG118 1
#define THRUST_PP_BOOL_IMPL_TAG119 1
#define THRUST_PP_BOOL_IMPL_TAG120 1
#define THRUST_PP_BOOL_IMPL_TAG121 1
#define THRUST_PP_BOOL_IMPL_TAG122 1
#define THRUST_PP_BOOL_IMPL_TAG123 1
#define THRUST_PP_BOOL_IMPL_TAG124 1
#define THRUST_PP_BOOL_IMPL_TAG125 1
#define THRUST_PP_BOOL_IMPL_TAG126 1
#define THRUST_PP_BOOL_IMPL_TAG127 1
#define THRUST_PP_BOOL_IMPL_TAG128 1
#define THRUST_PP_BOOL_IMPL_TAG129 1
#define THRUST_PP_BOOL_IMPL_TAG130 1
#define THRUST_PP_BOOL_IMPL_TAG131 1
#define THRUST_PP_BOOL_IMPL_TAG132 1
#define THRUST_PP_BOOL_IMPL_TAG133 1
#define THRUST_PP_BOOL_IMPL_TAG134 1
#define THRUST_PP_BOOL_IMPL_TAG135 1
#define THRUST_PP_BOOL_IMPL_TAG136 1
#define THRUST_PP_BOOL_IMPL_TAG137 1
#define THRUST_PP_BOOL_IMPL_TAG138 1
#define THRUST_PP_BOOL_IMPL_TAG139 1
#define THRUST_PP_BOOL_IMPL_TAG140 1
#define THRUST_PP_BOOL_IMPL_TAG141 1
#define THRUST_PP_BOOL_IMPL_TAG142 1
#define THRUST_PP_BOOL_IMPL_TAG143 1
#define THRUST_PP_BOOL_IMPL_TAG144 1
#define THRUST_PP_BOOL_IMPL_TAG145 1
#define THRUST_PP_BOOL_IMPL_TAG146 1
#define THRUST_PP_BOOL_IMPL_TAG147 1
#define THRUST_PP_BOOL_IMPL_TAG148 1
#define THRUST_PP_BOOL_IMPL_TAG149 1
#define THRUST_PP_BOOL_IMPL_TAG150 1
#define THRUST_PP_BOOL_IMPL_TAG151 1
#define THRUST_PP_BOOL_IMPL_TAG152 1
#define THRUST_PP_BOOL_IMPL_TAG153 1
#define THRUST_PP_BOOL_IMPL_TAG154 1
#define THRUST_PP_BOOL_IMPL_TAG155 1
#define THRUST_PP_BOOL_IMPL_TAG156 1
#define THRUST_PP_BOOL_IMPL_TAG157 1
#define THRUST_PP_BOOL_IMPL_TAG158 1
#define THRUST_PP_BOOL_IMPL_TAG159 1
#define THRUST_PP_BOOL_IMPL_TAG160 1
#define THRUST_PP_BOOL_IMPL_TAG161 1
#define THRUST_PP_BOOL_IMPL_TAG162 1
#define THRUST_PP_BOOL_IMPL_TAG163 1
#define THRUST_PP_BOOL_IMPL_TAG164 1
#define THRUST_PP_BOOL_IMPL_TAG165 1
#define THRUST_PP_BOOL_IMPL_TAG166 1
#define THRUST_PP_BOOL_IMPL_TAG167 1
#define THRUST_PP_BOOL_IMPL_TAG168 1
#define THRUST_PP_BOOL_IMPL_TAG169 1
#define THRUST_PP_BOOL_IMPL_TAG170 1
#define THRUST_PP_BOOL_IMPL_TAG171 1
#define THRUST_PP_BOOL_IMPL_TAG172 1
#define THRUST_PP_BOOL_IMPL_TAG173 1
#define THRUST_PP_BOOL_IMPL_TAG174 1
#define THRUST_PP_BOOL_IMPL_TAG175 1
#define THRUST_PP_BOOL_IMPL_TAG176 1
#define THRUST_PP_BOOL_IMPL_TAG177 1
#define THRUST_PP_BOOL_IMPL_TAG178 1
#define THRUST_PP_BOOL_IMPL_TAG179 1
#define THRUST_PP_BOOL_IMPL_TAG180 1
#define THRUST_PP_BOOL_IMPL_TAG181 1
#define THRUST_PP_BOOL_IMPL_TAG182 1
#define THRUST_PP_BOOL_IMPL_TAG183 1
#define THRUST_PP_BOOL_IMPL_TAG184 1
#define THRUST_PP_BOOL_IMPL_TAG185 1
#define THRUST_PP_BOOL_IMPL_TAG186 1
#define THRUST_PP_BOOL_IMPL_TAG187 1
#define THRUST_PP_BOOL_IMPL_TAG188 1
#define THRUST_PP_BOOL_IMPL_TAG189 1
#define THRUST_PP_BOOL_IMPL_TAG190 1
#define THRUST_PP_BOOL_IMPL_TAG191 1
#define THRUST_PP_BOOL_IMPL_TAG192 1
#define THRUST_PP_BOOL_IMPL_TAG193 1
#define THRUST_PP_BOOL_IMPL_TAG194 1
#define THRUST_PP_BOOL_IMPL_TAG195 1
#define THRUST_PP_BOOL_IMPL_TAG196 1
#define THRUST_PP_BOOL_IMPL_TAG197 1
#define THRUST_PP_BOOL_IMPL_TAG198 1
#define THRUST_PP_BOOL_IMPL_TAG199 1
#define THRUST_PP_BOOL_IMPL_TAG200 1
#define THRUST_PP_BOOL_IMPL_TAG201 1
#define THRUST_PP_BOOL_IMPL_TAG202 1
#define THRUST_PP_BOOL_IMPL_TAG203 1
#define THRUST_PP_BOOL_IMPL_TAG204 1
#define THRUST_PP_BOOL_IMPL_TAG205 1
#define THRUST_PP_BOOL_IMPL_TAG206 1
#define THRUST_PP_BOOL_IMPL_TAG207 1
#define THRUST_PP_BOOL_IMPL_TAG208 1
#define THRUST_PP_BOOL_IMPL_TAG209 1
#define THRUST_PP_BOOL_IMPL_TAG210 1
#define THRUST_PP_BOOL_IMPL_TAG211 1
#define THRUST_PP_BOOL_IMPL_TAG212 1
#define THRUST_PP_BOOL_IMPL_TAG213 1
#define THRUST_PP_BOOL_IMPL_TAG214 1
#define THRUST_PP_BOOL_IMPL_TAG215 1
#define THRUST_PP_BOOL_IMPL_TAG216 1
#define THRUST_PP_BOOL_IMPL_TAG217 1
#define THRUST_PP_BOOL_IMPL_TAG218 1
#define THRUST_PP_BOOL_IMPL_TAG219 1
#define THRUST_PP_BOOL_IMPL_TAG220 1
#define THRUST_PP_BOOL_IMPL_TAG221 1
#define THRUST_PP_BOOL_IMPL_TAG222 1
#define THRUST_PP_BOOL_IMPL_TAG223 1
#define THRUST_PP_BOOL_IMPL_TAG224 1
#define THRUST_PP_BOOL_IMPL_TAG225 1
#define THRUST_PP_BOOL_IMPL_TAG226 1
#define THRUST_PP_BOOL_IMPL_TAG227 1
#define THRUST_PP_BOOL_IMPL_TAG228 1
#define THRUST_PP_BOOL_IMPL_TAG229 1
#define THRUST_PP_BOOL_IMPL_TAG230 1
#define THRUST_PP_BOOL_IMPL_TAG231 1
#define THRUST_PP_BOOL_IMPL_TAG232 1
#define THRUST_PP_BOOL_IMPL_TAG233 1
#define THRUST_PP_BOOL_IMPL_TAG234 1
#define THRUST_PP_BOOL_IMPL_TAG235 1
#define THRUST_PP_BOOL_IMPL_TAG236 1
#define THRUST_PP_BOOL_IMPL_TAG237 1
#define THRUST_PP_BOOL_IMPL_TAG238 1
#define THRUST_PP_BOOL_IMPL_TAG239 1
#define THRUST_PP_BOOL_IMPL_TAG240 1
#define THRUST_PP_BOOL_IMPL_TAG241 1
#define THRUST_PP_BOOL_IMPL_TAG242 1
#define THRUST_PP_BOOL_IMPL_TAG243 1
#define THRUST_PP_BOOL_IMPL_TAG244 1
#define THRUST_PP_BOOL_IMPL_TAG245 1
#define THRUST_PP_BOOL_IMPL_TAG246 1
#define THRUST_PP_BOOL_IMPL_TAG247 1
#define THRUST_PP_BOOL_IMPL_TAG248 1
#define THRUST_PP_BOOL_IMPL_TAG249 1
#define THRUST_PP_BOOL_IMPL_TAG250 1
#define THRUST_PP_BOOL_IMPL_TAG251 1
#define THRUST_PP_BOOL_IMPL_TAG252 1
#define THRUST_PP_BOOL_IMPL_TAG253 1
#define THRUST_PP_BOOL_IMPL_TAG254 1
#define THRUST_PP_BOOL_IMPL_TAG255 1
#define THRUST_PP_BOOL_IMPL_TAG256 1

///////////////////////////////////////////////////////////////////////////////

#define THRUST_PP_IIF(bit, t, f) THRUST_PP_IIF_IMPL0(bit, t, f)

#if defined(_MSC_VER)
#  define THRUST_PP_IIF_IMPL0(bit, t, f)                                   \
    THRUST_PP_IIF_IMPL1(THRUST_PP_CAT2(THRUST_PP_IIF_IMPL_TAG, bit(t, f))) \
    /**/
#  define THRUST_PP_IIF_IMPL1(id) id
#else
#  define THRUST_PP_IIF_IMPL0(bit, t, f) THRUST_PP_CAT2(THRUST_PP_IIF_IMPL_TAG, bit(t, f))
/**/
#endif

#define THRUST_PP_IIF_IMPL_TAG0(t, f) f
#define THRUST_PP_IIF_IMPL_TAG1(t, f) t

#if defined(__EDG__)
#  define THRUST_PP_IF(cond, t, f) THRUST_PP_IF_IMPL0(cond, t, f)
#  define THRUST_PP_IF_IMPL0(cond, t, f)      \
    THRUST_PP_IIF(THRUST_PP_BOOL(cond), t, f) \
    /**/
#else
#  define THRUST_PP_IF(cond, t, f) THRUST_PP_IIF(THRUST_PP_BOOL(cond), t, f)
#endif

/// \def THRUST_COMMA_IF(cond)
/// \brief If \a cond is true, expands to a comma. Otherwise, expands to nothing.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_STRINGIZE(THRUST_COMMA_IF(0)) << "\n"
///             << THRUST_PP_STRINGIZE(THRUST_COMMA_IF(1)) << "\n";
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << ""  << "\n"
///             << "," << "\n";
/// }
/// \endcode
///
#if defined(__EDG__)
#  define THRUST_PP_COMMA_IF(cond)       THRUST_PP_COMMA_IF_IMPL0(cond)
#  define THRUST_PP_COMMA_IF_IMPL0(cond) THRUST_PP_IF(cond, THRUST_PP_COMMA, THRUST_PP_EMPTY)() /**/
#else
#  define THRUST_PP_COMMA_IF(cond) THRUST_PP_IF(cond, THRUST_PP_COMMA, THRUST_PP_EMPTY)() /**/
#endif

///////////////////////////////////////////////////////////////////////////////

// http://gustedt.wordpress.com/2010/06/08/detect-empty-macro-arguments

// clang-format off
#define THRUST_PP_64TH_ARG(                                                   \
     _1, _2, _3, _4, _5, _6, _7, _8, _9,_10,_11,_12,_13,_14,_15,_16           \
  , _17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32           \
  , _33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48           \
  , _49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,  N           \
  , ...                                                                       \
  ) N                                                                         \
  /**/

#define THRUST_PP_HAS_COMMA(...)                                              \
  THRUST_PP_EXPAND(THRUST_PP_64TH_ARG(                                        \
    __VA_ARGS__                                                               \
  , 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1                                           \
  , 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1                                           \
  , 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1                                           \
  , 1,1,1,1,1,1,1,1,1,1,1,1,1,1,0                                             \
  ))                                                                          \
  /**/
// clang-format on

#define THRUST_PP_TRIGGER_PAREN(...) ,

#define THRUST_PP_IS_VARIADIC_NULLARY(...)                                      \
  THRUST_PP_IS_VARIADIC_NULLARY_IMPL0(                                        \
    /* Test if there is just one argument, eventually an empty one. */        \
    THRUST_PP_HAS_COMMA(__VA_ARGS__),                                         \
    /* Test if THRUST_PP_TRIGGER_PAREN together with the argument adds a */   \
    /* comma. */                                                              \
    THRUST_PP_HAS_COMMA(THRUST_PP_TRIGGER_PAREN __VA_ARGS__),                 \
    /* Test if the argument together with a parenthesis adds a comma. */      \
    THRUST_PP_HAS_COMMA(__VA_ARGS__ (/*empty*/)),                             \
    /* Test if placing it between THRUST_PP_TRIGGER_PAREN and the */          \
    /* parenthesis adds a comma. */                                           \
    THRUST_PP_HAS_COMMA(THRUST_PP_TRIGGER_PAREN __VA_ARGS__ (/*empty*/))      \
  ) \
  /**/

#define THRUST_PP_IS_VARIADIC_NULLARY_IMPL0(_0, _1, _2, _3) \
  THRUST_PP_HAS_COMMA(THRUST_PP_CAT5(THRUST_PP_IS_VARIADIC_NULLARY_IMPL_TAG, _0, _1, _2, _3))

#define THRUST_PP_IS_VARIADIC_NULLARY_IMPL_TAG0001 ,

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_PP_ARITY(...)
/// \brief Returns the number of arguments that it was called with. Must be
///        called with less than 64 arguments.
///
/// \par <b>Example</b>:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << THRUST_PP_ARITY()        << "\n"
///             << THRUST_PP_ARITY(x)       << "\n"
///             << THRUST_PP_ARITY(x, y)    << "\n"
///             << THRUST_PP_ARITY(x, y, z) << "\n";
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << 0 << "\n"
///             << 1 << "\n"
///             << 2 << "\n"
///             << 3 << "\n";
/// }
/// \endcode
///
// clang-format off
#define THRUST_PP_ARITY(...)                                                  \
  THRUST_PP_EXPAND(                                                           \
    THRUST_PP_IF(                                                             \
      THRUST_PP_IS_VARIADIC_NULLARY(__VA_ARGS__)                              \
    , 0                                                                       \
    , THRUST_PP_64TH_ARG(                                                     \
        __VA_ARGS__                                                           \
      , 63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48                       \
      , 47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32                       \
      , 31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16                       \
      , 15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0                       \
      )                                                                       \
    )                                                                         \
  )                                                                           \
  /**/
// clang-format on

/// \def THRUST_PP_DISPATCH(basename, ...)
/// \brief Expands to <tt>basenameN(...)</tt>, where <tt>N</tt> is the
///        number of variadic arguments that \a THRUST_PP_DISPATCH was called
///        with. This macro can be used to implement "macro overloading".
///
/// \par <b>Example</b>:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// #define PLUS(...) THRUST_PP_DISPATCH(PLUS, __VA_ARGS__)
/// #define PLUS0()        0
/// #define PLUS1(x)       x
/// #define PLUS2(x, y)    x + y
/// #define PLUS3(x, y, z) x + y + z
///
/// int main()
/// {
///   std::cout << PLUS()        << "\n"
///             << PLUS(1)       << "\n"
///             << PLUS(1, 2)    << "\n"
///             << PLUS(1, 2, 3) << "\n";
/// }
/// \endcode
///
/// The above code expands to:
///
/// \code
/// #include <thrust/detail/preprocessor.h>
/// #include <iostream>
///
/// int main()
/// {
///   std::cout << 0         << "\n"
///             << 1         << "\n"
///             << 1 + 2     << "\n"
///             << 1 + 2 + 3 << "\n";
/// }
/// \endcode
///
#define THRUST_PP_DISPATCH(basename, ...)                                               \
  THRUST_PP_EXPAND(THRUST_PP_CAT2(basename, THRUST_PP_ARITY(__VA_ARGS__))(__VA_ARGS__)) \
  /**/

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_CURRENT_FUNCTION
/// \brief The name of the current function as a string.
///
#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || defined(__ghs__)
#  define THRUST_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
#  define THRUST_CURRENT_FUNCTION __PRETTY_FUNCTION__
#elif defined(__FUNCSIG__)
#  define THRUST_CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) \
  || (defined(__IBMCTHRUST_PP__) && (__IBMCTHRUST_PP__ >= 500))
#  define THRUST_CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
#  define THRUST_CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#  define THRUST_CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#  define THRUST_CURRENT_FUNCTION __func__
#else
#  define THRUST_CURRENT_FUNCTION "(unknown)"
#endif

///////////////////////////////////////////////////////////////////////////////
