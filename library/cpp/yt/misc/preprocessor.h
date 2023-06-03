#pragma once

/*!
 * \file preprocessor.h
 * \brief Preprocessor metaprogramming macros
 */

#if !defined(_MSC_VER) && !defined(__GNUC__)
#   error Your compiler is not currently supported.
#endif

/*!
 * \defgroup yt_pp Preprocessor metaprogramming macros
 * \ingroup yt_commons
 *
 * This is collection of macro definitions for various metaprogramming tasks
 * with the preprocessor.
 *
 * \{
 *
 * \page yt_pp_sequences Sequences
 * Everything revolves around the concept of a \em sequence. A typical
 * sequence is encoded like <tt>(1)(2)(3)...</tt>. Internally this allows
 * to apply some macro to the every element in the sequence (see #PP_FOR_EACH).
 *
 * Note that sequences can be nested, i. e. <tt>((1)(2)(3))(a)(b)(c)</tt>
 *
 * \page yt_pp_examples Examples
 * Please refer to the unit test for an actual example of usage
 * (unittests/preprocessor_ut.cpp).
 *
 */

//! Concatenates two tokens.
#define PP_CONCAT(x, y)   PP_CONCAT_A(x, y)
//! \cond Implementation
#define PP_CONCAT_A(x, y) PP_CONCAT_B(x, y)
#define PP_CONCAT_B(x, y) x ## y
//! \endcond

//! Transforms token into the string forcing argument expansion.
#define PP_STRINGIZE(x)   PP_STRINGIZE_A(x)
//! \cond Implementation
#define PP_STRINGIZE_A(x) PP_STRINGIZE_B(x)
#define PP_STRINGIZE_B(x) #x
//! \endcond

//! \cond Implementation
#define PP_LEFT_PARENTHESIS (
#define PP_RIGHT_PARENTHESIS )
#define PP_COMMA() ,
#define PP_EMPTY()
//! \endcond

//! Removes the enclosing parens, if any.
#define PP_DEPAREN(...) PP_DEPAREN_A(PP_DEPAREN_C __VA_ARGS__)
//! \cond Implementation
#define PP_DEPAREN_C(...) PP_DEPAREN_C __VA_ARGS__
#define PP_DEPAREN_A(...) PP_DEPAREN_B(__VA_ARGS__)
#define PP_DEPAREN_B(...) PP_DEPAREN_D_ ## __VA_ARGS__
#define PP_DEPAREN_D_PP_DEPAREN_C
//! \endcond

//! Performs (non-lazy) conditional expansion.
/*!
 * \param cond Condition; should expands to either \c PP_TRUE or \c PP_FALSE.
 * \param _then Expansion result in case when \c cond holds.
 * \param _else Expansion result in case when \c cond does not hold.
 */
#define PP_IF(cond, _then, _else) PP_CONCAT(PP_IF_, cond)(_then, _else)
//! \cond Implementation
#define PP_IF_PP_TRUE(x, y) x
#define PP_IF_PP_FALSE(x, y) y
//! \endcond

//! Tests whether supplied argument can be treated as a sequence
//! (i. e. <tt>()()()...</tt>)
#define PP_IS_SEQUENCE(arg) PP_CONCAT(PP_IS_SEQUENCE_B_, PP_COUNT((PP_NIL PP_IS_SEQUENCE_A arg PP_NIL)))
//! \cond Implementation
#define PP_IS_SEQUENCE_A(_) PP_RIGHT_PARENTHESIS PP_LEFT_PARENTHESIS
#define PP_IS_SEQUENCE_B_1 PP_FALSE
#define PP_IS_SEQUENCE_B_2 PP_TRUE
//! \endcond

//! Computes the number of elements in the sequence.
#define PP_COUNT(...) PP_COUNT_IMPL(__VA_ARGS__)

//! Removes first \c n elements from the sequence.
#define PP_KILL(seq, n) PP_KILL_IMPL(seq, n)

//! Extracts the head of the sequence.
/*! For example, \code PP_HEAD((0)(1)(2)(3)) == 0 \endcode
 */
#define PP_HEAD(...) PP_HEAD_IMPL(__VA_ARGS__)

//! Extracts the tail of the sequence.
/*! For example, \code PP_TAIL((0)(1)(2)(3)) == (1)(2)(3) \endcode
 */
#define PP_TAIL(...) PP_TAIL_IMPL(__VA_ARGS__)

//! Extracts the element with the specified index from the sequence.
/*! For example, \code PP_ELEMENT((0)(1)(2)(3), 1) == 1 \endcode
 */
#define PP_ELEMENT(seq, index) PP_ELEMENT_IMPL(seq, index)

//! Applies the macro to every member of the sequence.
/*! For example,
 * \code
 * #define MyFunctor(x) +x+
 * PP_FOR_EACH(MyFunctor, (0)(1)(2)(3)) == +0+ +1+ +2+ +3+
 * \encode
 */
#define PP_FOR_EACH(what, seq) PP_FOR_EACH_IMPL(what, seq)

//! Declares an anonymous variable.
#ifdef __COUNTER__
#define PP_ANONYMOUS_VARIABLE(str) PP_CONCAT(str, __COUNTER__)
#else
#define PP_ANONYMOUS_VARIABLE(str) PP_CONCAT(str, __LINE__)
#endif

//! Insert prefix based on presence of additional arguments.
#define PP_ONE_OR_NONE(a, ...) PP_THIRD(a, ## __VA_ARGS__, a)
#define PP_THIRD(a, b, ...) __VA_ARGS__

//! \cond Implementation
#define PREPROCESSOR_GEN_H_
#include "preprocessor-gen.h"
#undef PREPROCESSOR_GEN_H_
//! \endcond

/*! \} */
