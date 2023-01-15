#pragma once

/// @file va_args.h
///
/// Some handy macros for preprocessor metaprogramming.

// NOTE: this file has been generated with "./va_args_gen.py", do not edit -- use the generator instead

// DO_NOT_STYLE

#include <util/system/defaults.h>

/**
 * Triggers another level of macro expansion, use whenever passing __VA_ARGS__ to another macro.
 *
 * Used merely for working around an MSVC++ bug.
 * See http://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
 */
#define Y_PASS_VA_ARGS(x) x

/**
 * Count number of arguments in `__VA_ARGS__`.
 * Doesn't work with empty arguments list.
 */
#define Y_COUNT_ARGS(...) Y_PASS_VA_ARGS(__Y_COUNT_ARGS(__VA_ARGS__, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define __Y_COUNT_ARGS(_50, _49, _48, _47, _46, _45, _44, _43, _42, _41, _40, _39, _38, _37, _36, _35, _34, _33, _32, _31, _30, _29, _28, _27, _26, _25, _24, _23, _22, _21, _20, _19, _18, _17, _16, _15, _14, _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3, _2, _1, N, ...) N

/**
 * Get the i-th element from `__VA_ARGS__`.
 */
#define Y_GET_ARG(N, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_GET_ARG_, N))(__VA_ARGS__))
#define __Y_GET_ARG_0(_0, ...) _0
#define __Y_GET_ARG_1(_0, _1, ...) _1
#define __Y_GET_ARG_2(_0, _1, _2, ...) _2
#define __Y_GET_ARG_3(_0, _1, _2, _3, ...) _3
#define __Y_GET_ARG_4(_0, _1, _2, _3, _4, ...) _4
#define __Y_GET_ARG_5(_0, _1, _2, _3, _4, _5, ...) _5
#define __Y_GET_ARG_6(_0, _1, _2, _3, _4, _5, _6, ...) _6
#define __Y_GET_ARG_7(_0, _1, _2, _3, _4, _5, _6, _7, ...) _7
#define __Y_GET_ARG_8(_0, _1, _2, _3, _4, _5, _6, _7, _8, ...) _8
#define __Y_GET_ARG_9(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...) _9
#define __Y_GET_ARG_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, ...) _10
#define __Y_GET_ARG_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, ...) _11
#define __Y_GET_ARG_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, ...) _12
#define __Y_GET_ARG_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, ...) _13
#define __Y_GET_ARG_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, ...) _14
#define __Y_GET_ARG_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, ...) _15
#define __Y_GET_ARG_16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, ...) _16
#define __Y_GET_ARG_17(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, ...) _17
#define __Y_GET_ARG_18(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, ...) _18
#define __Y_GET_ARG_19(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, ...) _19
#define __Y_GET_ARG_20(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, ...) _20
#define __Y_GET_ARG_21(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, ...) _21
#define __Y_GET_ARG_22(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, ...) _22
#define __Y_GET_ARG_23(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, ...) _23
#define __Y_GET_ARG_24(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, ...) _24
#define __Y_GET_ARG_25(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, ...) _25
#define __Y_GET_ARG_26(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, ...) _26
#define __Y_GET_ARG_27(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, ...) _27
#define __Y_GET_ARG_28(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, ...) _28
#define __Y_GET_ARG_29(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, ...) _29
#define __Y_GET_ARG_30(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, ...) _30
#define __Y_GET_ARG_31(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, ...) _31
#define __Y_GET_ARG_32(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, ...) _32
#define __Y_GET_ARG_33(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, ...) _33
#define __Y_GET_ARG_34(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, ...) _34
#define __Y_GET_ARG_35(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, ...) _35
#define __Y_GET_ARG_36(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, ...) _36
#define __Y_GET_ARG_37(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, ...) _37
#define __Y_GET_ARG_38(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, ...) _38
#define __Y_GET_ARG_39(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, ...) _39
#define __Y_GET_ARG_40(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, ...) _40
#define __Y_GET_ARG_41(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, ...) _41
#define __Y_GET_ARG_42(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, ...) _42
#define __Y_GET_ARG_43(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, ...) _43
#define __Y_GET_ARG_44(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, ...) _44
#define __Y_GET_ARG_45(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, ...) _45
#define __Y_GET_ARG_46(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, ...) _46
#define __Y_GET_ARG_47(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, ...) _47
#define __Y_GET_ARG_48(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, ...) _48
#define __Y_GET_ARG_49(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, ...) _49
#define __Y_GET_ARG_50(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, ...) _50

/**
 * Expands a macro for each of the variable arguments.
 * Doesn't work with empty arguments list.
 */
#define Y_MAP_ARGS(ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_MAP_ARGS_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_0(...)
#define __Y_MAP_ARGS_1(ACTION, x, ...) ACTION(x)
#define __Y_MAP_ARGS_2(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_1(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_3(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_2(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_4(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_3(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_5(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_4(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_6(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_5(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_7(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_6(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_8(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_7(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_9(ACTION, x, ...) \
    ACTION(x)                          \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_8(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_10(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_9(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_11(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_10(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_12(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_11(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_13(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_12(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_14(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_13(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_15(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_14(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_16(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_15(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_17(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_16(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_18(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_17(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_19(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_18(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_20(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_19(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_21(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_20(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_22(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_21(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_23(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_22(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_24(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_23(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_25(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_24(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_26(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_25(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_27(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_26(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_28(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_27(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_29(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_28(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_30(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_29(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_31(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_30(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_32(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_31(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_33(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_32(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_34(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_33(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_35(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_34(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_36(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_35(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_37(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_36(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_38(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_37(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_39(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_38(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_40(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_39(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_41(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_40(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_42(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_41(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_43(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_42(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_44(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_43(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_45(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_44(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_46(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_45(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_47(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_46(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_48(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_47(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_49(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_48(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_50(ACTION, x, ...) \
    ACTION(x)                           \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_49(ACTION, __VA_ARGS__))

/**
 * Expands a macro for each of the variable arguments with it's sequence number and value.
 * Corresponding sequence numbers will expand in descending order.
 * Doesn't work with empty arguments list.
 */
#define Y_MAP_ARGS_N(ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_MAP_ARGS_N_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_0(...)
#define __Y_MAP_ARGS_N_1(ACTION, x, ...) ACTION(1, x)
#define __Y_MAP_ARGS_N_2(ACTION, x, ...) \
    ACTION(2, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_1(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_3(ACTION, x, ...) \
    ACTION(3, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_2(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_4(ACTION, x, ...) \
    ACTION(4, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_3(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_5(ACTION, x, ...) \
    ACTION(5, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_4(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_6(ACTION, x, ...) \
    ACTION(6, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_5(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_7(ACTION, x, ...) \
    ACTION(7, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_6(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_8(ACTION, x, ...) \
    ACTION(8, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_7(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_9(ACTION, x, ...) \
    ACTION(9, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_8(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_10(ACTION, x, ...) \
    ACTION(10, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_9(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_11(ACTION, x, ...) \
    ACTION(11, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_10(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_12(ACTION, x, ...) \
    ACTION(12, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_11(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_13(ACTION, x, ...) \
    ACTION(13, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_12(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_14(ACTION, x, ...) \
    ACTION(14, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_13(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_15(ACTION, x, ...) \
    ACTION(15, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_14(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_16(ACTION, x, ...) \
    ACTION(16, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_15(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_17(ACTION, x, ...) \
    ACTION(17, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_16(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_18(ACTION, x, ...) \
    ACTION(18, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_17(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_19(ACTION, x, ...) \
    ACTION(19, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_18(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_20(ACTION, x, ...) \
    ACTION(20, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_19(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_21(ACTION, x, ...) \
    ACTION(21, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_20(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_22(ACTION, x, ...) \
    ACTION(22, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_21(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_23(ACTION, x, ...) \
    ACTION(23, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_22(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_24(ACTION, x, ...) \
    ACTION(24, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_23(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_25(ACTION, x, ...) \
    ACTION(25, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_24(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_26(ACTION, x, ...) \
    ACTION(26, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_25(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_27(ACTION, x, ...) \
    ACTION(27, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_26(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_28(ACTION, x, ...) \
    ACTION(28, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_27(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_29(ACTION, x, ...) \
    ACTION(29, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_28(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_30(ACTION, x, ...) \
    ACTION(30, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_29(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_31(ACTION, x, ...) \
    ACTION(31, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_30(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_32(ACTION, x, ...) \
    ACTION(32, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_31(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_33(ACTION, x, ...) \
    ACTION(33, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_32(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_34(ACTION, x, ...) \
    ACTION(34, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_33(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_35(ACTION, x, ...) \
    ACTION(35, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_34(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_36(ACTION, x, ...) \
    ACTION(36, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_35(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_37(ACTION, x, ...) \
    ACTION(37, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_36(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_38(ACTION, x, ...) \
    ACTION(38, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_37(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_39(ACTION, x, ...) \
    ACTION(39, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_38(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_40(ACTION, x, ...) \
    ACTION(40, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_39(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_41(ACTION, x, ...) \
    ACTION(41, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_40(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_42(ACTION, x, ...) \
    ACTION(42, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_41(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_43(ACTION, x, ...) \
    ACTION(43, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_42(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_44(ACTION, x, ...) \
    ACTION(44, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_43(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_45(ACTION, x, ...) \
    ACTION(45, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_44(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_46(ACTION, x, ...) \
    ACTION(46, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_45(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_47(ACTION, x, ...) \
    ACTION(47, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_46(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_48(ACTION, x, ...) \
    ACTION(48, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_47(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_49(ACTION, x, ...) \
    ACTION(49, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_48(ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_N_50(ACTION, x, ...) \
    ACTION(50, x)                         \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_49(ACTION, __VA_ARGS__))

/**
 * Expands a macro for each of the variable arguments.
 * Doesn't work with empty arguments list.
 */
#define Y_MAP_ARGS_WITH_LAST(ACTION, LAST_ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_MAP_ARGS_WITH_LAST_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_0(...)
#define __Y_MAP_ARGS_WITH_LAST_1(ACTION, LAST_ACTION, x, ...) LAST_ACTION(x)
#define __Y_MAP_ARGS_WITH_LAST_2(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_1(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_3(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_2(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_4(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_3(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_5(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_4(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_6(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_5(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_7(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_6(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_8(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_7(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_9(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                 \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_8(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_10(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_9(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_11(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_10(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_12(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_11(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_13(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_12(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_14(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_13(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_15(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_14(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_16(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_15(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_17(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_16(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_18(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_17(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_19(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_18(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_20(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_19(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_21(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_20(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_22(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_21(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_23(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_22(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_24(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_23(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_25(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_24(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_26(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_25(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_27(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_26(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_28(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_27(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_29(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_28(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_30(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_29(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_31(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_30(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_32(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_31(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_33(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_32(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_34(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_33(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_35(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_34(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_36(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_35(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_37(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_36(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_38(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_37(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_39(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_38(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_40(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_39(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_41(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_40(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_42(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_41(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_43(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_42(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_44(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_43(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_45(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_44(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_46(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_45(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_47(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_46(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_48(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_47(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_49(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_48(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_50(ACTION, LAST_ACTION, x, ...) \
    ACTION(x)                                                  \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_49(ACTION, LAST_ACTION, __VA_ARGS__))

/**
 * Expands a macro for each of the variable arguments with it's sequence number and value.
 * Corresponding sequence numbers will expand in descending order.
 * Doesn't work with empty arguments list.
 */
#define Y_MAP_ARGS_WITH_LAST_N(ACTION, LAST_ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_MAP_ARGS_WITH_LAST_N_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_0(...)
#define __Y_MAP_ARGS_WITH_LAST_N_1(ACTION, LAST_ACTION, x, ...) LAST_ACTION(1, x)
#define __Y_MAP_ARGS_WITH_LAST_N_2(ACTION, LAST_ACTION, x, ...) \
    ACTION(2, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_1(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_3(ACTION, LAST_ACTION, x, ...) \
    ACTION(3, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_2(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_4(ACTION, LAST_ACTION, x, ...) \
    ACTION(4, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_3(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_5(ACTION, LAST_ACTION, x, ...) \
    ACTION(5, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_4(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_6(ACTION, LAST_ACTION, x, ...) \
    ACTION(6, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_5(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_7(ACTION, LAST_ACTION, x, ...) \
    ACTION(7, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_6(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_8(ACTION, LAST_ACTION, x, ...) \
    ACTION(8, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_7(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_9(ACTION, LAST_ACTION, x, ...) \
    ACTION(9, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_8(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_10(ACTION, LAST_ACTION, x, ...) \
    ACTION(10, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_9(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_11(ACTION, LAST_ACTION, x, ...) \
    ACTION(11, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_10(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_12(ACTION, LAST_ACTION, x, ...) \
    ACTION(12, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_11(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_13(ACTION, LAST_ACTION, x, ...) \
    ACTION(13, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_12(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_14(ACTION, LAST_ACTION, x, ...) \
    ACTION(14, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_13(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_15(ACTION, LAST_ACTION, x, ...) \
    ACTION(15, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_14(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_16(ACTION, LAST_ACTION, x, ...) \
    ACTION(16, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_15(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_17(ACTION, LAST_ACTION, x, ...) \
    ACTION(17, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_16(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_18(ACTION, LAST_ACTION, x, ...) \
    ACTION(18, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_17(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_19(ACTION, LAST_ACTION, x, ...) \
    ACTION(19, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_18(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_20(ACTION, LAST_ACTION, x, ...) \
    ACTION(20, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_19(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_21(ACTION, LAST_ACTION, x, ...) \
    ACTION(21, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_20(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_22(ACTION, LAST_ACTION, x, ...) \
    ACTION(22, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_21(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_23(ACTION, LAST_ACTION, x, ...) \
    ACTION(23, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_22(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_24(ACTION, LAST_ACTION, x, ...) \
    ACTION(24, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_23(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_25(ACTION, LAST_ACTION, x, ...) \
    ACTION(25, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_24(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_26(ACTION, LAST_ACTION, x, ...) \
    ACTION(26, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_25(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_27(ACTION, LAST_ACTION, x, ...) \
    ACTION(27, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_26(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_28(ACTION, LAST_ACTION, x, ...) \
    ACTION(28, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_27(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_29(ACTION, LAST_ACTION, x, ...) \
    ACTION(29, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_28(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_30(ACTION, LAST_ACTION, x, ...) \
    ACTION(30, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_29(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_31(ACTION, LAST_ACTION, x, ...) \
    ACTION(31, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_30(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_32(ACTION, LAST_ACTION, x, ...) \
    ACTION(32, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_31(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_33(ACTION, LAST_ACTION, x, ...) \
    ACTION(33, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_32(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_34(ACTION, LAST_ACTION, x, ...) \
    ACTION(34, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_33(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_35(ACTION, LAST_ACTION, x, ...) \
    ACTION(35, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_34(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_36(ACTION, LAST_ACTION, x, ...) \
    ACTION(36, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_35(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_37(ACTION, LAST_ACTION, x, ...) \
    ACTION(37, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_36(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_38(ACTION, LAST_ACTION, x, ...) \
    ACTION(38, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_37(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_39(ACTION, LAST_ACTION, x, ...) \
    ACTION(39, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_38(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_40(ACTION, LAST_ACTION, x, ...) \
    ACTION(40, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_39(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_41(ACTION, LAST_ACTION, x, ...) \
    ACTION(41, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_40(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_42(ACTION, LAST_ACTION, x, ...) \
    ACTION(42, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_41(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_43(ACTION, LAST_ACTION, x, ...) \
    ACTION(43, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_42(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_44(ACTION, LAST_ACTION, x, ...) \
    ACTION(44, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_43(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_45(ACTION, LAST_ACTION, x, ...) \
    ACTION(45, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_44(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_46(ACTION, LAST_ACTION, x, ...) \
    ACTION(46, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_45(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_47(ACTION, LAST_ACTION, x, ...) \
    ACTION(47, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_46(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_48(ACTION, LAST_ACTION, x, ...) \
    ACTION(48, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_47(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_49(ACTION, LAST_ACTION, x, ...) \
    ACTION(49, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_48(ACTION, LAST_ACTION, __VA_ARGS__))
#define __Y_MAP_ARGS_WITH_LAST_N_50(ACTION, LAST_ACTION, x, ...) \
    ACTION(50, x)                                                \
    Y_PASS_VA_ARGS(__Y_MAP_ARGS_WITH_LAST_N_49(ACTION, LAST_ACTION, __VA_ARGS__))

/**
 * Get all elements but the last one from `__VA_ARGS__`.
 * Doesn't work with empty arguments list.
 */
#define Y_ALL_BUT_LAST(...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_ALL_BUT_LAST_, Y_COUNT_ARGS(__VA_ARGS__)))(__VA_ARGS__))
#define __Y_ALL_BUT_LAST_0(...)
#define __Y_ALL_BUT_LAST_1(...)
#define __Y_ALL_BUT_LAST_2(_0, ...) _0
#define __Y_ALL_BUT_LAST_3(_0, _1, ...) _0, _1
#define __Y_ALL_BUT_LAST_4(_0, _1, _2, ...) _0, _1, _2
#define __Y_ALL_BUT_LAST_5(_0, _1, _2, _3, ...) _0, _1, _2, _3
#define __Y_ALL_BUT_LAST_6(_0, _1, _2, _3, _4, ...) _0, _1, _2, _3, _4
#define __Y_ALL_BUT_LAST_7(_0, _1, _2, _3, _4, _5, ...) _0, _1, _2, _3, _4, _5
#define __Y_ALL_BUT_LAST_8(_0, _1, _2, _3, _4, _5, _6, ...) _0, _1, _2, _3, _4, _5, _6
#define __Y_ALL_BUT_LAST_9(_0, _1, _2, _3, _4, _5, _6, _7, ...) _0, _1, _2, _3, _4, _5, _6, _7
#define __Y_ALL_BUT_LAST_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8
#define __Y_ALL_BUT_LAST_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9
#define __Y_ALL_BUT_LAST_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10
#define __Y_ALL_BUT_LAST_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11
#define __Y_ALL_BUT_LAST_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12
#define __Y_ALL_BUT_LAST_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13
#define __Y_ALL_BUT_LAST_16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14
#define __Y_ALL_BUT_LAST_17(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15
#define __Y_ALL_BUT_LAST_18(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16
#define __Y_ALL_BUT_LAST_19(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17
#define __Y_ALL_BUT_LAST_20(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18
#define __Y_ALL_BUT_LAST_21(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19
#define __Y_ALL_BUT_LAST_22(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20
#define __Y_ALL_BUT_LAST_23(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21
#define __Y_ALL_BUT_LAST_24(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22
#define __Y_ALL_BUT_LAST_25(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23
#define __Y_ALL_BUT_LAST_26(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24
#define __Y_ALL_BUT_LAST_27(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25
#define __Y_ALL_BUT_LAST_28(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26
#define __Y_ALL_BUT_LAST_29(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27
#define __Y_ALL_BUT_LAST_30(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28
#define __Y_ALL_BUT_LAST_31(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29
#define __Y_ALL_BUT_LAST_32(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30
#define __Y_ALL_BUT_LAST_33(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31
#define __Y_ALL_BUT_LAST_34(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32
#define __Y_ALL_BUT_LAST_35(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33
#define __Y_ALL_BUT_LAST_36(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34
#define __Y_ALL_BUT_LAST_37(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35
#define __Y_ALL_BUT_LAST_38(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36
#define __Y_ALL_BUT_LAST_39(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37
#define __Y_ALL_BUT_LAST_40(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38
#define __Y_ALL_BUT_LAST_41(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39
#define __Y_ALL_BUT_LAST_42(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40
#define __Y_ALL_BUT_LAST_43(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41
#define __Y_ALL_BUT_LAST_44(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42
#define __Y_ALL_BUT_LAST_45(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43
#define __Y_ALL_BUT_LAST_46(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44
#define __Y_ALL_BUT_LAST_47(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45
#define __Y_ALL_BUT_LAST_48(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46
#define __Y_ALL_BUT_LAST_49(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47
#define __Y_ALL_BUT_LAST_50(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, ...) _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48

/**
 * Get the last element from `__VA_ARGS__`.
 * Doesn't work with empty arguments list.
 */
#define Y_LAST(...) Y_PASS_VA_ARGS(Y_GET_ARG(Y_COUNT_ARGS(__VA_ARGS__), , __VA_ARGS__, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ))

/**
 * Macros for implementing overload by number of arguments.
 *
 * Example usage:
 *
 * @code{cpp}
 * #define I1(arg1) Cout << Y_STRINGIZE(arg1) << Endl;
 * #define I2(arg1, arg2) Cout << Y_STRINGIZE(arg1) << ';' << Y_STRINGIZE(arg2) << Endl;
 *
 * #define Y_PRINT(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, I2, I1)(__VA_ARGS__))
 * @endcode
 */
/// @{
#define Y_MACRO_IMPL_DISPATCHER_2(_0, _1, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_3(_0, _1, _2, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_4(_0, _1, _2, _3, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_5(_0, _1, _2, _3, _4, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_6(_0, _1, _2, _3, _4, _5, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_7(_0, _1, _2, _3, _4, _5, _6, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_8(_0, _1, _2, _3, _4, _5, _6, _7, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_9(_0, _1, _2, _3, _4, _5, _6, _7, _8, IMPL, ...) IMPL
#define Y_MACRO_IMPL_DISPATCHER_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, IMPL, ...) IMPL
/// }@
