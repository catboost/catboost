#pragma once

// NOTE: this file has been generated with "va_args_gen.py 30", do not edit - use the generator instead

// Used merely for working around an MSVC++ bug:
// http://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
// Triggers another level of macro expansion, use whenever passing __VA_ARGS__ to another macro
#define Y_PASS_VA_ARGS(x) x

// Usage: Y_MAP_ARGS(ACTION, ...) - expands the ACTION(...) macro for each of the variable arguments
#define Y_MAP_ARGS(ACTION, ...) Y_PASS_VA_ARGS(__MAP_ARGS_IMPL__(ACTION, __VA_ARGS__))
// It is possible to adapt for multi-argument ACTIONs to use something like Y_MAP_ARGS(ACTION_PROXY, (1, 2), (3, 4)).
// For that, #define ACTION_PROXY(x) ACTION x - the (1, 2) given to ACTION_PROXY will expand to ACTION (1, 2).

// Usage: Y_MAP_ARGS_WITH_LAST(ACTION, LAST_ACTION, ...) - expands the ACTION(...) macro
// for each except the last of the varargs, for the latter the LAST_ACTION(...) macro is expanded
#define Y_MAP_ARGS_WITH_LAST(ACTION, LAST_ACTION, ...) \
    Y_PASS_VA_ARGS(__MAP_ARGS_WITH_LAST_IMPL__(ACTION, LAST_ACTION, __VA_ARGS__))

/* @def Y_MACRO_IMPL_DISPATCHER_2
 *
 * This macro is intended to use as a helper for macro overload by number of arguments.
 *
 * @code
 * #include <util/system/defaults.h>
 *
 * #define Y_PRINT_IMPL_1(arg1) Cout << Y_STRINGIZE(arg1) << Endl;
 * #define Y_PRINT_IMPL_2(arg1, arg2) Cout << Y_STRINGIZE(arg1) << ';' << Y_STRINGIZE(arg2) << Endl;
 *
 * #define Y_PRINT(...) Y_PASS_VA_ARGS(Y_MACRO_IMPL_DISPATCHER_2(__VA_ARGS__, Y_PRINT_IMPL_2, Y_PRINT_IMPL_1)(__VA_ARGS__))
 * @endcode
 */
#define Y_MACRO_IMPL_DISPATCHER_2(_1, _2, IMPL, ...) IMPL

// Implementation details follow
#define __APPLY_1__(MACRO, x) MACRO(x)
#define __APPLY_2__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_1__(MACRO, __VA_ARGS__))
#define __APPLY_3__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_2__(MACRO, __VA_ARGS__))
#define __APPLY_4__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_3__(MACRO, __VA_ARGS__))
#define __APPLY_5__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_4__(MACRO, __VA_ARGS__))
#define __APPLY_6__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_5__(MACRO, __VA_ARGS__))
#define __APPLY_7__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_6__(MACRO, __VA_ARGS__))
#define __APPLY_8__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_7__(MACRO, __VA_ARGS__))
#define __APPLY_9__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_8__(MACRO, __VA_ARGS__))
#define __APPLY_10__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_9__(MACRO, __VA_ARGS__))
#define __APPLY_11__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_10__(MACRO, __VA_ARGS__))
#define __APPLY_12__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_11__(MACRO, __VA_ARGS__))
#define __APPLY_13__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_12__(MACRO, __VA_ARGS__))
#define __APPLY_14__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_13__(MACRO, __VA_ARGS__))
#define __APPLY_15__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_14__(MACRO, __VA_ARGS__))
#define __APPLY_16__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_15__(MACRO, __VA_ARGS__))
#define __APPLY_17__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_16__(MACRO, __VA_ARGS__))
#define __APPLY_18__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_17__(MACRO, __VA_ARGS__))
#define __APPLY_19__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_18__(MACRO, __VA_ARGS__))
#define __APPLY_20__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_19__(MACRO, __VA_ARGS__))
#define __APPLY_21__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_20__(MACRO, __VA_ARGS__))
#define __APPLY_22__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_21__(MACRO, __VA_ARGS__))
#define __APPLY_23__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_22__(MACRO, __VA_ARGS__))
#define __APPLY_24__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_23__(MACRO, __VA_ARGS__))
#define __APPLY_25__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_24__(MACRO, __VA_ARGS__))
#define __APPLY_26__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_25__(MACRO, __VA_ARGS__))
#define __APPLY_27__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_26__(MACRO, __VA_ARGS__))
#define __APPLY_28__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_27__(MACRO, __VA_ARGS__))
#define __APPLY_29__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_28__(MACRO, __VA_ARGS__))
#define __APPLY_30__(MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_29__(MACRO, __VA_ARGS__))
// ...
#define __APPLY_WITH_LAST_1__(MACRO, LAST_MACRO, x) LAST_MACRO(x)
#define __APPLY_WITH_LAST_2__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_1__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_3__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_2__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_4__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_3__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_5__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_4__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_6__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_5__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_7__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_6__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_8__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_7__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_9__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_8__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_10__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_9__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_11__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_10__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_12__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_11__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_13__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_12__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_14__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_13__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_15__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_14__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_16__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_15__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_17__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_16__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_18__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_17__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_19__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_18__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_20__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_19__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_21__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_20__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_22__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_21__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_23__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_22__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_24__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_23__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_25__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_24__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_26__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_25__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_27__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_26__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_28__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_27__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_29__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_28__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_30__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_29__(MACRO, LAST_MACRO, __VA_ARGS__))
// ...
#define __GET_MACRO__(_1, \
    _2, \
    _3, \
    _4, \
    _5, \
    _6, \
    _7, \
    _8, \
    _9, \
    _10, \
    _11, \
    _12, \
    _13, \
    _14, \
    _15, \
    _16, \
    _17, \
    _18, \
    _19, \
    _20, \
    _21, \
    _22, \
    _23, \
    _24, \
    _25, \
    _26, \
    _27, \
    _28, \
    _29, \
    _30, MACRO, ...) MACRO
#define __MAP_ARGS_IMPL__(MACRO, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(__GET_MACRO__(__VA_ARGS__, \
    __APPLY_30__, \
    __APPLY_29__, \
    __APPLY_28__, \
    __APPLY_27__, \
    __APPLY_26__, \
    __APPLY_25__, \
    __APPLY_24__, \
    __APPLY_23__, \
    __APPLY_22__, \
    __APPLY_21__, \
    __APPLY_20__, \
    __APPLY_19__, \
    __APPLY_18__, \
    __APPLY_17__, \
    __APPLY_16__, \
    __APPLY_15__, \
    __APPLY_14__, \
    __APPLY_13__, \
    __APPLY_12__, \
    __APPLY_11__, \
    __APPLY_10__, \
    __APPLY_9__, \
    __APPLY_8__, \
    __APPLY_7__, \
    __APPLY_6__, \
    __APPLY_5__, \
    __APPLY_4__, \
    __APPLY_3__, \
    __APPLY_2__, \
    __APPLY_1__))(MACRO, __VA_ARGS__))
#define __MAP_ARGS_WITH_LAST_IMPL__(MACRO, LAST_MACRO, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(__GET_MACRO__(__VA_ARGS__, \
    __APPLY_WITH_LAST_30__, \
    __APPLY_WITH_LAST_29__, \
    __APPLY_WITH_LAST_28__, \
    __APPLY_WITH_LAST_27__, \
    __APPLY_WITH_LAST_26__, \
    __APPLY_WITH_LAST_25__, \
    __APPLY_WITH_LAST_24__, \
    __APPLY_WITH_LAST_23__, \
    __APPLY_WITH_LAST_22__, \
    __APPLY_WITH_LAST_21__, \
    __APPLY_WITH_LAST_20__, \
    __APPLY_WITH_LAST_19__, \
    __APPLY_WITH_LAST_18__, \
    __APPLY_WITH_LAST_17__, \
    __APPLY_WITH_LAST_16__, \
    __APPLY_WITH_LAST_15__, \
    __APPLY_WITH_LAST_14__, \
    __APPLY_WITH_LAST_13__, \
    __APPLY_WITH_LAST_12__, \
    __APPLY_WITH_LAST_11__, \
    __APPLY_WITH_LAST_10__, \
    __APPLY_WITH_LAST_9__, \
    __APPLY_WITH_LAST_8__, \
    __APPLY_WITH_LAST_7__, \
    __APPLY_WITH_LAST_6__, \
    __APPLY_WITH_LAST_5__, \
    __APPLY_WITH_LAST_4__, \
    __APPLY_WITH_LAST_3__, \
    __APPLY_WITH_LAST_2__, \
    __APPLY_WITH_LAST_1__))(MACRO, LAST_MACRO, __VA_ARGS__))
