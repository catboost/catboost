#pragma once

// NOTE: this file has been generated with "./va_args_gen.py", do not edit - use the generator instead

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
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_1__(MACRO, __VA_ARGS__))
#define __APPLY_3__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_2__(MACRO, __VA_ARGS__))
#define __APPLY_4__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_3__(MACRO, __VA_ARGS__))
#define __APPLY_5__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_4__(MACRO, __VA_ARGS__))
#define __APPLY_6__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_5__(MACRO, __VA_ARGS__))
#define __APPLY_7__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_6__(MACRO, __VA_ARGS__))
#define __APPLY_8__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_7__(MACRO, __VA_ARGS__))
#define __APPLY_9__(MACRO, x, ...) \
    MACRO(x)                       \
    Y_PASS_VA_ARGS(__APPLY_8__(MACRO, __VA_ARGS__))
#define __APPLY_10__(MACRO, x, ...) \
    MACRO(x)                        \
    Y_PASS_VA_ARGS(__APPLY_9__(MACRO, __VA_ARGS__))
// ...
#define __APPLY_WITH_LAST_1__(MACRO, LAST_MACRO, x) LAST_MACRO(x)
#define __APPLY_WITH_LAST_2__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_1__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_3__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_2__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_4__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_3__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_5__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_4__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_6__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_5__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_7__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_6__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_8__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_7__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_9__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                             \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_8__(MACRO, LAST_MACRO, __VA_ARGS__))
#define __APPLY_WITH_LAST_10__(MACRO, LAST_MACRO, x, ...) \
    MACRO(x)                                              \
    Y_PASS_VA_ARGS(__APPLY_WITH_LAST_9__(MACRO, LAST_MACRO, __VA_ARGS__))
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
                      _10, MACRO, ...) MACRO
#define __MAP_ARGS_IMPL__(MACRO, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(__GET_MACRO__(__VA_ARGS__,  \
                                                                                  __APPLY_10__, \
                                                                                  __APPLY_9__,  \
                                                                                  __APPLY_8__,  \
                                                                                  __APPLY_7__,  \
                                                                                  __APPLY_6__,  \
                                                                                  __APPLY_5__,  \
                                                                                  __APPLY_4__,  \
                                                                                  __APPLY_3__,  \
                                                                                  __APPLY_2__,  \
                                                                                  __APPLY_1__))(MACRO, __VA_ARGS__))
#define __MAP_ARGS_WITH_LAST_IMPL__(MACRO, LAST_MACRO, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(__GET_MACRO__(__VA_ARGS__,            \
                                                                                                        __APPLY_WITH_LAST_10__, \
                                                                                                        __APPLY_WITH_LAST_9__,  \
                                                                                                        __APPLY_WITH_LAST_8__,  \
                                                                                                        __APPLY_WITH_LAST_7__,  \
                                                                                                        __APPLY_WITH_LAST_6__,  \
                                                                                                        __APPLY_WITH_LAST_5__,  \
                                                                                                        __APPLY_WITH_LAST_4__,  \
                                                                                                        __APPLY_WITH_LAST_3__,  \
                                                                                                        __APPLY_WITH_LAST_2__,  \
                                                                                                        __APPLY_WITH_LAST_1__))(MACRO, LAST_MACRO, __VA_ARGS__))
