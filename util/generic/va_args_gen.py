#!/usr/bin/env python

import sys


def generate(limit):
    print '''#pragma once

// NOTE: this file has been generated with "{command}", do not edit - use the generator instead

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
#define Y_MAP_ARGS_WITH_LAST(ACTION, LAST_ACTION, ...) \\
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

// Implementation details follow'''.format(command=' '.join(sys.argv))

    print '#define __APPLY_1__(MACRO, x) MACRO(x)'
    for depth in xrange(2, limit + 1):
        print '#define __APPLY_{}__(MACRO, x, ...) \\\n\
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_{}__(MACRO, __VA_ARGS__))'.format(
            depth, depth - 1
        )
    print '// ...'

    print '#define __APPLY_WITH_LAST_1__(MACRO, LAST_MACRO, x) LAST_MACRO(x)'
    for depth in xrange(2, limit + 1):
        print '#define __APPLY_WITH_LAST_{}__(MACRO, LAST_MACRO, x, ...) \\\n\
    MACRO(x) Y_PASS_VA_ARGS(__APPLY_WITH_LAST_{}__(MACRO, LAST_MACRO, __VA_ARGS__))'.format(
            depth, depth - 1
        )
    print '// ...'

    print '#define __GET_MACRO__({}, MACRO, ...) MACRO'.format(
        ', \\\n    '.join(map(lambda x: '_' + str(x), xrange(1, limit + 1)))
    )

    print '#define __MAP_ARGS_IMPL__(MACRO, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(__GET_MACRO__(__VA_ARGS__, \\'
    for depth in xrange(limit, 1, -1):
        print '    __APPLY_{}__, \\'.format(depth)
    print '    __APPLY_1__))(MACRO, __VA_ARGS__))'

    print '#define __MAP_ARGS_WITH_LAST_IMPL__(MACRO, LAST_MACRO, ...) ' + \
        'Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(__GET_MACRO__(__VA_ARGS__, \\'
    for depth in xrange(limit, 1, -1):
        print '    __APPLY_WITH_LAST_{}__, \\'.format(depth)
    print '    __APPLY_WITH_LAST_1__))(MACRO, LAST_MACRO, __VA_ARGS__))'


def main():
    if len(sys.argv) > 2:
        print >>sys.stderr, 'Usage: {} [limit=10]'.format(sys.argv[0])
        sys.exit(1)
    limit = 10
    if len(sys.argv) == 2:
        limit = int(sys.argv[1])
    generate(limit)


if __name__ == '__main__':
    main()
