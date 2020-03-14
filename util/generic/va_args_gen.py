#!/usr/bin/env python
"""
Generates some handy macros for preprocessor metaprogramming.

"""

from __future__ import print_function

import sys
import textwrap

if sys.version_info >= (3, 0, 0):
    xrange = range


def generate(limit):
    print('#pragma once')
    print(textwrap.dedent('''
        /// @file va_args.h
        ///
        /// Some handy macros for preprocessor metaprogramming.
    '''.rstrip()))
    print('')
    command = ' '.join(sys.argv)
    print('// NOTE: this file has been generated with "{}", do not edit -- use the generator instead'.format(command))
    print('')
    print('// DO_NOT_STYLE')
    print('')
    print('#include <util/system/defaults.h>')
    print('')

    pass_va_args()
    count(limit)
    get_elem(limit)
    map_args(limit)
    map_args_n(limit)
    map_args_with_last(limit)
    map_args_with_last_n(limit)
    all_but_last(limit)
    last(limit)
    impl_dispatcher()


def pass_va_args():
    print(textwrap.dedent('''
        /**
         * Triggers another level of macro expansion, use whenever passing __VA_ARGS__ to another macro.
         *
         * Used merely for working around an MSVC++ bug.
         * See http://stackoverflow.com/questions/5134523/msvc-doesnt-expand-va-args-correctly
         */
    '''.rstrip()))
    print('#define Y_PASS_VA_ARGS(x) x')


def count(limit):
    print(textwrap.dedent('''
        /**
         * Count number of arguments in `__VA_ARGS__`.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    numbers = ', '.join(map(str, xrange(limit, -1, -1)))
    u_numbers = ', '.join(map('_{}'.format, xrange(limit, 0, -1)))
    print('#define Y_COUNT_ARGS(...) Y_PASS_VA_ARGS('
          '__Y_COUNT_ARGS(__VA_ARGS__, {}))'.format(numbers))
    print('#define __Y_COUNT_ARGS({}, N, ...) N'.format(u_numbers))


def get_elem(limit):
    print(textwrap.dedent('''
        /**
         * Get the i-th element from `__VA_ARGS__`.
         */
    '''.rstrip()))
    print('#define Y_GET_ARG(N, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_GET_ARG_, '
          'N))(__VA_ARGS__))')
    for i in xrange(0, limit + 1):
        args = ', '.join(map('_{}'.format, xrange(i + 1)))
        print('#define __Y_GET_ARG_{}({}, ...) _{}'.format(i, args, i))


def map_args(limit):
    print(textwrap.dedent('''
        /**
         * Expands a macro for each of the variable arguments.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    print('#define Y_MAP_ARGS(ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT('
          '__Y_MAP_ARGS_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, __VA_ARGS__))')
    print('#define __Y_MAP_ARGS_0(...)')
    print('#define __Y_MAP_ARGS_1(ACTION, x, ...) ACTION(x)')
    for i in xrange(2, limit + 1):
        print('#define __Y_MAP_ARGS_{}(ACTION, x, ...) ACTION(x) Y_PASS_VA_ARGS(__Y_MAP_ARGS_{}('
              'ACTION, __VA_ARGS__))'.format(i, i - 1))


def map_args_n(limit):
    print(textwrap.dedent('''
        /**
         * Expands a macro for each of the variable arguments with it's sequence number and value.
         * Corresponding sequence numbers will expand in descending order.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    print('#define Y_MAP_ARGS_N(ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT('
          '__Y_MAP_ARGS_N_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, __VA_ARGS__))')
    print('#define __Y_MAP_ARGS_N_0(...)')
    print('#define __Y_MAP_ARGS_N_1(ACTION, x, ...) ACTION(1, x)')
    for i in xrange(2, limit + 1):
        print('#define __Y_MAP_ARGS_N_{}(ACTION, x, ...) ACTION({}, x) Y_PASS_VA_ARGS(__Y_MAP_ARGS_N_{}('
              'ACTION, __VA_ARGS__))'.format(i, i, i - 1))


def map_args_with_last(limit):
    print(textwrap.dedent('''
        /**
         * Expands a macro for each of the variable arguments.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    print('#define Y_MAP_ARGS_WITH_LAST(ACTION, LAST_ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS('
          'Y_CAT(__Y_MAP_ARGS_WITH_LAST_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, LAST_ACTION, '
          '__VA_ARGS__))')
    print('#define __Y_MAP_ARGS_WITH_LAST_0(...)')
    print('#define __Y_MAP_ARGS_WITH_LAST_1(ACTION, LAST_ACTION, x, ...) LAST_ACTION(x)')
    for i in xrange(2, limit + 1):
        print('#define __Y_MAP_ARGS_WITH_LAST_{}(ACTION, LAST_ACTION, x, ...) ACTION(x) Y_PASS_VA_ARGS('
              '__Y_MAP_ARGS_WITH_LAST_{}(ACTION, LAST_ACTION, __VA_ARGS__))'.format(i, i - 1))


def map_args_with_last_n(limit):
    print(textwrap.dedent('''
        /**
         * Expands a macro for each of the variable arguments with it's sequence number and value.
         * Corresponding sequence numbers will expand in descending order.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    print('#define Y_MAP_ARGS_WITH_LAST_N(ACTION, LAST_ACTION, ...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS('
          'Y_CAT(__Y_MAP_ARGS_WITH_LAST_N_, Y_COUNT_ARGS(__VA_ARGS__)))(ACTION, LAST_ACTION, '
          '__VA_ARGS__))')
    print('#define __Y_MAP_ARGS_WITH_LAST_N_0(...)')
    print('#define __Y_MAP_ARGS_WITH_LAST_N_1(ACTION, LAST_ACTION, x, ...) LAST_ACTION(1, x)')
    for i in xrange(2, limit + 1):
        print('#define __Y_MAP_ARGS_WITH_LAST_N_{}(ACTION, LAST_ACTION, x, ...) ACTION({}, x) Y_PASS_VA_ARGS('
              '__Y_MAP_ARGS_WITH_LAST_N_{}(ACTION, LAST_ACTION, __VA_ARGS__))'.format(i, i, i - 1))


def all_but_last(limit):
    print(textwrap.dedent('''
        /**
         * Get all elements but the last one from `__VA_ARGS__`.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    print('#define Y_ALL_BUT_LAST(...) Y_PASS_VA_ARGS(Y_PASS_VA_ARGS(Y_CAT(__Y_ALL_BUT_LAST_, '
          'Y_COUNT_ARGS(__VA_ARGS__)))(__VA_ARGS__))')
    print('#define __Y_ALL_BUT_LAST_0(...)')
    print('#define __Y_ALL_BUT_LAST_1(...)')
    for i in xrange(2, limit + 1):
        args = ', '.join(map('_{}'.format, xrange(i - 1)))
        print('#define __Y_ALL_BUT_LAST_{}({}, ...) {}'.format(i, args, args))


def last(limit):
    print(textwrap.dedent('''
        /**
         * Get the last element from `__VA_ARGS__`.
         * Doesn't work with empty arguments list.
         */
    '''.rstrip()))
    print('#define Y_LAST(...) Y_PASS_VA_ARGS('
          'Y_GET_ARG(Y_COUNT_ARGS(__VA_ARGS__), , __VA_ARGS__, {}))'.format(',' * limit))


def impl_dispatcher():
    print(textwrap.dedent('''
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
    '''.rstrip()))
    print('/// @{')
    for i in xrange(2, 11):
        args = ', '.join(map('_{}'.format, xrange(i)))
        print('#define Y_MACRO_IMPL_DISPATCHER_{}({}, IMPL, ...) IMPL'.format(i, args))
    print('/// }@')


def main():
    if len(sys.argv) > 2:
        sys.stderr.write('Usage: {} [limit=50]\n'.format(sys.argv[0]))
        sys.exit(1)
    limit = 50
    if len(sys.argv) == 2:
        limit = int(sys.argv[1])
    generate(limit)


if __name__ == '__main__':
    main()
