# coding=utf-8
# Copyright (c) 2008-2011 Volvox Development Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Author: Konstantin Lepa <konstantin.lepa@gmail.com>

"""ANSII Color formatting for output in terminal."""

from __future__ import print_function

import os

__ALL__ = ['colored', 'cprint', 'get_code_by_spec']

VERSION = (1, 1, 0)

ATTRIBUTES = dict(
    list(zip(
        [
            'bold',
            'dark',
            '',
            'underline',
            'blink',
            '',
            'reverse',
            'concealed'
        ], list(range(1, 9)))))
del ATTRIBUTES['']

ATTRIBUTES['light'] = ATTRIBUTES['bold']


HIGHLIGHTS = dict(
    list(zip([
        'on_grey',
        'on_red',
        'on_green',
        'on_yellow',
        'on_blue',
        'on_magenta',
        'on_cyan',
        'on_white'
    ], list(range(40, 48)))))

HIGHLIGHTS['on_gray'] = HIGHLIGHTS['on_grey']
HIGHLIGHTS['on_purple'] = HIGHLIGHTS['on_magenta']

COLORS = dict(
    list(zip([
        'grey',
        'red',
        'green',
        'yellow',
        'blue',
        'magenta',
        'cyan',
        'white',
    ], list(range(30, 38)))))

COLORS['gray'] = COLORS['grey']
COLORS['purple'] = COLORS['magenta']

COLORS['reset'] = 0


def get_code(code):
    if os.getenv('ANSI_COLORS_DISABLED') is None:
        return "\033[{}m".format(code)
    return ""


def get_code_by_spec(color_spec):
    color, on_color, attrs = get_spec(color_spec)
    return get_color(color, on_color, attrs)


def get_color(color, on_color, attrs):
    res = ''

    if color is not None:
        res += get_code(COLORS[color])

    if on_color is not None:
        res += get_code(HIGHLIGHTS[on_color])

    if attrs is not None:
        for attr in attrs:
            res += get_code(ATTRIBUTES[attr])

    return res


def colored(text, color=None, on_color=None, attrs=None):
    """Colorize text.

    Available text colors:
        red, green, yellow, blue, magenta, cyan, white.

    Available text highlights:
        on_red, on_green, on_yellow, on_blue, on_magenta, on_cyan, on_white.

    Available attributes:
        bold, dark, underline, blink, reverse, concealed.

    Example:
        colored('Hello, World!', 'red', 'on_grey', ['blue', 'blink'])
        colored('Hello, World!', 'green')
    """
    return get_color(color, on_color, attrs) + text + get_code(COLORS['reset'])


def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Print colorize text.

    It accepts arguments of print function.
    """

    print((colored(text, color, on_color, attrs)), **kwargs)


def get_spec(color_spec):
    parts = color_spec.split('-')  # we assume following format: 'green-bold-on_red'
    color = None
    on_color = None
    attrs = []
    for part in parts:
        part = part.lower()
        if part in COLORS:
            color = part
        if part in HIGHLIGHTS:
            on_color = part
        if part in ATTRIBUTES:
            attrs.append(part)
    return color, on_color, attrs


def tcolor(text, color_spec):
    color, on_color, attrs = get_spec(color_spec)
    return colored(text, color=color, on_color=on_color, attrs=attrs)


if __name__ == '__main__':
    print('Current terminal type: %s' % os.getenv('TERM'))
    print('Test basic colors:')
    cprint('Grey color', 'grey')
    cprint('Red color', 'red')
    cprint('Green color', 'green')
    cprint('Yellow color', 'yellow')
    cprint('Blue color', 'blue')
    cprint('Magenta color', 'magenta')
    cprint('Cyan color', 'cyan')
    cprint('White color', 'white')
    print(('-' * 78))

    print('Test highlights:')
    cprint('On grey color', on_color='on_grey')
    cprint('On red color', on_color='on_red')
    cprint('On green color', on_color='on_green')
    cprint('On yellow color', on_color='on_yellow')
    cprint('On blue color', on_color='on_blue')
    cprint('On magenta color', on_color='on_magenta')
    cprint('On cyan color', on_color='on_cyan')
    cprint('On white color', color='grey', on_color='on_white')
    print('-' * 78)

    print('Test attributes:')
    cprint('Bold grey color', 'grey', attrs=['bold'])
    cprint('Dark red color', 'red', attrs=['dark'])
    cprint('Underline green color', 'green', attrs=['underline'])
    cprint('Blink yellow color', 'yellow', attrs=['blink'])
    cprint('Reversed blue color', 'blue', attrs=['reverse'])
    cprint('Concealed Magenta color', 'magenta', attrs=['concealed'])
    cprint('Bold underline reverse cyan color', 'cyan', attrs=['bold', 'underline', 'reverse'])
    cprint('Dark blink concealed white color', 'white', attrs=['dark', 'blink', 'concealed'])
    print(('-' * 78))

    print('Test mixing:')
    cprint('Underline red on grey color', 'red', 'on_grey', ['underline'])
    cprint('Reversed green on red color', 'green', 'on_red', ['reverse'])
