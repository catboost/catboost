"""
get_terminal_size() -- return width and height of terminal as a tuple

code from:
http://stackoverflow.com/questions/566746/how-to-get-console- window-width-in-
python

written by
Harco Kuppens (http://stackoverflow.com/users/825214/harco-kuppens)

It is mentioned in the stackoverflow response that this code works
on linux, os x, windows and cygwin (windows).
"""
from __future__ import print_function

import os
import shutil
import subprocess

from pandas.compat import PY3

__all__ = ['get_terminal_size', 'is_terminal']


def get_terminal_size():
    """
    Detect terminal size and return tuple = (width, height).

    Only to be used when running in a terminal. Note that the IPython notebook,
    IPython zmq frontends, or IDLE do not run in a terminal,
    """
    import platform

    if PY3:
        return shutil.get_terminal_size()

    current_os = platform.system()
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _get_terminal_size_windows()
        if tuple_xy is None:
            tuple_xy = _get_terminal_size_tput()
            # needed for window's python in cygwin's xterm!
    if (current_os == 'Linux' or current_os == 'Darwin' or
            current_os.startswith('CYGWIN')):
        tuple_xy = _get_terminal_size_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)      # default value
    return tuple_xy


def is_terminal():
    """
    Detect if Python is running in a terminal.

    Returns True if Python is running in a terminal or False if not.
    """
    try:
        ip = get_ipython()
    except NameError:  # assume standard Python interpreter in a terminal
        return True
    else:
        if hasattr(ip, 'kernel'):  # IPython as a Jupyter kernel
            return False
        else:  # IPython in a terminal
            return True


def _get_terminal_size_windows():

    try:
        from ctypes import windll, create_string_buffer

        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12

        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    except (AttributeError, ValueError):
        return None
    if res:
        import struct
        (bufx, bufy, curx, cury, wattr, left, top, right, bottom, maxx,
         maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)
        sizex = right - left + 1
        sizey = bottom - top + 1
        return sizex, sizey
    else:
        return None


def _get_terminal_size_tput():
    # get terminal width
    # src: http://stackoverflow.com/questions/263890/how-do-i-find-the-width
    # -height-of-a-terminal-window

    try:
        proc = subprocess.Popen(["tput", "cols"],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        output_cols = proc.communicate(input=None)
        proc = subprocess.Popen(["tput", "lines"],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        output_rows = proc.communicate(input=None)
    except OSError:
        return None

    try:
        # Some terminals (e.g. spyder) may report a terminal size of '',
        # making the `int` fail.

        cols = int(output_cols[0])
        rows = int(output_rows[0])
        return cols, rows
    except (ValueError, IndexError):
        return None


def _get_terminal_size_linux():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl
            import termios
            import struct
            cr = struct.unpack(
                'hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except (struct.error, IOError):
            return None
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except OSError:
            pass
    if not cr or cr == (0, 0):
        try:
            from os import environ as env
            cr = (env['LINES'], env['COLUMNS'])
        except (ValueError, KeyError):
            return None
    return int(cr[1]), int(cr[0])


if __name__ == "__main__":
    sizex, sizey = get_terminal_size()
    print('width = {w} height = {h}'.format(w=sizex, h=sizey))
