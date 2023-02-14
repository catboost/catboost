import sys


def termsize(stream=sys.stdout):
    import fcntl
    import struct
    import termios
    rows, cols, _, _ = struct.unpack(
        'HHHH',
        fcntl.ioctl(stream.fileno(),
                    termios.TIOCGWINSZ,
                    struct.pack('HHHH', 0, 0, 0, 0)))
    return rows, cols


def termsize_or_default(stream=sys.stdout, default=(25, 80)):
    try:
        return termsize(stream)
    except Exception:
        return default
