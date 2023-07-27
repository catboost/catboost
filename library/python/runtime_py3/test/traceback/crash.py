import argparse
import sys
import time

from IPython.core import ultratb

from .mod import modfunc


def one():
    modfunc(two)  # aaa


def two():
    three(42)


def three(x):
    raise RuntimeError(f"Kaboom! I'm dead: {x}")


def main():
    hooks = {
        "default": lambda: sys.excepthook,
        "ultratb_color": lambda: ultratb.ColorTB(ostream=sys.stderr),
        "ultratb_verbose": lambda: ultratb.VerboseTB(ostream=sys.stderr),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("hook", choices=sorted(hooks), default="default")

    args = parser.parse_args()

    sys.excepthook = hooks[args.hook]()

    print("__name__ =", __name__)
    print("__file__ =", __file__)

    time.time = lambda: 1531996624.0  # Freeze time
    sys.executable = "<traceback test>"

    one()
