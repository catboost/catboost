"""
Some syntax errors are a bit complicated and need exact checking. Here we
gather some of the potentially dangerous ones.
"""

from __future__ import division

# With a dot it's not a future import anymore.
from .__future__ import absolute_import

'' ''
''r''u''
b'' BR''


for x in [1]:
    break
    continue

try:
    pass
except ZeroDivisionError:
    pass
    #: E722:0
except:
    pass

try:
    pass
    #: E722:0 E901:0
except:
    pass
except ZeroDivisionError:
    pass


r'\n'
r'\x'
b'\n'


a = 3


def x(b=a):
    global a
