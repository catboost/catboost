import os

ref_dir = os.path.join(os.path.dirname(__file__))

__all__ = [
    "constants",
    "ufuncs",
]

for f in __all__:
    __import__(__name__ + '.' + f)

del f, ref_dir

__doc__ = """\
Topical documentation
=====================

The following topics are available:
%s

You can view them by

>>> help(np.doc.TOPIC)                                      #doctest: +SKIP

""" % '\n- '.join([''] + __all__)

__all__.extend(['__doc__'])
