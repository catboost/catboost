#!/usr/bin/env python

# Change content of this file to change uids for cython programs - cython 0.29.37 r1

#
#   Cython -- Main Program, generic
#

if __name__ == '__main__':

    import os
    import sys
    sys.dont_write_bytecode = True

    # Make sure we import the right Cython
    cythonpath, _ = os.path.split(os.path.realpath(__file__))
    sys.path.insert(0, cythonpath)

    from Cython.Compiler.Main import main
    main(command_line = 1)

else:
    # Void cython.* directives.
    from Cython.Shadow import *
    ## and bring in the __version__
    from Cython import __version__
    from Cython import load_ipython_extension
