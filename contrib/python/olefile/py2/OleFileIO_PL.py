#!/usr/local/bin/python
# -*- coding: latin-1 -*-
"""
olefile (formerly OleFileIO_PL)

Module to read/write Microsoft OLE2 files (also called Structured Storage or
Microsoft Compound Document File Format), such as Microsoft Office 97-2003
documents, Image Composer and FlashPix files, Outlook messages, ...
This version is compatible with Python 2.6+ and 3.x

Project website: http://www.decalage.info/olefile

olefile is copyright (c) 2005-2015 Philippe Lagadec (http://www.decalage.info)

olefile is based on the OleFileIO module from the PIL library v1.1.6
See: http://www.pythonware.com/products/pil/index.htm

The Python Imaging Library (PIL) is
    Copyright (c) 1997-2005 by Secret Labs AB
    Copyright (c) 1995-2005 by Fredrik Lundh

See source code and LICENSE.txt for information on usage and redistribution.
"""

# The OleFileIO_PL module is for backward compatibility

try:
    # first try to import olefile for Python 2.6+/3.x
    from olefile.olefile import *
    # import metadata not covered by *:
    from olefile.olefile import __version__, __author__, __date__

except:
    # if it fails, fallback to the old version olefile2 for Python 2.x:
    from olefile.olefile2 import *
    # import metadata not covered by *:
    from olefile.olefile2 import __doc__, __version__, __author__, __date__
