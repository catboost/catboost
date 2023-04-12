Chardet: The Universal Character Encoding Detector
--------------------------------------------------

.. image:: https://img.shields.io/travis/chardet/chardet/stable.svg
   :alt: Build status
   :target: https://travis-ci.org/chardet/chardet

.. image:: https://img.shields.io/coveralls/chardet/chardet/stable.svg
   :target: https://coveralls.io/r/chardet/chardet

.. image:: https://img.shields.io/pypi/v/chardet.svg
   :target: https://warehouse.python.org/project/chardet/
   :alt: Latest version on PyPI

.. image:: https://img.shields.io/pypi/l/chardet.svg
   :alt: License


Detects
 - ASCII, UTF-8, UTF-16 (2 variants), UTF-32 (4 variants)
 - Big5, GB2312, EUC-TW, HZ-GB-2312, ISO-2022-CN (Traditional and Simplified Chinese)
 - EUC-JP, SHIFT_JIS, CP932, ISO-2022-JP (Japanese)
 - EUC-KR, ISO-2022-KR, Johab (Korean)
 - KOI8-R, MacCyrillic, IBM855, IBM866, ISO-8859-5, windows-1251 (Cyrillic)
 - ISO-8859-5, windows-1251 (Bulgarian)
 - ISO-8859-1, windows-1252, MacRoman (Western European languages)
 - ISO-8859-7, windows-1253 (Greek)
 - ISO-8859-8, windows-1255 (Visual and Logical Hebrew)
 - TIS-620 (Thai)

.. note::
   Our ISO-8859-2 and windows-1250 (Hungarian) probers have been temporarily
   disabled until we can retrain the models.

Requires Python 3.7+.

Installation
------------

Install from `PyPI <https://pypi.org/project/chardet/>`_::

    pip install chardet

Documentation
-------------

For users, docs are now available at https://chardet.readthedocs.io/.

Command-line Tool
-----------------

chardet comes with a command-line script which reports on the encodings of one
or more files::

    % chardetect somefile someotherfile
    somefile: windows-1252 with confidence 0.5
    someotherfile: ascii with confidence 1.0

About
-----

This is a continuation of Mark Pilgrim's excellent original chardet port from C, and `Ian Cordasco <https://github.com/sigmavirus24>`_'s
`charade <https://github.com/sigmavirus24/charade>`_ Python 3-compatible fork.

:maintainer: Dan Blanchard
