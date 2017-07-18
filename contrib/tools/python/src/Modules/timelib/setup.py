#! /usr/bin/env python

import os

try:
    from setuptools import setup, Extension
    extra = dict(zip_safe=False)
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    extra = dict()

from distutils import sysconfig
if sysconfig.get_config_var("LIBM") == "-lm":
    libraries = ["m"]
else:
    libraries = []

sources = ["ext-date-lib/astro.c", "ext-date-lib/dow.c",
           "ext-date-lib/parse_date.c", "ext-date-lib/parse_tz.c",
           "ext-date-lib/timelib.c", "ext-date-lib/tm2unixtime.c",
           "ext-date-lib/unixtime2tm.c", "timelib.c"]

if not os.path.exists("timelib.c"):
    os.system("cython timelib.pyx")

setup(name="timelib",
      version="0.2.4",
      description="parse english textual date descriptions",
      author="Ralf Schmitt",
      author_email="ralf@systemexit.de",
      url="https://github.com/pediapress/timelib/",
      ext_modules=[Extension("timelib", sources=sources,
                             libraries=libraries,
                             define_macros=[("HAVE_STRING_H", 1)])],
      include_dirs=[".", "ext-date-lib"],
      long_description=open("README.rst").read(),
      license="zlib/php",
      **extra)
