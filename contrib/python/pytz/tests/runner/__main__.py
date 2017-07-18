# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import doctest

import pytz

nfailures, ntests = doctest.testmod(pytz)

raise SystemExit(1 if nfailures else 0)
