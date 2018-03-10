LIBRARY(dateutil)

LICENSE(
    BSD
)



PEERDIR(
    contrib/python/six
)

PY_SRCS(
    TOP_LEVEL
    dateutil/__init__.py
    dateutil/_common.py
    dateutil/_version.py
    dateutil/easter.py
    dateutil/parser.py
    dateutil/relativedelta.py
    dateutil/rrule.py
    dateutil/tz/__init__.py
    dateutil/tz/_common.py
    dateutil/tz/tz.py
    dateutil/tz/win.py
    dateutil/tzwin.py
    dateutil/zoneinfo/__init__.py
    dateutil/zoneinfo/rebuild.py
)

NO_LINT()

END()
