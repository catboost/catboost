LIBRARY(dateutil)



PEERDIR(
    contrib/python/six
)

PY_SRCS(
    TOP_LEVEL
    dateutil/__init__.py
    dateutil/easter.py
    dateutil/parser.py
    dateutil/relativedelta.py
    dateutil/rrule.py
    dateutil/tz.py
    dateutil/tzwin.py
    dateutil/zoneinfo/__init__.py
)

NO_LINT()

END()
