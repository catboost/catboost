LIBRARY()

LICENSE(
    GPL-2.0-or-later
    GPL-3.0-or-later
    LGPL-2.0-or-later
    LGPL-2.1-only
    LGPL-2.1-or-later
)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(3.5.0)

# Since this is a header-only LGPL library, dependents are not bound by LGPL
# according to LGPL 2.1 section 5, as clarified in r-source doc/COPYRIGHTS.



ADDINCL(GLOBAL contrib/libs/r-lang)

END()
