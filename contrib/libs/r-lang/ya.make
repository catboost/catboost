LIBRARY()

LICENSE(LGPL-2.1-or-later)

# Since this is a header-only LGPL library, dependents are not bound by LGPL
# according to LGPL 2.1 section 5, as clarified in r-source doc/COPYRIGHTS.



ADDINCL(
    GLOBAL contrib/libs/r-lang
)

END()
