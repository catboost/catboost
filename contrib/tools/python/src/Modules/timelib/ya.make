PYMODULE(timelib)


NO_COMPILER_WARNINGS()

ADDINCLSELF()
ADDINCL(contrib/tools/python/src/Modules/timelib/ext-date-lib)

ALLOCATOR(LF)

SRCS(
	timelib.c
	ext-date-lib/astro.c
	ext-date-lib/dow.c
	ext-date-lib/parse_date.c
	ext-date-lib/parse_tz.c
	ext-date-lib/timelib.c
	ext-date-lib/tm2unixtime.c
	ext-date-lib/unixtime2tm.c
)

END()
