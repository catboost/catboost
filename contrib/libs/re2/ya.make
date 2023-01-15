LIBRARY()


LICENSE(
    BSD3
)

NO_COMPILER_WARNINGS()

#SRCDIR(
#    contrib/libs/re2
#)

ADDINCL(
    GLOBAL contrib/libs/re2
)

#ENABLE(REGENERATE)

IF (REGENERATE)
    PYTHON(re2/make_unicode_groups.py STDOUT re2/unicode_groups.cc OUTPUT_INCLUDES re2/unicode_groups.h)
    PYTHON(re2/make_unicode_casefold.py STDOUT re2/unicode_casefold.cc OUTPUT_INCLUDES re2/unicode_casefold.h)
ELSE ()
    SRCS(
        re2/unicode_groups.cc
        re2/unicode_casefold.cc
    )
ENDIF ()

SRCS(
    #util/pcre.cc
    util/rune.cc
    util/strutil.cc

    #output.cc
    re2/stringpiece.cc

    re2/bitstate.cc
    re2/compile.cc
    re2/dfa.cc
    re2/filtered_re2.cc
    re2/mimics_pcre.cc
    re2/nfa.cc
    re2/onepass.cc
    re2/parse.cc
    re2/perl_groups.cc
    re2/prefilter.cc
    re2/prefilter_tree.cc
    re2/prog.cc
    re2/re2.cc
    re2/regexp.cc
    re2/set.cc
    re2/simplify.cc
    re2/tostring.cc
)

END()
