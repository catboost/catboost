
/* #line 1 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
#include <algorithm>
#include <cstring>

#ifdef NLP_DEBUG
#   include <util/string/printf.h>
#endif

#include <util/generic/yexception.h>
#include <library/cpp/tokenizer/nlpparser.h>

#ifdef __clang__
    #pragma clang diagnostic ignored "-Wunused-variable"
#endif


/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */



/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
static const int NlpLexer_start = 75;
static const int NlpLexer_first_final = 75;
static const int NlpLexer_error = -1;

static const int NlpLexer_en_main = 75;


/* #line 158 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */


template<> void TVersionedNlpParser<2>::ExecuteImpl(const unsigned char* text, size_t len) {
    Y_ASSERT(text);
    Text = text;
    const unsigned char *p = text, *pe = p + len; // 'pe' must never be dereferenced
    const unsigned char *ts, *te, *eof = pe;
    int act, cs;

    
/* #line 42 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	{
	cs = NlpLexer_start;
	ts = 0;
	te = 0;
	act = 0;
	}

/* #line 168 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
    
/* #line 52 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
tr0:
/* #line 139 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{{p = ((te))-1;}{
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr7:
/* #line 95 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{{p = ((te))-1;}{
        ProcessMultitoken(ts, te);
    }}
	goto st75;
tr209:
/* #line 145 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{{p = ((te))-1;}{
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr217:
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        EnsureSentenceBreak(p);
    }
/* #line 108 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p+1;{
        p = ts + MakeSentenceBreak(ts, te - ts) - 1;
    }}
	goto st75;
tr224:
/* #line 129 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{{p = ((te))-1;}{
#ifdef NLP_DEBUG
        Cdbg << Sprintf("met othermisc at %p\n", (void *)ts);
        Cdbg.write(ts, te - ts);
        Cdbg << Endl;
#endif
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr233:
/* #line 150 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p+1;{
        Y_ASSERT(*ts == 0);
        MakeEntry(ts, 1, NLP_MISCTEXT);
    }}
	goto st75;
tr247:
/* #line 129 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
#ifdef NLP_DEBUG
        Cdbg << Sprintf("met othermisc at %p\n", (void *)ts);
        Cdbg.write(ts, te - ts);
        Cdbg << Endl;
#endif
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr248:
/* #line 123 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr249:
/* #line 139 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr251:
/* #line 112 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        MakeEntry(ts, te - ts, SpacePreserve ? NLP_PARABREAK : NLP_MISCTEXT);
    }}
	goto st75;
tr254:
/* #line 145 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        CancelToken();
        MakeEntry(ts, te - ts, NLP_MISCTEXT);
    }}
	goto st75;
tr256:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 95 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        ProcessMultitoken(ts, te);
    }}
	goto st75;
tr268:
/* #line 95 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        ProcessMultitoken(ts, te);
    }}
	goto st75;
tr319:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 30 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateSuffix(*p);
    }
/* #line 95 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p+1;{
        ProcessMultitoken(ts, te);
    }}
	goto st75;
tr339:
/* #line 30 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateSuffix(*p);
    }
/* #line 95 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p+1;{
        ProcessMultitoken(ts, te);
    }}
	goto st75;
tr362:
/* #line 103 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        ProcessIdeographs(ts, te);
    }}
	goto st75;
tr363:
/* #line 99 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{te = p;p--;{
        ProcessSurrogatePairs(ts, te);
    }}
	goto st75;
st75:
/* #line 1 "NONE" */
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof75;
case 75:
/* #line 1 "NONE" */
	{ts = p;}
/* #line 206 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 0u: goto tr233;
		case 9u: goto st77;
		case 10u: goto tr236;
		case 13u: goto tr237;
		case 32u: goto st77;
		case 34u: goto st77;
		case 46u: goto tr239;
		case 49u: goto tr240;
		case 64u: goto tr238;
		case 65u: goto tr241;
		case 95u: goto st77;
		case 97u: goto tr241;
		case 128u: goto tr242;
		case 129u: goto tr241;
		case 143u: goto st77;
		case 159u: goto st135;
		case 160u: goto st77;
		case 167u: goto st77;
		case 169u: goto tr244;
		case 176u: goto st77;
		case 182u: goto st77;
		case 183u: goto tr244;
		case 186u: goto tr246;
		case 187u: goto tr239;
		case 192u: goto st77;
	}
	if ( (*p) < 178u ) {
		if ( (*p) < 37u ) {
			if ( 35u <= (*p) && (*p) <= 36u )
				goto tr238;
		} else if ( (*p) > 39u ) {
			if ( 42u <= (*p) && (*p) <= 47u )
				goto st77;
		} else
			goto st77;
	} else if ( (*p) > 179u ) {
		if ( (*p) < 189u ) {
			if ( 180u <= (*p) && (*p) <= 181u )
				goto st136;
		} else if ( (*p) > 190u ) {
			if ( 208u <= (*p) && (*p) <= 210u )
				goto tr244;
		} else
			goto st77;
	} else
		goto st77;
	goto st76;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
	switch( (*p) ) {
		case 0u: goto tr247;
		case 13u: goto tr247;
		case 32u: goto tr247;
		case 49u: goto tr247;
		case 95u: goto tr247;
		case 97u: goto tr247;
		case 143u: goto tr247;
		case 167u: goto tr247;
		case 169u: goto tr247;
		case 176u: goto tr247;
		case 187u: goto tr247;
		case 192u: goto tr247;
	}
	if ( (*p) < 128u ) {
		if ( (*p) < 34u ) {
			if ( 9u <= (*p) && (*p) <= 10u )
				goto tr247;
		} else if ( (*p) > 39u ) {
			if ( (*p) > 47u ) {
				if ( 64u <= (*p) && (*p) <= 65u )
					goto tr247;
			} else if ( (*p) >= 42u )
				goto tr247;
		} else
			goto tr247;
	} else if ( (*p) > 129u ) {
		if ( (*p) < 178u ) {
			if ( 159u <= (*p) && (*p) <= 160u )
				goto tr247;
		} else if ( (*p) > 183u ) {
			if ( (*p) > 190u ) {
				if ( 208u <= (*p) && (*p) <= 210u )
					goto tr247;
			} else if ( (*p) >= 189u )
				goto tr247;
		} else
			goto tr247;
	} else
		goto tr247;
	goto st76;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
	switch( (*p) ) {
		case 9u: goto st77;
		case 32u: goto st77;
		case 34u: goto st77;
		case 47u: goto st77;
		case 95u: goto st77;
		case 128u: goto st77;
		case 143u: goto st77;
		case 160u: goto st77;
		case 167u: goto st77;
		case 176u: goto st77;
		case 182u: goto st77;
		case 192u: goto st77;
	}
	if ( (*p) < 42u ) {
		if ( 37u <= (*p) && (*p) <= 39u )
			goto st77;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st77;
		} else if ( (*p) >= 178u )
			goto st77;
	} else
		goto st77;
	goto tr248;
tr236:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st78;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
/* #line 338 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 9u: goto st0;
		case 10u: goto st80;
		case 13u: goto st79;
		case 32u: goto st0;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st0;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr249;
st0:
	if ( ++p == pe )
		goto _test_eof0;
case 0:
	switch( (*p) ) {
		case 9u: goto st1;
		case 10u: goto st79;
		case 13u: goto st79;
		case 32u: goto st1;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st1;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr0;
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	switch( (*p) ) {
		case 9u: goto st2;
		case 10u: goto st79;
		case 13u: goto st79;
		case 32u: goto st2;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st2;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr0;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
	switch( (*p) ) {
		case 9u: goto st3;
		case 10u: goto st79;
		case 13u: goto st79;
		case 32u: goto st3;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st3;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr0;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	switch( (*p) ) {
		case 13u: goto st79;
		case 32u: goto st79;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st79;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( (*p) > 10u ) {
			if ( 34u <= (*p) && (*p) <= 39u )
				goto st4;
		} else if ( (*p) >= 9u )
			goto st79;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr0;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
	switch( (*p) ) {
		case 13u: goto st79;
		case 32u: goto st79;
		case 47u: goto st79;
		case 64u: goto st79;
		case 95u: goto st79;
		case 128u: goto st79;
		case 143u: goto st79;
		case 160u: goto st79;
		case 167u: goto st79;
		case 176u: goto st79;
		case 182u: goto st79;
		case 192u: goto st79;
	}
	if ( (*p) < 42u ) {
		if ( (*p) > 10u ) {
			if ( 34u <= (*p) && (*p) <= 39u )
				goto st79;
		} else if ( (*p) >= 9u )
			goto st79;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st79;
		} else if ( (*p) >= 178u )
			goto st79;
	} else
		goto st79;
	goto tr251;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	switch( (*p) ) {
		case 9u: goto st4;
		case 10u: goto st79;
		case 13u: goto st79;
		case 32u: goto st4;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st4;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr0;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
	switch( (*p) ) {
		case 9u: goto st79;
		case 10u: goto st80;
		case 13u: goto st79;
		case 32u: goto st79;
		case 47u: goto st79;
		case 64u: goto st79;
		case 95u: goto st79;
		case 128u: goto st79;
		case 143u: goto st79;
		case 160u: goto st79;
		case 167u: goto st79;
		case 176u: goto st79;
		case 182u: goto st79;
		case 192u: goto st79;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st79;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st79;
		} else if ( (*p) >= 178u )
			goto st79;
	} else
		goto st79;
	goto tr251;
tr237:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st81;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
/* #line 601 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 9u: goto st0;
		case 10u: goto st5;
		case 13u: goto st82;
		case 32u: goto st0;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st0;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr249;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	switch( (*p) ) {
		case 9u: goto st0;
		case 10u: goto st79;
		case 13u: goto st79;
		case 32u: goto st0;
		case 47u: goto st4;
		case 64u: goto st4;
		case 95u: goto st4;
		case 128u: goto st4;
		case 143u: goto st4;
		case 160u: goto st0;
		case 167u: goto st4;
		case 176u: goto st4;
		case 182u: goto st4;
		case 192u: goto st4;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 39u )
			goto st4;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st4;
		} else if ( (*p) >= 178u )
			goto st4;
	} else
		goto st4;
	goto tr0;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
	switch( (*p) ) {
		case 13u: goto st82;
		case 32u: goto st79;
		case 47u: goto st79;
		case 64u: goto st79;
		case 95u: goto st79;
		case 128u: goto st79;
		case 143u: goto st79;
		case 160u: goto st79;
		case 167u: goto st79;
		case 176u: goto st79;
		case 182u: goto st79;
		case 192u: goto st79;
	}
	if ( (*p) < 42u ) {
		if ( (*p) > 10u ) {
			if ( 34u <= (*p) && (*p) <= 39u )
				goto st79;
		} else if ( (*p) >= 9u )
			goto st79;
	} else if ( (*p) > 45u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st79;
		} else if ( (*p) >= 178u )
			goto st79;
	} else
		goto st79;
	goto tr251;
tr238:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st83;
st83:
	if ( ++p == pe )
		goto _test_eof83;
case 83:
/* #line 707 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr240;
		case 65u: goto tr241;
		case 97u: goto tr241;
		case 128u: goto tr255;
		case 129u: goto tr241;
		case 169u: goto tr244;
		case 183u: goto tr244;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr244;
	goto tr254;
tr240:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st84;
tr210:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st84;
tr263:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st84;
tr358:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st84;
st84:
	if ( ++p == pe )
		goto _test_eof84;
case 84:
/* #line 776 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr258;
		case 43u: goto tr259;
		case 45u: goto tr260;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr263;
		case 64u: goto tr264;
		case 65u: goto tr265;
		case 95u: goto tr266;
		case 97u: goto tr265;
		case 128u: goto tr263;
		case 129u: goto tr265;
		case 143u: goto tr267;
	}
	goto tr256;
tr257:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 30 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateSuffix(*p);
    }
	goto st85;
st85:
	if ( ++p == pe )
		goto _test_eof85;
case 85:
/* #line 810 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 43u: goto st6;
		case 46u: goto st12;
		case 47u: goto st17;
		case 64u: goto st19;
		case 95u: goto st20;
	}
	goto tr268;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	switch( (*p) ) {
		case 49u: goto tr9;
		case 64u: goto tr8;
		case 65u: goto tr10;
		case 97u: goto tr10;
		case 128u: goto tr11;
		case 129u: goto tr10;
		case 169u: goto tr12;
		case 183u: goto tr12;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr12;
	} else if ( (*p) >= 35u )
		goto tr8;
	goto tr7;
tr8:
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); }
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st7;
tr33:
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_DOT, p[-1]); }
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st7;
tr54:
/* #line 43 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_SLASH, p[-1]); }
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st7;
tr63:
/* #line 44 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_AT_SIGN, p[-1]); }
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st7;
tr68:
/* #line 42 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_UNDERSCORE, p[-1]); }
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 883 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr13;
		case 65u: goto tr14;
		case 97u: goto tr14;
		case 128u: goto tr15;
		case 129u: goto tr14;
		case 169u: goto tr16;
		case 183u: goto tr16;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr16;
	goto tr7;
tr9:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr13:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr34:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_DOT, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr55:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 43 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_SLASH, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr59:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr64:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 44 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_AT_SIGN, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr69:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 42 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_UNDERSCORE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr276:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
tr294:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st86;
st86:
	if ( ++p == pe )
		goto _test_eof86;
case 86:
/* #line 1042 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr274;
		case 43u: goto tr259;
		case 45u: goto tr275;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr276;
		case 64u: goto tr264;
		case 65u: goto tr277;
		case 95u: goto tr266;
		case 97u: goto tr277;
		case 128u: goto tr276;
		case 129u: goto tr277;
	}
	goto tr256;
tr274:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 1069 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr17;
		case 65u: goto tr18;
		case 97u: goto tr18;
		case 128u: goto tr19;
		case 129u: goto tr18;
		case 169u: goto tr20;
		case 183u: goto tr20;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr20;
	goto tr7;
tr17:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st87;
tr38:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st87;
tr85:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st87;
tr280:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st87;
tr296:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st87;
st87:
	if ( ++p == pe )
		goto _test_eof87;
case 87:
/* #line 1158 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr278;
		case 43u: goto tr259;
		case 45u: goto tr279;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr280;
		case 64u: goto tr264;
		case 65u: goto tr281;
		case 95u: goto tr266;
		case 97u: goto tr281;
		case 128u: goto tr280;
		case 129u: goto tr281;
	}
	goto tr256;
tr278:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 1185 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr21;
		case 65u: goto tr22;
		case 97u: goto tr22;
		case 128u: goto tr23;
		case 129u: goto tr22;
		case 169u: goto tr24;
		case 183u: goto tr24;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr24;
	goto tr7;
tr21:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st88;
tr42:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st88;
tr81:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st88;
tr284:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st88;
tr298:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st88;
st88:
	if ( ++p == pe )
		goto _test_eof88;
case 88:
/* #line 1274 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr282;
		case 43u: goto tr259;
		case 45u: goto tr283;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr284;
		case 64u: goto tr264;
		case 65u: goto tr285;
		case 95u: goto tr266;
		case 97u: goto tr285;
		case 128u: goto tr284;
		case 129u: goto tr285;
	}
	goto tr256;
tr282:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 1301 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr25;
		case 65u: goto tr26;
		case 97u: goto tr26;
		case 128u: goto tr27;
		case 129u: goto tr26;
		case 169u: goto tr28;
		case 183u: goto tr28;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr28;
	goto tr7;
tr25:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st89;
tr46:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st89;
tr77:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st89;
tr288:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st89;
tr300:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st89;
st89:
	if ( ++p == pe )
		goto _test_eof89;
case 89:
/* #line 1390 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr286;
		case 43u: goto tr259;
		case 45u: goto tr287;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr288;
		case 64u: goto tr264;
		case 65u: goto tr289;
		case 95u: goto tr266;
		case 97u: goto tr289;
		case 128u: goto tr288;
		case 129u: goto tr289;
	}
	goto tr256;
tr286:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 1417 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr29;
		case 65u: goto tr30;
		case 97u: goto tr30;
		case 128u: goto tr31;
		case 129u: goto tr30;
		case 169u: goto tr32;
		case 183u: goto tr32;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr32;
	goto tr7;
tr29:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st90;
tr50:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st90;
tr73:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st90;
tr290:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st90;
tr302:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st90;
st90:
	if ( ++p == pe )
		goto _test_eof90;
case 90:
/* #line 1506 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 43u: goto tr259;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr290;
		case 64u: goto tr264;
		case 65u: goto tr291;
		case 95u: goto tr266;
		case 97u: goto tr291;
		case 128u: goto tr290;
		case 129u: goto tr291;
	}
	goto tr256;
tr259:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 30 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateSuffix(*p);
    }
	goto st91;
st91:
	if ( ++p == pe )
		goto _test_eof91;
case 91:
/* #line 1537 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 43u: goto tr292;
		case 46u: goto st12;
		case 47u: goto st17;
		case 49u: goto tr9;
		case 64u: goto tr293;
		case 65u: goto tr10;
		case 95u: goto st20;
		case 97u: goto tr10;
		case 128u: goto tr11;
		case 129u: goto tr10;
		case 169u: goto tr12;
		case 183u: goto tr12;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr12;
	} else if ( (*p) >= 35u )
		goto tr8;
	goto tr268;
tr292:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 30 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateSuffix(*p);
    }
	goto st92;
st92:
	if ( ++p == pe )
		goto _test_eof92;
case 92:
/* #line 1570 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 43u: goto st6;
		case 46u: goto st12;
		case 47u: goto st17;
		case 49u: goto tr9;
		case 64u: goto tr293;
		case 65u: goto tr10;
		case 95u: goto st20;
		case 97u: goto tr10;
		case 128u: goto tr11;
		case 129u: goto tr10;
		case 169u: goto tr12;
		case 183u: goto tr12;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr12;
	} else if ( (*p) >= 35u )
		goto tr8;
	goto tr268;
tr261:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 1601 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr34;
		case 64u: goto tr33;
		case 65u: goto tr35;
		case 97u: goto tr35;
		case 128u: goto tr36;
		case 129u: goto tr35;
		case 169u: goto tr37;
		case 183u: goto tr37;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr37;
	} else if ( (*p) >= 35u )
		goto tr33;
	goto tr7;
tr10:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr14:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr35:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_DOT, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr56:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 43 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_SLASH, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr60:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr65:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 44 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_AT_SIGN, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr70:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 42 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_UNDERSCORE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr295:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
tr277:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st93;
st93:
	if ( ++p == pe )
		goto _test_eof93;
case 93:
/* #line 1764 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr274;
		case 43u: goto tr259;
		case 45u: goto tr275;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr294;
		case 64u: goto tr264;
		case 65u: goto tr295;
		case 95u: goto tr266;
		case 97u: goto tr295;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr295;
	goto tr256;
tr275:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 1791 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr38;
		case 65u: goto tr39;
		case 97u: goto tr39;
		case 128u: goto tr40;
		case 129u: goto tr39;
		case 169u: goto tr41;
		case 183u: goto tr41;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr41;
	goto tr7;
tr18:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st94;
tr39:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st94;
tr86:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st94;
tr297:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st94;
tr281:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st94;
st94:
	if ( ++p == pe )
		goto _test_eof94;
case 94:
/* #line 1880 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr278;
		case 43u: goto tr259;
		case 45u: goto tr279;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr296;
		case 64u: goto tr264;
		case 65u: goto tr297;
		case 95u: goto tr266;
		case 97u: goto tr297;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr297;
	goto tr256;
tr279:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 1907 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr42;
		case 65u: goto tr43;
		case 97u: goto tr43;
		case 128u: goto tr44;
		case 129u: goto tr43;
		case 169u: goto tr45;
		case 183u: goto tr45;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr45;
	goto tr7;
tr22:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st95;
tr43:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st95;
tr82:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st95;
tr299:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st95;
tr285:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st95;
st95:
	if ( ++p == pe )
		goto _test_eof95;
case 95:
/* #line 1996 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr282;
		case 43u: goto tr259;
		case 45u: goto tr283;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr298;
		case 64u: goto tr264;
		case 65u: goto tr299;
		case 95u: goto tr266;
		case 97u: goto tr299;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr299;
	goto tr256;
tr283:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 2023 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr46;
		case 65u: goto tr47;
		case 97u: goto tr47;
		case 128u: goto tr48;
		case 129u: goto tr47;
		case 169u: goto tr49;
		case 183u: goto tr49;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr49;
	goto tr7;
tr26:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st96;
tr47:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st96;
tr78:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st96;
tr301:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st96;
tr289:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st96;
st96:
	if ( ++p == pe )
		goto _test_eof96;
case 96:
/* #line 2112 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr286;
		case 43u: goto tr259;
		case 45u: goto tr287;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr300;
		case 64u: goto tr264;
		case 65u: goto tr301;
		case 95u: goto tr266;
		case 97u: goto tr301;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr301;
	goto tr256;
tr287:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 2139 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr50;
		case 65u: goto tr51;
		case 97u: goto tr51;
		case 128u: goto tr52;
		case 129u: goto tr51;
		case 169u: goto tr53;
		case 183u: goto tr53;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr53;
	goto tr7;
tr30:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st97;
tr51:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st97;
tr74:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st97;
tr303:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st97;
tr291:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st97;
st97:
	if ( ++p == pe )
		goto _test_eof97;
case 97:
/* #line 2228 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 43u: goto tr259;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr302;
		case 64u: goto tr264;
		case 65u: goto tr303;
		case 95u: goto tr266;
		case 97u: goto tr303;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr303;
	goto tr256;
tr262:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 2253 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr55;
		case 64u: goto tr54;
		case 65u: goto tr56;
		case 97u: goto tr56;
		case 128u: goto tr57;
		case 129u: goto tr56;
		case 169u: goto tr58;
		case 183u: goto tr58;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr58;
	} else if ( (*p) >= 35u )
		goto tr54;
	goto tr7;
tr11:
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
tr15:
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
tr36:
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_DOT, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
tr57:
/* #line 43 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_SLASH, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
tr61:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
tr66:
/* #line 44 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_AT_SIGN, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
tr71:
/* #line 42 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_UNDERSCORE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 2350 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr59;
		case 65u: goto tr60;
		case 97u: goto tr60;
		case 128u: goto tr61;
		case 129u: goto tr60;
		case 169u: goto tr62;
		case 183u: goto tr62;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr62;
	goto tr7;
tr12:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
tr16:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
tr37:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_DOT, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
tr58:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 43 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_SLASH, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
tr62:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
tr67:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 44 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_AT_SIGN, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
tr72:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 42 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_UNDERSCORE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st98;
st98:
	if ( ++p == pe )
		goto _test_eof98;
case 98:
/* #line 2485 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr274;
		case 43u: goto tr259;
		case 45u: goto tr275;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
	}
	goto tr256;
tr293:
/* #line 41 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_PLUS, p[-1]); }
/* #line 26 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdatePrefix(*p);
    }
	goto st19;
tr264:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 2515 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr64;
		case 64u: goto tr63;
		case 65u: goto tr65;
		case 97u: goto tr65;
		case 128u: goto tr66;
		case 129u: goto tr65;
		case 169u: goto tr67;
		case 183u: goto tr67;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr67;
	} else if ( (*p) >= 35u )
		goto tr63;
	goto tr7;
tr266:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
/* #line 2542 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr69;
		case 64u: goto tr68;
		case 65u: goto tr70;
		case 97u: goto tr70;
		case 128u: goto tr71;
		case 129u: goto tr70;
		case 169u: goto tr72;
		case 183u: goto tr72;
	}
	if ( (*p) > 36u ) {
		if ( 208u <= (*p) && (*p) <= 210u )
			goto tr72;
	} else if ( (*p) >= 35u )
		goto tr68;
	goto tr7;
tr31:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st21;
tr52:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st21;
tr75:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st21;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
/* #line 2593 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr73;
		case 65u: goto tr74;
		case 97u: goto tr74;
		case 128u: goto tr75;
		case 129u: goto tr74;
		case 169u: goto tr76;
		case 183u: goto tr76;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr76;
	goto tr7;
tr32:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st99;
tr53:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st99;
tr76:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st99;
st99:
	if ( ++p == pe )
		goto _test_eof99;
case 99:
/* #line 2658 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 43u: goto tr259;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
	}
	goto tr256;
tr27:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st22;
tr48:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st22;
tr79:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st22;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
/* #line 2702 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr77;
		case 65u: goto tr78;
		case 97u: goto tr78;
		case 128u: goto tr79;
		case 129u: goto tr78;
		case 169u: goto tr80;
		case 183u: goto tr80;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr80;
	goto tr7;
tr28:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st100;
tr49:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st100;
tr80:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st100;
st100:
	if ( ++p == pe )
		goto _test_eof100;
case 100:
/* #line 2767 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr286;
		case 43u: goto tr259;
		case 45u: goto tr287;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
	}
	goto tr256;
tr23:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st23;
tr44:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st23;
tr83:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st23;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
/* #line 2813 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr81;
		case 65u: goto tr82;
		case 97u: goto tr82;
		case 128u: goto tr83;
		case 129u: goto tr82;
		case 169u: goto tr84;
		case 183u: goto tr84;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr84;
	goto tr7;
tr24:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st101;
tr45:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st101;
tr84:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st101;
st101:
	if ( ++p == pe )
		goto _test_eof101;
case 101:
/* #line 2878 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr282;
		case 43u: goto tr259;
		case 45u: goto tr283;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
	}
	goto tr256;
tr19:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st24;
tr40:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st24;
tr87:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st24;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
/* #line 2924 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr85;
		case 65u: goto tr86;
		case 97u: goto tr86;
		case 128u: goto tr87;
		case 129u: goto tr86;
		case 169u: goto tr88;
		case 183u: goto tr88;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr88;
	goto tr7;
tr20:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st102;
tr41:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st102;
tr88:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st102;
st102:
	if ( ++p == pe )
		goto _test_eof102;
case 102:
/* #line 2989 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr278;
		case 43u: goto tr259;
		case 45u: goto tr279;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
	}
	goto tr256;
tr258:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st25;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
/* #line 3011 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr89;
		case 65u: goto tr90;
		case 97u: goto tr90;
		case 128u: goto tr91;
		case 129u: goto tr90;
		case 169u: goto tr92;
		case 183u: goto tr92;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr92;
	goto tr7;
tr89:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st103;
tr205:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st103;
tr201:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st103;
tr306:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st103;
tr356:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st103;
st103:
	if ( ++p == pe )
		goto _test_eof103;
case 103:
/* #line 3100 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr304;
		case 43u: goto tr259;
		case 45u: goto tr305;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr306;
		case 64u: goto tr264;
		case 65u: goto tr307;
		case 95u: goto tr266;
		case 97u: goto tr307;
		case 128u: goto tr306;
		case 129u: goto tr307;
		case 143u: goto tr267;
	}
	goto tr256;
tr304:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st26;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
/* #line 3128 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr93;
		case 65u: goto tr94;
		case 97u: goto tr94;
		case 128u: goto tr95;
		case 129u: goto tr94;
		case 169u: goto tr96;
		case 183u: goto tr96;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr96;
	goto tr7;
tr93:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st104;
tr197:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st104;
tr193:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st104;
tr310:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st104;
tr354:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st104;
st104:
	if ( ++p == pe )
		goto _test_eof104;
case 104:
/* #line 3217 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr308;
		case 43u: goto tr259;
		case 45u: goto tr309;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr310;
		case 64u: goto tr264;
		case 65u: goto tr311;
		case 95u: goto tr266;
		case 97u: goto tr311;
		case 128u: goto tr310;
		case 129u: goto tr311;
		case 143u: goto tr267;
	}
	goto tr256;
tr308:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st27;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
/* #line 3245 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr97;
		case 65u: goto tr98;
		case 97u: goto tr98;
		case 128u: goto tr99;
		case 129u: goto tr98;
		case 169u: goto tr100;
		case 183u: goto tr100;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr100;
	goto tr7;
tr97:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st105;
tr189:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st105;
tr185:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st105;
tr314:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st105;
tr352:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st105;
st105:
	if ( ++p == pe )
		goto _test_eof105;
case 105:
/* #line 3334 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr312;
		case 43u: goto tr259;
		case 45u: goto tr313;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr314;
		case 64u: goto tr264;
		case 65u: goto tr315;
		case 95u: goto tr266;
		case 97u: goto tr315;
		case 128u: goto tr314;
		case 129u: goto tr315;
		case 143u: goto tr267;
	}
	goto tr256;
tr312:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st28;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
/* #line 3362 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr101;
		case 65u: goto tr102;
		case 97u: goto tr102;
		case 128u: goto tr103;
		case 129u: goto tr102;
		case 169u: goto tr104;
		case 183u: goto tr104;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr104;
	goto tr7;
tr101:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st106;
tr181:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st106;
tr177:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st106;
tr317:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st106;
tr350:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st106;
st106:
	if ( ++p == pe )
		goto _test_eof106;
case 106:
/* #line 3451 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 43u: goto tr259;
		case 45u: goto tr316;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr317;
		case 64u: goto tr264;
		case 65u: goto tr318;
		case 95u: goto tr266;
		case 97u: goto tr318;
		case 128u: goto tr317;
		case 129u: goto tr318;
		case 143u: goto tr267;
	}
	goto tr256;
tr316:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st29;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
/* #line 3478 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 160u: goto st44;
	}
	goto tr7;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	switch( (*p) ) {
		case 32u: goto st31;
		case 49u: goto tr109;
		case 65u: goto tr110;
		case 97u: goto tr110;
		case 128u: goto tr111;
		case 129u: goto tr110;
		case 160u: goto st31;
		case 169u: goto tr112;
		case 183u: goto tr112;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr112;
	goto tr7;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
	switch( (*p) ) {
		case 32u: goto st32;
		case 49u: goto tr109;
		case 65u: goto tr110;
		case 97u: goto tr110;
		case 128u: goto tr111;
		case 129u: goto tr110;
		case 160u: goto st32;
		case 169u: goto tr112;
		case 183u: goto tr112;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr112;
	goto tr7;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
	switch( (*p) ) {
		case 32u: goto st33;
		case 49u: goto tr109;
		case 65u: goto tr110;
		case 97u: goto tr110;
		case 128u: goto tr111;
		case 129u: goto tr110;
		case 160u: goto st33;
		case 169u: goto tr112;
		case 183u: goto tr112;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr112;
	goto tr7;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
	switch( (*p) ) {
		case 32u: goto st34;
		case 49u: goto tr109;
		case 65u: goto tr110;
		case 97u: goto tr110;
		case 128u: goto tr111;
		case 129u: goto tr110;
		case 160u: goto st34;
		case 169u: goto tr112;
		case 183u: goto tr112;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr112;
	goto tr7;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
	switch( (*p) ) {
		case 49u: goto tr109;
		case 65u: goto tr110;
		case 97u: goto tr110;
		case 128u: goto tr111;
		case 129u: goto tr110;
		case 169u: goto tr112;
		case 183u: goto tr112;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr112;
	goto tr7;
tr142:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st107;
tr109:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetHyphenation();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st107;
tr132:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetSoftHyphen();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st107;
tr323:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st107;
tr342:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st107;
st107:
	if ( ++p == pe )
		goto _test_eof107;
case 107:
/* #line 3654 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr320;
		case 43u: goto tr321;
		case 45u: goto tr322;
		case 49u: goto tr323;
		case 65u: goto tr324;
		case 97u: goto tr324;
		case 128u: goto tr323;
		case 129u: goto tr324;
		case 143u: goto tr267;
	}
	goto tr256;
tr320:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st35;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
/* #line 3678 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr116;
		case 65u: goto tr117;
		case 97u: goto tr117;
		case 128u: goto tr118;
		case 129u: goto tr117;
		case 169u: goto tr119;
		case 183u: goto tr119;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr119;
	goto tr7;
tr116:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st108;
tr137:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st108;
tr173:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st108;
tr327:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st108;
tr344:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st108;
st108:
	if ( ++p == pe )
		goto _test_eof108;
case 108:
/* #line 3767 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr325;
		case 43u: goto tr321;
		case 45u: goto tr326;
		case 49u: goto tr327;
		case 65u: goto tr328;
		case 97u: goto tr328;
		case 128u: goto tr327;
		case 129u: goto tr328;
		case 143u: goto tr267;
	}
	goto tr256;
tr325:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st36;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
/* #line 3791 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr120;
		case 65u: goto tr121;
		case 97u: goto tr121;
		case 128u: goto tr122;
		case 129u: goto tr121;
		case 169u: goto tr123;
		case 183u: goto tr123;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr123;
	goto tr7;
tr120:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st109;
tr149:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st109;
tr169:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st109;
tr331:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st109;
tr346:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st109;
st109:
	if ( ++p == pe )
		goto _test_eof109;
case 109:
/* #line 3880 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr329;
		case 43u: goto tr321;
		case 45u: goto tr330;
		case 49u: goto tr331;
		case 65u: goto tr332;
		case 97u: goto tr332;
		case 128u: goto tr331;
		case 129u: goto tr332;
		case 143u: goto tr267;
	}
	goto tr256;
tr329:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st37;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
/* #line 3904 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr124;
		case 65u: goto tr125;
		case 97u: goto tr125;
		case 128u: goto tr126;
		case 129u: goto tr125;
		case 169u: goto tr127;
		case 183u: goto tr127;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr127;
	goto tr7;
tr124:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st110;
tr153:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st110;
tr165:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st110;
tr335:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st110;
tr348:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st110;
st110:
	if ( ++p == pe )
		goto _test_eof110;
case 110:
/* #line 3993 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr333;
		case 43u: goto tr321;
		case 45u: goto tr334;
		case 49u: goto tr335;
		case 65u: goto tr336;
		case 97u: goto tr336;
		case 128u: goto tr335;
		case 129u: goto tr336;
		case 143u: goto tr267;
	}
	goto tr256;
tr333:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st38;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
/* #line 4017 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr128;
		case 65u: goto tr129;
		case 97u: goto tr129;
		case 128u: goto tr130;
		case 129u: goto tr129;
		case 169u: goto tr131;
		case 183u: goto tr131;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr131;
	goto tr7;
tr128:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st111;
tr157:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st111;
tr161:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st111;
tr337:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st111;
tr340:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 14 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_NUMBER);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st111;
st111:
	if ( ++p == pe )
		goto _test_eof111;
case 111:
/* #line 4106 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 43u: goto tr321;
		case 45u: goto tr316;
		case 49u: goto tr337;
		case 65u: goto tr338;
		case 97u: goto tr338;
		case 128u: goto tr337;
		case 129u: goto tr338;
		case 143u: goto tr267;
	}
	goto tr256;
tr321:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 30 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateSuffix(*p);
    }
	goto st112;
st112:
	if ( ++p == pe )
		goto _test_eof112;
case 112:
/* #line 4133 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	if ( (*p) == 43u )
		goto tr339;
	goto tr268;
tr129:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st113;
tr158:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st113;
tr162:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st113;
tr341:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st113;
tr338:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st113;
st113:
	if ( ++p == pe )
		goto _test_eof113;
case 113:
/* #line 4213 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 43u: goto tr321;
		case 45u: goto tr316;
		case 49u: goto tr340;
		case 65u: goto tr341;
		case 97u: goto tr341;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr341;
	goto tr256;
tr267:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st39;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
/* #line 4236 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr132;
		case 65u: goto tr133;
		case 97u: goto tr133;
		case 128u: goto tr134;
		case 129u: goto tr133;
		case 143u: goto st39;
		case 169u: goto tr136;
		case 183u: goto tr136;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr136;
	goto tr7;
tr143:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st114;
tr110:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetHyphenation();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st114;
tr133:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetSoftHyphen();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st114;
tr343:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st114;
tr324:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st114;
st114:
	if ( ++p == pe )
		goto _test_eof114;
case 114:
/* #line 4330 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr320;
		case 43u: goto tr321;
		case 45u: goto tr322;
		case 49u: goto tr342;
		case 65u: goto tr343;
		case 97u: goto tr343;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr343;
	goto tr256;
tr322:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st40;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
/* #line 4354 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr137;
		case 65u: goto tr138;
		case 97u: goto tr138;
		case 128u: goto tr139;
		case 129u: goto tr138;
		case 160u: goto st44;
		case 169u: goto tr140;
		case 183u: goto tr140;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr140;
	goto tr7;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st42;
		case 32u: goto st31;
		case 49u: goto tr109;
		case 65u: goto tr110;
		case 97u: goto tr110;
		case 128u: goto tr111;
		case 129u: goto tr110;
		case 160u: goto st31;
		case 169u: goto tr112;
		case 183u: goto tr112;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr112;
	goto tr7;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 10u )
		goto st30;
	goto tr7;
tr144:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st43;
tr111:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetHyphenation();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st43;
tr134:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetSoftHyphen();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st43;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
/* #line 4436 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr142;
		case 65u: goto tr143;
		case 97u: goto tr143;
		case 128u: goto tr144;
		case 129u: goto tr143;
		case 169u: goto tr145;
		case 183u: goto tr145;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr145;
	goto tr7;
tr145:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st115;
tr112:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetHyphenation();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st115;
tr136:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        SetSoftHyphen();
    }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st115;
st115:
	if ( ++p == pe )
		goto _test_eof115;
case 115:
/* #line 4505 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr320;
		case 43u: goto tr321;
		case 45u: goto tr322;
		case 143u: goto tr267;
	}
	goto tr256;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st45;
		case 160u: goto st45;
	}
	goto tr7;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st46;
		case 160u: goto st46;
	}
	goto tr7;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st47;
		case 160u: goto st47;
	}
	goto tr7;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
	}
	goto tr7;
tr117:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st116;
tr138:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st116;
tr174:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st116;
tr345:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st116;
tr328:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st116;
st116:
	if ( ++p == pe )
		goto _test_eof116;
case 116:
/* #line 4632 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr325;
		case 43u: goto tr321;
		case 45u: goto tr326;
		case 49u: goto tr344;
		case 65u: goto tr345;
		case 97u: goto tr345;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr345;
	goto tr256;
tr326:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st48;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
/* #line 4656 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr149;
		case 65u: goto tr150;
		case 97u: goto tr150;
		case 128u: goto tr151;
		case 129u: goto tr150;
		case 160u: goto st44;
		case 169u: goto tr152;
		case 183u: goto tr152;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr152;
	goto tr7;
tr121:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st117;
tr150:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st117;
tr170:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st117;
tr347:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st117;
tr332:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st117;
st117:
	if ( ++p == pe )
		goto _test_eof117;
case 117:
/* #line 4749 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr329;
		case 43u: goto tr321;
		case 45u: goto tr330;
		case 49u: goto tr346;
		case 65u: goto tr347;
		case 97u: goto tr347;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr347;
	goto tr256;
tr330:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st49;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
/* #line 4773 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr153;
		case 65u: goto tr154;
		case 97u: goto tr154;
		case 128u: goto tr155;
		case 129u: goto tr154;
		case 160u: goto st44;
		case 169u: goto tr156;
		case 183u: goto tr156;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr156;
	goto tr7;
tr125:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st118;
tr154:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st118;
tr166:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st118;
tr349:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st118;
tr336:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st118;
st118:
	if ( ++p == pe )
		goto _test_eof118;
case 118:
/* #line 4866 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr333;
		case 43u: goto tr321;
		case 45u: goto tr334;
		case 49u: goto tr348;
		case 65u: goto tr349;
		case 97u: goto tr349;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr349;
	goto tr256;
tr334:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st50;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
/* #line 4890 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr157;
		case 65u: goto tr158;
		case 97u: goto tr158;
		case 128u: goto tr159;
		case 129u: goto tr158;
		case 160u: goto st44;
		case 169u: goto tr160;
		case 183u: goto tr160;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr160;
	goto tr7;
tr130:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st51;
tr159:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st51;
tr163:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st51;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
/* #line 4941 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr161;
		case 65u: goto tr162;
		case 97u: goto tr162;
		case 128u: goto tr163;
		case 129u: goto tr162;
		case 169u: goto tr164;
		case 183u: goto tr164;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr164;
	goto tr7;
tr131:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st119;
tr160:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st119;
tr164:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st119;
st119:
	if ( ++p == pe )
		goto _test_eof119;
case 119:
/* #line 5006 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 43u: goto tr321;
		case 45u: goto tr316;
		case 143u: goto tr267;
	}
	goto tr256;
tr126:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st52;
tr155:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st52;
tr167:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st52;
st52:
	if ( ++p == pe )
		goto _test_eof52;
case 52:
/* #line 5048 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr165;
		case 65u: goto tr166;
		case 97u: goto tr166;
		case 128u: goto tr167;
		case 129u: goto tr166;
		case 169u: goto tr168;
		case 183u: goto tr168;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr168;
	goto tr7;
tr127:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st120;
tr156:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st120;
tr168:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st120;
st120:
	if ( ++p == pe )
		goto _test_eof120;
case 120:
/* #line 5113 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr333;
		case 43u: goto tr321;
		case 45u: goto tr334;
		case 143u: goto tr267;
	}
	goto tr256;
tr122:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st53;
tr151:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st53;
tr171:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st53;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
/* #line 5156 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr169;
		case 65u: goto tr170;
		case 97u: goto tr170;
		case 128u: goto tr171;
		case 129u: goto tr170;
		case 169u: goto tr172;
		case 183u: goto tr172;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr172;
	goto tr7;
tr123:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st121;
tr152:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st121;
tr172:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st121;
st121:
	if ( ++p == pe )
		goto _test_eof121;
case 121:
/* #line 5221 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr329;
		case 43u: goto tr321;
		case 45u: goto tr330;
		case 143u: goto tr267;
	}
	goto tr256;
tr118:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st54;
tr139:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st54;
tr175:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st54;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
/* #line 5264 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr173;
		case 65u: goto tr174;
		case 97u: goto tr174;
		case 128u: goto tr175;
		case 129u: goto tr174;
		case 169u: goto tr176;
		case 183u: goto tr176;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr176;
	goto tr7;
tr119:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st122;
tr140:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st122;
tr176:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st122;
st122:
	if ( ++p == pe )
		goto _test_eof122;
case 122:
/* #line 5329 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr319;
		case 39u: goto tr325;
		case 43u: goto tr321;
		case 45u: goto tr326;
		case 143u: goto tr267;
	}
	goto tr256;
tr102:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st123;
tr182:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st123;
tr178:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st123;
tr351:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st123;
tr318:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st123;
st123:
	if ( ++p == pe )
		goto _test_eof123;
case 123:
/* #line 5414 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 43u: goto tr259;
		case 45u: goto tr316;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr350;
		case 64u: goto tr264;
		case 65u: goto tr351;
		case 95u: goto tr266;
		case 97u: goto tr351;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr351;
	goto tr256;
tr103:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st55;
tr183:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st55;
tr179:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st55;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
/* #line 5465 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr177;
		case 65u: goto tr178;
		case 97u: goto tr178;
		case 128u: goto tr179;
		case 129u: goto tr178;
		case 169u: goto tr180;
		case 183u: goto tr180;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr180;
	goto tr7;
tr104:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st124;
tr184:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st124;
tr180:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st124;
st124:
	if ( ++p == pe )
		goto _test_eof124;
case 124:
/* #line 5530 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 43u: goto tr259;
		case 45u: goto tr316;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
		case 143u: goto tr267;
	}
	goto tr256;
tr313:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st56;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
/* #line 5552 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr181;
		case 65u: goto tr182;
		case 97u: goto tr182;
		case 128u: goto tr183;
		case 129u: goto tr182;
		case 160u: goto st44;
		case 169u: goto tr184;
		case 183u: goto tr184;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr184;
	goto tr7;
tr98:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st125;
tr190:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st125;
tr186:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st125;
tr353:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st125;
tr315:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st125;
st125:
	if ( ++p == pe )
		goto _test_eof125;
case 125:
/* #line 5645 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr312;
		case 43u: goto tr259;
		case 45u: goto tr313;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr352;
		case 64u: goto tr264;
		case 65u: goto tr353;
		case 95u: goto tr266;
		case 97u: goto tr353;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr353;
	goto tr256;
tr99:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st57;
tr191:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st57;
tr187:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st57;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
/* #line 5697 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr185;
		case 65u: goto tr186;
		case 97u: goto tr186;
		case 128u: goto tr187;
		case 129u: goto tr186;
		case 169u: goto tr188;
		case 183u: goto tr188;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr188;
	goto tr7;
tr100:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st126;
tr192:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st126;
tr188:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st126;
st126:
	if ( ++p == pe )
		goto _test_eof126;
case 126:
/* #line 5762 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr312;
		case 43u: goto tr259;
		case 45u: goto tr313;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
		case 143u: goto tr267;
	}
	goto tr256;
tr309:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st58;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
/* #line 5785 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr189;
		case 65u: goto tr190;
		case 97u: goto tr190;
		case 128u: goto tr191;
		case 129u: goto tr190;
		case 160u: goto st44;
		case 169u: goto tr192;
		case 183u: goto tr192;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr192;
	goto tr7;
tr94:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st127;
tr198:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st127;
tr194:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st127;
tr355:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st127;
tr311:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st127;
st127:
	if ( ++p == pe )
		goto _test_eof127;
case 127:
/* #line 5878 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr308;
		case 43u: goto tr259;
		case 45u: goto tr309;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr354;
		case 64u: goto tr264;
		case 65u: goto tr355;
		case 95u: goto tr266;
		case 97u: goto tr355;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr355;
	goto tr256;
tr95:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st59;
tr199:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st59;
tr195:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st59;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
/* #line 5930 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr193;
		case 65u: goto tr194;
		case 97u: goto tr194;
		case 128u: goto tr195;
		case 129u: goto tr194;
		case 169u: goto tr196;
		case 183u: goto tr196;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr196;
	goto tr7;
tr96:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st128;
tr200:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st128;
tr196:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st128;
st128:
	if ( ++p == pe )
		goto _test_eof128;
case 128:
/* #line 5995 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr308;
		case 43u: goto tr259;
		case 45u: goto tr309;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
		case 143u: goto tr267;
	}
	goto tr256;
tr305:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st60;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
/* #line 6018 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr197;
		case 65u: goto tr198;
		case 97u: goto tr198;
		case 128u: goto tr199;
		case 129u: goto tr198;
		case 160u: goto st44;
		case 169u: goto tr200;
		case 183u: goto tr200;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr200;
	goto tr7;
tr90:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st129;
tr206:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st129;
tr202:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st129;
tr357:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st129;
tr307:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st129;
st129:
	if ( ++p == pe )
		goto _test_eof129;
case 129:
/* #line 6111 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr304;
		case 43u: goto tr259;
		case 45u: goto tr305;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr356;
		case 64u: goto tr264;
		case 65u: goto tr357;
		case 95u: goto tr266;
		case 97u: goto tr357;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr357;
	goto tr256;
tr91:
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st61;
tr207:
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st61;
tr203:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st61;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
/* #line 6163 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr201;
		case 65u: goto tr202;
		case 97u: goto tr202;
		case 128u: goto tr203;
		case 129u: goto tr202;
		case 169u: goto tr204;
		case 183u: goto tr204;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr204;
	goto tr7;
tr92:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 38 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_APOSTROPHE, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st130;
tr208:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 39 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{ SetTokenDelim(TOKDELIM_MINUS, p[-1]); }
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st130;
tr204:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st130;
st130:
	if ( ++p == pe )
		goto _test_eof130;
case 130:
/* #line 6228 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr304;
		case 43u: goto tr259;
		case 45u: goto tr305;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
		case 143u: goto tr267;
	}
	goto tr256;
tr260:
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
	goto st62;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
/* #line 6251 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 10u: goto st30;
		case 13u: goto st41;
		case 32u: goto st44;
		case 49u: goto tr205;
		case 65u: goto tr206;
		case 97u: goto tr206;
		case 128u: goto tr207;
		case 129u: goto tr206;
		case 160u: goto st44;
		case 169u: goto tr208;
		case 183u: goto tr208;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr208;
	goto tr7;
tr241:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st131;
tr211:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st131;
tr359:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st131;
tr265:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 22 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        AddToken();
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st131;
st131:
	if ( ++p == pe )
		goto _test_eof131;
case 131:
/* #line 6324 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr258;
		case 43u: goto tr259;
		case 45u: goto tr260;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 49u: goto tr358;
		case 64u: goto tr264;
		case 65u: goto tr359;
		case 95u: goto tr266;
		case 97u: goto tr359;
		case 143u: goto tr267;
	}
	if ( 128u <= (*p) && (*p) <= 129u )
		goto tr359;
	goto tr256;
tr255:
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st63;
tr212:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st63;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
/* #line 6362 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 49u: goto tr210;
		case 65u: goto tr211;
		case 97u: goto tr211;
		case 128u: goto tr212;
		case 129u: goto tr211;
		case 169u: goto tr213;
		case 183u: goto tr213;
	}
	if ( 208u <= (*p) && (*p) <= 210u )
		goto tr213;
	goto tr209;
tr244:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st132;
tr213:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 10 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p, TOKEN_WORD);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st132;
st132:
	if ( ++p == pe )
		goto _test_eof132;
case 132:
/* #line 6407 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 35u: goto tr257;
		case 39u: goto tr258;
		case 43u: goto tr259;
		case 45u: goto tr260;
		case 46u: goto tr261;
		case 47u: goto tr262;
		case 64u: goto tr264;
		case 95u: goto tr266;
		case 143u: goto tr267;
	}
	goto tr256;
tr360:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st133;
tr239:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        ResetSentenceBreak();
    }
	goto st133;
st133:
	if ( ++p == pe )
		goto _test_eof133;
case 133:
/* #line 6436 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 0u: goto st64;
		case 9u: goto st68;
		case 10u: goto st64;
		case 13u: goto st64;
		case 32u: goto st68;
		case 39u: goto st69;
		case 46u: goto tr360;
		case 64u: goto st69;
		case 95u: goto st69;
		case 128u: goto st69;
		case 143u: goto st69;
		case 160u: goto st68;
		case 167u: goto st69;
		case 176u: goto st69;
		case 182u: goto st68;
		case 187u: goto tr360;
		case 192u: goto st69;
	}
	if ( (*p) < 45u ) {
		if ( (*p) > 37u ) {
			if ( 42u <= (*p) && (*p) <= 43u )
				goto st69;
		} else if ( (*p) >= 34u )
			goto st69;
	} else if ( (*p) > 47u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st69;
		} else if ( (*p) >= 178u )
			goto st69;
	} else
		goto st69;
	goto tr249;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
	switch( (*p) ) {
		case 0u: goto st64;
		case 13u: goto st64;
		case 32u: goto st64;
		case 37u: goto st65;
		case 39u: goto tr215;
		case 45u: goto tr215;
		case 49u: goto tr217;
		case 64u: goto tr215;
		case 65u: goto tr217;
		case 95u: goto st65;
		case 128u: goto st65;
		case 129u: goto tr217;
		case 143u: goto st65;
		case 160u: goto st64;
		case 167u: goto st65;
		case 169u: goto tr218;
		case 176u: goto st65;
		case 178u: goto tr215;
		case 179u: goto st65;
		case 182u: goto st64;
		case 187u: goto st65;
		case 192u: goto st65;
	}
	if ( (*p) < 42u ) {
		if ( (*p) > 10u ) {
			if ( 34u <= (*p) && (*p) <= 36u )
				goto tr215;
		} else if ( (*p) >= 9u )
			goto st64;
	} else if ( (*p) > 43u ) {
		if ( (*p) > 47u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st65;
		} else if ( (*p) >= 46u )
			goto st65;
	} else
		goto tr215;
	goto tr0;
tr215:
/* #line 37 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        MarkSentenceBreak(p);
    }
	goto st65;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
/* #line 6524 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 9u: goto st65;
		case 32u: goto st65;
		case 37u: goto st65;
		case 39u: goto tr215;
		case 45u: goto tr215;
		case 49u: goto tr217;
		case 64u: goto tr215;
		case 65u: goto tr217;
		case 95u: goto st65;
		case 128u: goto st65;
		case 129u: goto tr217;
		case 143u: goto st65;
		case 160u: goto st65;
		case 167u: goto st65;
		case 169u: goto tr218;
		case 176u: goto st65;
		case 178u: goto tr215;
		case 179u: goto st65;
		case 182u: goto st65;
		case 187u: goto st65;
		case 192u: goto st65;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 36u )
			goto tr215;
	} else if ( (*p) > 43u ) {
		if ( (*p) > 47u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st65;
		} else if ( (*p) >= 46u )
			goto st65;
	} else
		goto tr215;
	goto tr0;
tr218:
/* #line 37 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        MarkSentenceBreak(p);
    }
	goto st66;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
/* #line 6570 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 32u: goto st67;
		case 39u: goto st66;
		case 45u: goto st66;
		case 49u: goto tr217;
		case 64u: goto st66;
		case 65u: goto tr217;
		case 129u: goto tr217;
		case 160u: goto st67;
		case 169u: goto st66;
		case 178u: goto st66;
	}
	if ( (*p) > 36u ) {
		if ( 42u <= (*p) && (*p) <= 43u )
			goto st66;
	} else if ( (*p) >= 34u )
		goto st66;
	goto tr0;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
	switch( (*p) ) {
		case 32u: goto st67;
		case 49u: goto tr217;
		case 65u: goto tr217;
		case 129u: goto tr217;
		case 160u: goto st67;
	}
	goto tr0;
tr222:
/* #line 37 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        MarkSentenceBreak(p);
    }
	goto st68;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
/* #line 6611 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 0u: goto st64;
		case 9u: goto st68;
		case 10u: goto st64;
		case 13u: goto st64;
		case 32u: goto st68;
		case 37u: goto st68;
		case 39u: goto tr222;
		case 45u: goto tr222;
		case 49u: goto tr217;
		case 64u: goto tr222;
		case 65u: goto tr217;
		case 95u: goto st68;
		case 128u: goto st68;
		case 129u: goto tr217;
		case 143u: goto st68;
		case 160u: goto st68;
		case 167u: goto st68;
		case 169u: goto tr218;
		case 176u: goto st68;
		case 178u: goto tr222;
		case 179u: goto st68;
		case 182u: goto st68;
		case 187u: goto st68;
		case 192u: goto st68;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 36u )
			goto tr222;
	} else if ( (*p) > 43u ) {
		if ( (*p) > 47u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st68;
		} else if ( (*p) >= 46u )
			goto st68;
	} else
		goto tr222;
	goto tr0;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
	switch( (*p) ) {
		case 0u: goto st64;
		case 9u: goto st68;
		case 10u: goto st64;
		case 13u: goto st64;
		case 32u: goto st68;
		case 39u: goto st69;
		case 64u: goto st69;
		case 95u: goto st69;
		case 128u: goto st69;
		case 143u: goto st69;
		case 160u: goto st68;
		case 167u: goto st69;
		case 176u: goto st69;
		case 182u: goto st68;
		case 187u: goto st69;
		case 192u: goto st69;
	}
	if ( (*p) < 45u ) {
		if ( (*p) > 37u ) {
			if ( 42u <= (*p) && (*p) <= 43u )
				goto st69;
		} else if ( (*p) >= 34u )
			goto st69;
	} else if ( (*p) > 47u ) {
		if ( (*p) > 179u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st69;
		} else if ( (*p) >= 178u )
			goto st69;
	} else
		goto st69;
	goto tr0;
tr242:
/* #line 6 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        BeginToken(ts, p);
    }
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st134;
tr361:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/multitoken_v2.rl" */
	{
        UpdateToken();
    }
	goto st134;
st134:
	if ( ++p == pe )
		goto _test_eof134;
case 134:
/* #line 6707 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 9u: goto st77;
		case 32u: goto st77;
		case 34u: goto st77;
		case 47u: goto st77;
		case 49u: goto tr210;
		case 65u: goto tr211;
		case 95u: goto st77;
		case 97u: goto tr211;
		case 128u: goto tr361;
		case 129u: goto tr211;
		case 143u: goto st77;
		case 160u: goto st77;
		case 167u: goto st77;
		case 169u: goto tr213;
		case 176u: goto st77;
		case 182u: goto st77;
		case 183u: goto tr213;
		case 192u: goto st77;
	}
	if ( (*p) < 178u ) {
		if ( (*p) > 39u ) {
			if ( 42u <= (*p) && (*p) <= 45u )
				goto st77;
		} else if ( (*p) >= 37u )
			goto st77;
	} else if ( (*p) > 179u ) {
		if ( (*p) > 190u ) {
			if ( 208u <= (*p) && (*p) <= 210u )
				goto tr213;
		} else if ( (*p) >= 189u )
			goto st77;
	} else
		goto st77;
	goto tr248;
st135:
	if ( ++p == pe )
		goto _test_eof135;
case 135:
	if ( (*p) == 159u )
		goto st135;
	goto tr362;
st136:
	if ( ++p == pe )
		goto _test_eof136;
case 136:
	if ( 180u <= (*p) && (*p) <= 181u )
		goto st136;
	goto tr363;
tr364:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st137;
tr246:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        ResetSentenceBreak();
    }
	goto st137;
st137:
	if ( ++p == pe )
		goto _test_eof137;
case 137:
/* #line 6773 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 0u: goto st70;
		case 9u: goto st74;
		case 10u: goto st70;
		case 13u: goto st70;
		case 32u: goto st74;
		case 37u: goto st74;
		case 38u: goto tr247;
		case 44u: goto tr247;
		case 49u: goto tr217;
		case 64u: goto tr232;
		case 65u: goto tr217;
		case 95u: goto st74;
		case 97u: goto tr247;
		case 128u: goto st74;
		case 129u: goto tr217;
		case 143u: goto st74;
		case 159u: goto tr217;
		case 160u: goto st74;
		case 167u: goto st74;
		case 169u: goto tr228;
		case 176u: goto st74;
		case 178u: goto tr232;
		case 179u: goto st74;
		case 182u: goto st74;
		case 186u: goto tr364;
		case 187u: goto st74;
		case 192u: goto st74;
	}
	if ( (*p) < 46u ) {
		if ( (*p) > 39u ) {
			if ( 42u <= (*p) && (*p) <= 45u )
				goto tr232;
		} else if ( (*p) >= 34u )
			goto tr232;
	} else if ( (*p) > 47u ) {
		if ( (*p) < 189u ) {
			if ( 180u <= (*p) && (*p) <= 183u )
				goto tr247;
		} else if ( (*p) > 190u ) {
			if ( 208u <= (*p) && (*p) <= 210u )
				goto tr247;
		} else
			goto st74;
	} else
		goto st74;
	goto st76;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
	switch( (*p) ) {
		case 0u: goto st70;
		case 13u: goto st70;
		case 32u: goto st70;
		case 37u: goto st71;
		case 39u: goto tr226;
		case 45u: goto tr226;
		case 49u: goto tr217;
		case 64u: goto tr226;
		case 65u: goto tr217;
		case 95u: goto st71;
		case 128u: goto st71;
		case 129u: goto tr217;
		case 143u: goto st71;
		case 159u: goto tr217;
		case 160u: goto st70;
		case 167u: goto st71;
		case 169u: goto tr228;
		case 176u: goto st71;
		case 178u: goto tr226;
		case 179u: goto st71;
		case 182u: goto st70;
		case 187u: goto st71;
		case 192u: goto st71;
	}
	if ( (*p) < 42u ) {
		if ( (*p) > 10u ) {
			if ( 34u <= (*p) && (*p) <= 36u )
				goto tr226;
		} else if ( (*p) >= 9u )
			goto st70;
	} else if ( (*p) > 43u ) {
		if ( (*p) > 47u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st71;
		} else if ( (*p) >= 46u )
			goto st71;
	} else
		goto tr226;
	goto tr224;
tr226:
/* #line 37 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        MarkSentenceBreak(p);
    }
	goto st71;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
/* #line 6875 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 9u: goto st71;
		case 32u: goto st71;
		case 37u: goto st71;
		case 39u: goto tr226;
		case 45u: goto tr226;
		case 49u: goto tr217;
		case 64u: goto tr226;
		case 65u: goto tr217;
		case 95u: goto st71;
		case 128u: goto st71;
		case 129u: goto tr217;
		case 143u: goto st71;
		case 159u: goto tr217;
		case 160u: goto st71;
		case 167u: goto st71;
		case 169u: goto tr228;
		case 176u: goto st71;
		case 178u: goto tr226;
		case 179u: goto st71;
		case 182u: goto st71;
		case 187u: goto st71;
		case 192u: goto st71;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 36u )
			goto tr226;
	} else if ( (*p) > 43u ) {
		if ( (*p) > 47u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st71;
		} else if ( (*p) >= 46u )
			goto st71;
	} else
		goto tr226;
	goto tr224;
tr228:
/* #line 37 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        MarkSentenceBreak(p);
    }
	goto st72;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
/* #line 6922 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 32u: goto st73;
		case 39u: goto st72;
		case 45u: goto st72;
		case 49u: goto tr217;
		case 64u: goto st72;
		case 65u: goto tr217;
		case 129u: goto tr217;
		case 159u: goto tr217;
		case 160u: goto st73;
		case 169u: goto st72;
		case 178u: goto st72;
	}
	if ( (*p) > 36u ) {
		if ( 42u <= (*p) && (*p) <= 43u )
			goto st72;
	} else if ( (*p) >= 34u )
		goto st72;
	goto tr224;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
	switch( (*p) ) {
		case 32u: goto st73;
		case 49u: goto tr217;
		case 65u: goto tr217;
		case 129u: goto tr217;
		case 159u: goto tr217;
		case 160u: goto st73;
	}
	goto tr224;
tr232:
/* #line 37 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */
	{
        MarkSentenceBreak(p);
    }
	goto st74;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
/* #line 6965 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/tokenizer/nlptok_v2.rl6.cpp" */
	switch( (*p) ) {
		case 0u: goto st70;
		case 9u: goto st74;
		case 10u: goto st70;
		case 13u: goto st70;
		case 32u: goto st74;
		case 37u: goto st74;
		case 39u: goto tr232;
		case 45u: goto tr232;
		case 49u: goto tr217;
		case 64u: goto tr232;
		case 65u: goto tr217;
		case 95u: goto st74;
		case 128u: goto st74;
		case 129u: goto tr217;
		case 143u: goto st74;
		case 159u: goto tr217;
		case 160u: goto st74;
		case 167u: goto st74;
		case 169u: goto tr228;
		case 176u: goto st74;
		case 178u: goto tr232;
		case 179u: goto st74;
		case 182u: goto st74;
		case 187u: goto st74;
		case 192u: goto st74;
	}
	if ( (*p) < 42u ) {
		if ( 34u <= (*p) && (*p) <= 36u )
			goto tr232;
	} else if ( (*p) > 43u ) {
		if ( (*p) > 47u ) {
			if ( 189u <= (*p) && (*p) <= 190u )
				goto st74;
		} else if ( (*p) >= 46u )
			goto st74;
	} else
		goto tr232;
	goto tr224;
	}
	_test_eof75: cs = 75; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof0: cs = 0; goto _test_eof; 
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof83: cs = 83; goto _test_eof; 
	_test_eof84: cs = 84; goto _test_eof; 
	_test_eof85: cs = 85; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof86: cs = 86; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof87: cs = 87; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof88: cs = 88; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof89: cs = 89; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof90: cs = 90; goto _test_eof; 
	_test_eof91: cs = 91; goto _test_eof; 
	_test_eof92: cs = 92; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof93: cs = 93; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof94: cs = 94; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof95: cs = 95; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof96: cs = 96; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof97: cs = 97; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof98: cs = 98; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof99: cs = 99; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof100: cs = 100; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof101: cs = 101; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof102: cs = 102; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof103: cs = 103; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof104: cs = 104; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof105: cs = 105; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof106: cs = 106; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof107: cs = 107; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof108: cs = 108; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof109: cs = 109; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof110: cs = 110; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof111: cs = 111; goto _test_eof; 
	_test_eof112: cs = 112; goto _test_eof; 
	_test_eof113: cs = 113; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof114: cs = 114; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof115: cs = 115; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof116: cs = 116; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof117: cs = 117; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof118: cs = 118; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof119: cs = 119; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof120: cs = 120; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof121: cs = 121; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof122: cs = 122; goto _test_eof; 
	_test_eof123: cs = 123; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof124: cs = 124; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof125: cs = 125; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof126: cs = 126; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof127: cs = 127; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof128: cs = 128; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof129: cs = 129; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof130: cs = 130; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof131: cs = 131; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof132: cs = 132; goto _test_eof; 
	_test_eof133: cs = 133; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof134: cs = 134; goto _test_eof; 
	_test_eof135: cs = 135; goto _test_eof; 
	_test_eof136: cs = 136; goto _test_eof; 
	_test_eof137: cs = 137; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 

	_test_eof: {}
	if ( p == eof )
	{
	switch ( cs ) {
	case 76: goto tr247;
	case 77: goto tr248;
	case 78: goto tr249;
	case 0: goto tr0;
	case 1: goto tr0;
	case 2: goto tr0;
	case 3: goto tr0;
	case 79: goto tr251;
	case 4: goto tr0;
	case 80: goto tr251;
	case 81: goto tr249;
	case 5: goto tr0;
	case 82: goto tr251;
	case 83: goto tr254;
	case 84: goto tr256;
	case 85: goto tr268;
	case 6: goto tr7;
	case 7: goto tr7;
	case 86: goto tr256;
	case 8: goto tr7;
	case 87: goto tr256;
	case 9: goto tr7;
	case 88: goto tr256;
	case 10: goto tr7;
	case 89: goto tr256;
	case 11: goto tr7;
	case 90: goto tr256;
	case 91: goto tr268;
	case 92: goto tr268;
	case 12: goto tr7;
	case 93: goto tr256;
	case 13: goto tr7;
	case 94: goto tr256;
	case 14: goto tr7;
	case 95: goto tr256;
	case 15: goto tr7;
	case 96: goto tr256;
	case 16: goto tr7;
	case 97: goto tr256;
	case 17: goto tr7;
	case 18: goto tr7;
	case 98: goto tr256;
	case 19: goto tr7;
	case 20: goto tr7;
	case 21: goto tr7;
	case 99: goto tr256;
	case 22: goto tr7;
	case 100: goto tr256;
	case 23: goto tr7;
	case 101: goto tr256;
	case 24: goto tr7;
	case 102: goto tr256;
	case 25: goto tr7;
	case 103: goto tr256;
	case 26: goto tr7;
	case 104: goto tr256;
	case 27: goto tr7;
	case 105: goto tr256;
	case 28: goto tr7;
	case 106: goto tr256;
	case 29: goto tr7;
	case 30: goto tr7;
	case 31: goto tr7;
	case 32: goto tr7;
	case 33: goto tr7;
	case 34: goto tr7;
	case 107: goto tr256;
	case 35: goto tr7;
	case 108: goto tr256;
	case 36: goto tr7;
	case 109: goto tr256;
	case 37: goto tr7;
	case 110: goto tr256;
	case 38: goto tr7;
	case 111: goto tr256;
	case 112: goto tr268;
	case 113: goto tr256;
	case 39: goto tr7;
	case 114: goto tr256;
	case 40: goto tr7;
	case 41: goto tr7;
	case 42: goto tr7;
	case 43: goto tr7;
	case 115: goto tr256;
	case 44: goto tr7;
	case 45: goto tr7;
	case 46: goto tr7;
	case 47: goto tr7;
	case 116: goto tr256;
	case 48: goto tr7;
	case 117: goto tr256;
	case 49: goto tr7;
	case 118: goto tr256;
	case 50: goto tr7;
	case 51: goto tr7;
	case 119: goto tr256;
	case 52: goto tr7;
	case 120: goto tr256;
	case 53: goto tr7;
	case 121: goto tr256;
	case 54: goto tr7;
	case 122: goto tr256;
	case 123: goto tr256;
	case 55: goto tr7;
	case 124: goto tr256;
	case 56: goto tr7;
	case 125: goto tr256;
	case 57: goto tr7;
	case 126: goto tr256;
	case 58: goto tr7;
	case 127: goto tr256;
	case 59: goto tr7;
	case 128: goto tr256;
	case 60: goto tr7;
	case 129: goto tr256;
	case 61: goto tr7;
	case 130: goto tr256;
	case 62: goto tr7;
	case 131: goto tr256;
	case 63: goto tr209;
	case 132: goto tr256;
	case 133: goto tr249;
	case 64: goto tr0;
	case 65: goto tr0;
	case 66: goto tr0;
	case 67: goto tr0;
	case 68: goto tr0;
	case 69: goto tr0;
	case 134: goto tr248;
	case 135: goto tr362;
	case 136: goto tr363;
	case 137: goto tr247;
	case 70: goto tr224;
	case 71: goto tr224;
	case 72: goto tr224;
	case 73: goto tr224;
	case 74: goto tr224;
	}
	}

	}

/* #line 169 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/tokenizer/nlptok_v2.rl6" */

    if (cs == NlpLexer_error)
        throw yexception() << "execute error at position " << static_cast<int>(p - text);
    else if (cs < NlpLexer_first_final)
        throw yexception() << "execute finished in non-final state";

    Y_UNUSED(act);
}

