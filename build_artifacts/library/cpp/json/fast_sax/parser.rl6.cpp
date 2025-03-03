
/* #line 1 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
#include <library/cpp/json/fast_sax/unescape.h>
#include <library/cpp/json/fast_sax/parser.h>

#include <util/string/cast.h>
#include <util/generic/buffer.h>
#include <util/generic/strbuf.h>
#include <util/generic/ymath.h>

namespace NJson {

enum EStoredStr {
    SS_NONE = 0, SS_NOCOPY, SS_MUSTCOPY
};

struct TParserCtx {
    TJsonCallbacks& Hndl;

    TBuffer Buffer;
    TStringBuf String;
    EStoredStr Stored = SS_NONE;
    bool ExpectValue = true;

    const char* p0 = nullptr;
    const char* p = nullptr;
    const char* pe = nullptr;
    const char* eof = nullptr;
    const char* ts = nullptr;
    const char* te = nullptr;
    int cs = 0;
    int act = 0;

    TParserCtx(TJsonCallbacks& h, TStringBuf data)
        : Hndl(h)
        , p0(data.data())
        , p(data.data())
        , pe(data.end())
        , eof(data.end())
    {}

    static inline bool GoodPtrs(const char* b, const char* e) {
        return b && e && b <= e;
    }

    bool OnError(TStringBuf reason = TStringBuf(""), bool end = false) const {
        size_t off = 0;
        TStringBuf token;

        if (GoodPtrs(p0, ts)) {
            off = ts - p0;
        } else if (end && GoodPtrs(p0, pe)) {
            off = pe - p0;
        }

        if (GoodPtrs(ts, te)) {
            token = TStringBuf(ts, te);
        }

        if (!token) {
            Hndl.OnError(off, reason);
        } else {
            Hndl.OnError(off, TString::Join(reason, " at token: '", token, "'"));
        }

        return false;
    }

    bool OnVal() {
        if (Y_UNLIKELY(!ExpectValue)) {
            return false;
        }
        ExpectValue = false;
        return true;
    }

    bool OnNull() {
        return Y_LIKELY(OnVal())
               && Hndl.OnNull();
    }

    bool OnTrue() {
        return Y_LIKELY(OnVal())
               && Hndl.OnBoolean(true);
    }

    bool OnFalse() {
        return Y_LIKELY(OnVal())
               && Hndl.OnBoolean(false);
    }

    bool OnPInt() {
        unsigned long long res = 0;
        return Y_LIKELY(OnVal())
               && TryFromString<unsigned long long>(TStringBuf(ts, te), res)
               && Hndl.OnUInteger(res);
    }

    bool OnNInt() {
        long long res = 0;
        return Y_LIKELY(OnVal())
               && TryFromString<long long>(TStringBuf(ts, te), res)
               && Hndl.OnInteger(res);
    }

    bool OnFlt() {
        double res = 0;
        return Y_LIKELY(OnVal())
               && TryFromString<double>(TStringBuf(ts, te), res)
               && IsFinite(res)
               && Hndl.OnDouble(res);
    }

    bool OnMapOpen() {
        bool res = Y_LIKELY(OnVal())
                   && Hndl.OnOpenMap();
        ExpectValue = true;
        return res;
    }

    bool OnArrOpen() {
        bool res = Y_LIKELY(OnVal())
                   && Hndl.OnOpenArray();
        ExpectValue = true;
        return res;
    }

    bool OnString(TStringBuf s, EStoredStr t) {
        if (Y_LIKELY(OnVal())) {
            String = s;
            Stored = t;
            return true;
        } else {
            return false;
        }
    }

    bool OnStrU() {
        return OnString(TStringBuf(ts, te), SS_NOCOPY);
    }

    bool OnStrQ() {
        return OnString(TStringBuf(ts + 1, te - 1), SS_NOCOPY);
    }

    bool OnStrE() {
        Buffer.Clear();
        Buffer.Reserve(2 * (te - ts));

        return OnString(UnescapeJsonUnicode(TStringBuf(ts + 1, te - ts - 2), Buffer.data()), SS_MUSTCOPY);
    }

    bool OnMapClose() {
        ExpectValue = false;
        return Y_LIKELY(OnAfterVal())
               && Hndl.OnCloseMap();
    }

    bool OnArrClose() {
        ExpectValue = false;
        return Y_LIKELY(OnAfterVal())
               && Hndl.OnCloseArray();
    }

    bool OnColon() {
        if (ExpectValue) {
            return false;
        }

        ExpectValue = true;
        const auto stored = Stored;
        Stored = SS_NONE;

        switch (stored) {
        default:
            return false;
        case SS_NOCOPY:
            return Hndl.OnMapKeyNoCopy(String);
        case SS_MUSTCOPY:
            return Hndl.OnMapKey(String);
        }
    }

    bool OnAfterVal() {
        const auto stored = Stored;
        Stored = SS_NONE;

        switch (stored) {
        default:
            return true;
        case SS_NOCOPY:
            return Hndl.OnStringNoCopy(String);
        case SS_MUSTCOPY:
            return Hndl.OnString(String);
        }
    }

    bool OnComma() {
        if (Y_UNLIKELY(ExpectValue)) {
            return false;
        }
        ExpectValue = true;
        return OnAfterVal();
    }

    bool Parse();
};

#if 0

/* #line 288 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */

#endif

bool TParserCtx::Parse() {
    try {
        
/* #line 219 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
static const int fastjson_start = 9;

static const int fastjson_en_main = 9;


/* #line 225 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	{
	cs = fastjson_start;
	ts = 0;
	te = 0;
	act = 0;
	}

/* #line 233 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
tr0:
/* #line 228 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{{p = ((te))-1;}{ goto TOKEN_ERROR; }}
	goto st9;
tr2:
/* #line 220 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnStrQ()))  goto TOKEN_ERROR; }}
	goto st9;
tr5:
/* #line 221 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnStrE()))  goto TOKEN_ERROR; }}
	goto st9;
tr12:
/* #line 282 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;}
	goto st9;
tr15:
/* #line 228 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ goto TOKEN_ERROR; }}
	goto st9;
tr18:
/* #line 226 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnComma())) goto TOKEN_ERROR; }}
	goto st9;
tr23:
/* #line 227 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnColon())) goto TOKEN_ERROR; }}
	goto st9;
tr24:
/* #line 224 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnArrOpen()))  goto TOKEN_ERROR; }}
	goto st9;
tr25:
/* #line 225 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnArrClose())) goto TOKEN_ERROR; }}
	goto st9;
tr29:
/* #line 222 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnMapOpen()))  goto TOKEN_ERROR; }}
	goto st9;
tr30:
/* #line 223 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ if (Y_UNLIKELY(!OnMapClose())) goto TOKEN_ERROR; }}
	goto st9;
tr31:
/* #line 1 "NONE" */
	{	switch( act ) {
	case 1:
	{{p = ((te))-1;} if (Y_UNLIKELY(!OnNull()))  goto TOKEN_ERROR; }
	break;
	case 2:
	{{p = ((te))-1;} if (Y_UNLIKELY(!OnTrue()))  goto TOKEN_ERROR; }
	break;
	case 3:
	{{p = ((te))-1;} if (Y_UNLIKELY(!OnFalse())) goto TOKEN_ERROR; }
	break;
	case 7:
	{{p = ((te))-1;} if (Y_UNLIKELY(!OnStrU()))  goto TOKEN_ERROR; }
	break;
	}
	}
	goto st9;
tr32:
/* #line 281 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p;p--;}
	goto st9;
tr33:
/* #line 228 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p;p--;{ goto TOKEN_ERROR; }}
	goto st9;
tr36:
/* #line 218 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p;p--;{ if (Y_UNLIKELY(!OnFlt()))   goto TOKEN_ERROR; }}
	goto st9;
tr37:
/* #line 228 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p+1;{ goto TOKEN_ERROR; }}
	goto st9;
tr38:
/* #line 217 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p;p--;{ if (Y_UNLIKELY(!OnNInt()))  goto TOKEN_ERROR; }}
	goto st9;
tr39:
/* #line 216 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p;p--;{ if (Y_UNLIKELY(!OnPInt()))  goto TOKEN_ERROR; }}
	goto st9;
tr40:
/* #line 219 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{te = p;p--;{ if (Y_UNLIKELY(!OnStrU()))  goto TOKEN_ERROR; }}
	goto st9;
st9:
/* #line 1 "NONE" */
	{ts = 0;}
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 1 "NONE" */
	{ts = p;}
/* #line 337 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	switch( (*p) ) {
		case 34: goto tr16;
		case 39: goto tr17;
		case 44: goto tr18;
		case 45: goto tr19;
		case 46: goto st17;
		case 47: goto tr21;
		case 58: goto tr23;
		case 91: goto tr24;
		case 93: goto tr25;
		case 96: goto tr15;
		case 102: goto st20;
		case 110: goto st24;
		case 116: goto st27;
		case 123: goto tr29;
		case 125: goto tr30;
		case 127: goto st11;
	}
	if ( (*p) < 48 ) {
		if ( (*p) < 33 ) {
			if ( 0 <= (*p) && (*p) <= 32 )
				goto st11;
		} else if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 43 )
				goto tr15;
		} else
			goto tr15;
	} else if ( (*p) > 57 ) {
		if ( (*p) < 92 ) {
			if ( 59 <= (*p) && (*p) <= 63 )
				goto tr15;
		} else if ( (*p) > 94 ) {
			if ( 124 <= (*p) && (*p) <= 126 )
				goto tr15;
		} else
			goto tr15;
	} else
		goto st19;
	goto tr13;
tr13:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 219 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{act = 7;}
	goto st10;
tr44:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 215 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{act = 3;}
	goto st10;
tr47:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{act = 1;}
	goto st10;
tr50:
/* #line 1 "NONE" */
	{te = p+1;}
/* #line 214 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */
	{act = 2;}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 405 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	switch( (*p) ) {
		case 47: goto tr31;
		case 96: goto tr31;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr31;
		} else if ( (*p) >= 0 )
			goto tr31;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr31;
		} else if ( (*p) >= 91 )
			goto tr31;
	} else
		goto tr31;
	goto tr13;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
	if ( (*p) == 127 )
		goto st11;
	if ( 0 <= (*p) && (*p) <= 32 )
		goto st11;
	goto tr32;
tr16:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 442 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	switch( (*p) ) {
		case 34: goto tr2;
		case 92: goto st1;
	}
	goto st0;
st0:
	if ( ++p == pe )
		goto _test_eof0;
case 0:
	switch( (*p) ) {
		case 34: goto tr2;
		case 92: goto st1;
	}
	goto st0;
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
	switch( (*p) ) {
		case 34: goto tr5;
		case 92: goto st1;
	}
	goto st2;
tr17:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 479 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	switch( (*p) ) {
		case 39: goto tr2;
		case 92: goto st4;
	}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	switch( (*p) ) {
		case 39: goto tr2;
		case 92: goto st4;
	}
	goto st3;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	switch( (*p) ) {
		case 39: goto tr5;
		case 92: goto st4;
	}
	goto st5;
tr19:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 516 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	if ( (*p) == 46 )
		goto st6;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st16;
	goto tr33;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	switch( (*p) ) {
		case 43: goto st15;
		case 69: goto st15;
		case 101: goto st15;
	}
	if ( (*p) > 46 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st15;
	} else if ( (*p) >= 45 )
		goto st15;
	goto tr0;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
	switch( (*p) ) {
		case 44: goto tr36;
		case 58: goto tr36;
		case 69: goto st15;
		case 91: goto tr36;
		case 93: goto tr36;
		case 101: goto st15;
		case 123: goto tr36;
		case 125: goto tr36;
		case 127: goto tr36;
	}
	if ( (*p) < 43 ) {
		if ( 0 <= (*p) && (*p) <= 32 )
			goto tr36;
	} else if ( (*p) > 46 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st15;
	} else
		goto st15;
	goto tr37;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
	switch( (*p) ) {
		case 44: goto tr38;
		case 58: goto tr38;
		case 69: goto st15;
		case 91: goto tr38;
		case 93: goto tr38;
		case 101: goto st15;
		case 123: goto tr38;
		case 125: goto tr38;
		case 127: goto tr38;
	}
	if ( (*p) < 43 ) {
		if ( 0 <= (*p) && (*p) <= 32 )
			goto tr38;
	} else if ( (*p) > 46 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st16;
	} else
		goto st15;
	goto tr37;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
	switch( (*p) ) {
		case 43: goto st15;
		case 69: goto st15;
		case 101: goto st15;
	}
	if ( (*p) > 46 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st15;
	} else if ( (*p) >= 45 )
		goto st15;
	goto tr33;
tr21:
/* #line 1 "NONE" */
	{te = p+1;}
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 608 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/library/cpp/json/fast_sax/parser.rl6.cpp" */
	if ( (*p) == 42 )
		goto st7;
	goto tr33;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
	if ( (*p) == 42 )
		goto st8;
	goto st7;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
	switch( (*p) ) {
		case 42: goto st8;
		case 47: goto tr12;
	}
	goto st7;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
	switch( (*p) ) {
		case 44: goto tr39;
		case 58: goto tr39;
		case 69: goto st15;
		case 91: goto tr39;
		case 93: goto tr39;
		case 101: goto st15;
		case 123: goto tr39;
		case 125: goto tr39;
		case 127: goto tr39;
	}
	if ( (*p) < 43 ) {
		if ( 0 <= (*p) && (*p) <= 32 )
			goto tr39;
	} else if ( (*p) > 46 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st19;
	} else
		goto st15;
	goto tr37;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 97: goto st21;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 108: goto st22;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 115: goto st23;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 101: goto tr44;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 117: goto st25;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 108: goto st26;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 108: goto tr47;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 114: goto st28;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 117: goto st29;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
	switch( (*p) ) {
		case 47: goto tr40;
		case 96: goto tr40;
		case 101: goto tr50;
	}
	if ( (*p) < 58 ) {
		if ( (*p) > 35 ) {
			if ( 37 <= (*p) && (*p) <= 44 )
				goto tr40;
		} else if ( (*p) >= 0 )
			goto tr40;
	} else if ( (*p) > 63 ) {
		if ( (*p) > 94 ) {
			if ( 123 <= (*p) )
				goto tr40;
		} else if ( (*p) >= 91 )
			goto tr40;
	} else
		goto tr40;
	goto tr13;
	}
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof0: cs = 0; goto _test_eof; 
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 

	_test_eof: {}
	if ( p == eof )
	{
	switch ( cs ) {
	case 10: goto tr31;
	case 11: goto tr32;
	case 12: goto tr33;
	case 0: goto tr0;
	case 1: goto tr0;
	case 2: goto tr0;
	case 13: goto tr33;
	case 3: goto tr0;
	case 4: goto tr0;
	case 5: goto tr0;
	case 14: goto tr33;
	case 6: goto tr0;
	case 15: goto tr36;
	case 16: goto tr38;
	case 17: goto tr33;
	case 18: goto tr33;
	case 7: goto tr0;
	case 8: goto tr0;
	case 19: goto tr39;
	case 20: goto tr40;
	case 21: goto tr40;
	case 22: goto tr40;
	case 23: goto tr40;
	case 24: goto tr40;
	case 25: goto tr40;
	case 26: goto tr40;
	case 27: goto tr40;
	case 28: goto tr40;
	case 29: goto tr40;
	}
	}

	}

/* #line 297 "/Users/makar/Documents/Course work/Repository/catboost/library/cpp/json/fast_sax/parser.rl6" */

        Y_UNUSED(fastjson_en_main);
    } catch (const TFromStringException& e) {
        return OnError(e.what());
    }

    return OnAfterVal() && Hndl.OnEnd() || OnError("invalid or truncated", true);

    TOKEN_ERROR:
    return OnError("invalid syntax");
}

bool ReadJsonFast(TStringBuf data, TJsonCallbacks* h) {
    return TParserCtx(*h, data).Parse();
}

}
