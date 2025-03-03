
/* #line 1 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <numeric>

#include <util/datetime/parser.h>
#include <util/generic/ymath.h>



/* #line 86 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */



/* #line 20 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
static const int RFC822DateParser_start = 1;
static const int RFC822DateParser_first_final = 76;

static const int RFC822DateParser_en_main = 1;


/* #line 165 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */


TRfc822DateTimeParserDeprecated::TRfc822DateTimeParserDeprecated() {
    
/* #line 32 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = RFC822DateParser_start;
	}

/* #line 169 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TRfc822DateTimeParserDeprecated::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 45 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	switch( (*p) ) {
		case 32: goto st1;
		case 70: goto st63;
		case 77: goto st67;
		case 83: goto st69;
		case 84: goto st71;
		case 87: goto st74;
		case 102: goto st63;
		case 109: goto st67;
		case 115: goto st69;
		case 116: goto st71;
		case 119: goto st74;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr2;
	} else if ( (*p) >= 9 )
		goto st1;
	goto st0;
st0:
cs = 0;
	goto _out;
tr2:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 93 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr8;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr9;
	} else if ( (*p) >= 9 )
		goto tr8;
	goto st0;
tr8:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 110 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st3;
		case 65: goto st4;
		case 68: goto st37;
		case 70: goto st40;
		case 74: goto st43;
		case 77: goto st49;
		case 78: goto st53;
		case 79: goto st56;
		case 83: goto st59;
		case 97: goto st4;
		case 100: goto st37;
		case 102: goto st40;
		case 106: goto st43;
		case 109: goto st49;
		case 110: goto st53;
		case 111: goto st56;
		case 115: goto st59;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st3;
	goto st0;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	switch( (*p) ) {
		case 80: goto st5;
		case 85: goto st35;
		case 112: goto st5;
		case 117: goto st35;
	}
	goto st0;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	switch( (*p) ) {
		case 82: goto st6;
		case 114: goto st6;
	}
	goto st0;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	if ( (*p) == 32 )
		goto tr22;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr22;
	goto st0;
tr22:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st7;
tr63:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st7;
tr66:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st7;
tr69:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st7;
tr73:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st7;
tr76:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st7;
tr77:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st7;
tr81:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st7;
tr82:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st7;
tr85:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st7;
tr88:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st7;
tr91:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 214 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st7;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr24;
	} else if ( (*p) >= 9 )
		goto st7;
	goto st0;
tr24:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 239 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr25;
	goto st0;
tr25:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 254 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr26;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr27;
	} else if ( (*p) >= 9 )
		goto tr26;
	goto st0;
tr26:
/* #line 82 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.SetLooseYear(I); }
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 271 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st10;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr29;
	} else if ( (*p) >= 9 )
		goto st10;
	goto st0;
tr29:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 296 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr30;
	goto st0;
tr30:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 311 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr31;
	goto st0;
tr31:
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 323 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr32;
	goto st0;
tr32:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 343 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr33;
	goto st0;
tr33:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 358 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto tr34;
		case 58: goto tr35;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr34;
	goto st0;
tr34:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st16;
tr60:
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 378 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st16;
		case 43: goto tr37;
		case 45: goto tr37;
		case 67: goto tr39;
		case 69: goto tr40;
		case 71: goto tr41;
		case 77: goto tr42;
		case 80: goto tr43;
		case 85: goto tr44;
	}
	if ( (*p) < 75 ) {
		if ( (*p) > 13 ) {
			if ( 65 <= (*p) && (*p) <= 73 )
				goto tr38;
		} else if ( (*p) >= 9 )
			goto st16;
	} else if ( (*p) > 90 ) {
		if ( (*p) > 105 ) {
			if ( 107 <= (*p) && (*p) <= 122 )
				goto tr38;
		} else if ( (*p) >= 97 )
			goto tr38;
	} else
		goto tr38;
	goto st0;
tr37:
/* #line 155 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ Sign = (*p) == '+' ? 1 : -1; }
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 413 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr45;
	goto st0;
tr45:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 433 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr46;
	goto st0;
tr46:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 448 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr47;
	goto st0;
tr47:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
/* #line 463 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr48;
	goto st0;
tr38:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st76;
tr48:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 133 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    DateTimeFields.ZoneOffsetMinutes = Sign * (i32)(TDuration::Hours(I / 100) + TDuration::Minutes(I % 100)).Minutes();
}
	goto st76;
tr49:
/* #line 149 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(5).Minutes(); }
	goto st76;
tr50:
/* #line 148 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(6).Minutes();}
	goto st76;
tr51:
/* #line 147 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(4).Minutes(); }
	goto st76;
tr52:
/* #line 146 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(5).Minutes();}
	goto st76;
tr53:
/* #line 145 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st76;
tr54:
/* #line 151 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(6).Minutes(); }
	goto st76;
tr55:
/* #line 150 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(7).Minutes();}
	goto st76;
tr56:
/* #line 153 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(7).Minutes(); }
	goto st76;
tr57:
/* #line 152 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(8).Minutes();}
	goto st76;
tr110:
/* #line 144 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st76;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
/* #line 542 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st76;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
tr39:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st77;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
/* #line 572 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st21;
		case 83: goto st22;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
	if ( (*p) == 84 )
		goto tr49;
	goto st0;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	if ( (*p) == 84 )
		goto tr50;
	goto st0;
tr40:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st78;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
/* #line 619 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st23;
		case 83: goto st24;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
	if ( (*p) == 84 )
		goto tr51;
	goto st0;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
	if ( (*p) == 84 )
		goto tr52;
	goto st0;
tr41:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st79;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
/* #line 666 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 77: goto st25;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
	if ( (*p) == 84 )
		goto tr53;
	goto st0;
tr42:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st80;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
/* #line 705 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st26;
		case 83: goto st27;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
	if ( (*p) == 84 )
		goto tr54;
	goto st0;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
	if ( (*p) == 84 )
		goto tr55;
	goto st0;
tr43:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st81;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
/* #line 752 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st28;
		case 83: goto st29;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	if ( (*p) == 84 )
		goto tr56;
	goto st0;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
	if ( (*p) == 84 )
		goto tr57;
	goto st0;
tr44:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st82;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
/* #line 799 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 84: goto tr110;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
tr35:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st30;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
/* #line 815 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr58;
	goto st0;
tr58:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st31;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
/* #line 835 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr59;
	goto st0;
tr59:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st32;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
/* #line 850 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr60;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr60;
	goto st0;
tr27:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st33;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
/* #line 867 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr61;
	goto st0;
tr61:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st34;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
/* #line 882 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr26;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr26;
	goto st0;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
	switch( (*p) ) {
		case 71: goto st36;
		case 103: goto st36;
	}
	goto st0;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
	if ( (*p) == 32 )
		goto tr63;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr63;
	goto st0;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
	switch( (*p) ) {
		case 69: goto st38;
		case 101: goto st38;
	}
	goto st0;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
	switch( (*p) ) {
		case 67: goto st39;
		case 99: goto st39;
	}
	goto st0;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
	if ( (*p) == 32 )
		goto tr66;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr66;
	goto st0;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
	switch( (*p) ) {
		case 69: goto st41;
		case 101: goto st41;
	}
	goto st0;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	switch( (*p) ) {
		case 66: goto st42;
		case 98: goto st42;
	}
	goto st0;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 32 )
		goto tr69;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr69;
	goto st0;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
	switch( (*p) ) {
		case 65: goto st44;
		case 85: goto st46;
		case 97: goto st44;
		case 117: goto st46;
	}
	goto st0;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
	switch( (*p) ) {
		case 78: goto st45;
		case 110: goto st45;
	}
	goto st0;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
	if ( (*p) == 32 )
		goto tr73;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr73;
	goto st0;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
	switch( (*p) ) {
		case 76: goto st47;
		case 78: goto st48;
		case 108: goto st47;
		case 110: goto st48;
	}
	goto st0;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	if ( (*p) == 32 )
		goto tr76;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr76;
	goto st0;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
	if ( (*p) == 32 )
		goto tr77;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr77;
	goto st0;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
	switch( (*p) ) {
		case 65: goto st50;
		case 97: goto st50;
	}
	goto st0;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
	switch( (*p) ) {
		case 82: goto st51;
		case 89: goto st52;
		case 114: goto st51;
		case 121: goto st52;
	}
	goto st0;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
	if ( (*p) == 32 )
		goto tr81;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr81;
	goto st0;
st52:
	if ( ++p == pe )
		goto _test_eof52;
case 52:
	if ( (*p) == 32 )
		goto tr82;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr82;
	goto st0;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
	switch( (*p) ) {
		case 79: goto st54;
		case 111: goto st54;
	}
	goto st0;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
	switch( (*p) ) {
		case 86: goto st55;
		case 118: goto st55;
	}
	goto st0;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
	if ( (*p) == 32 )
		goto tr85;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr85;
	goto st0;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
	switch( (*p) ) {
		case 67: goto st57;
		case 99: goto st57;
	}
	goto st0;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
	switch( (*p) ) {
		case 84: goto st58;
		case 116: goto st58;
	}
	goto st0;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
	if ( (*p) == 32 )
		goto tr88;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr88;
	goto st0;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
	switch( (*p) ) {
		case 69: goto st60;
		case 101: goto st60;
	}
	goto st0;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
	switch( (*p) ) {
		case 80: goto st61;
		case 112: goto st61;
	}
	goto st0;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
	if ( (*p) == 32 )
		goto tr91;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr91;
	goto st0;
tr9:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st62;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
/* #line 1148 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr8;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr8;
	goto st0;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
	switch( (*p) ) {
		case 82: goto st64;
		case 114: goto st64;
	}
	goto st0;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
	switch( (*p) ) {
		case 73: goto st65;
		case 105: goto st65;
	}
	goto st0;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
	if ( (*p) == 44 )
		goto st66;
	goto st0;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
	if ( (*p) == 32 )
		goto st66;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr2;
	} else if ( (*p) >= 9 )
		goto st66;
	goto st0;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
	switch( (*p) ) {
		case 79: goto st68;
		case 111: goto st68;
	}
	goto st0;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
	switch( (*p) ) {
		case 78: goto st65;
		case 110: goto st65;
	}
	goto st0;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
	switch( (*p) ) {
		case 65: goto st70;
		case 85: goto st68;
		case 97: goto st70;
		case 117: goto st68;
	}
	goto st0;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
	switch( (*p) ) {
		case 84: goto st65;
		case 116: goto st65;
	}
	goto st0;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
	switch( (*p) ) {
		case 72: goto st72;
		case 85: goto st73;
		case 104: goto st72;
		case 117: goto st73;
	}
	goto st0;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
	switch( (*p) ) {
		case 85: goto st65;
		case 117: goto st65;
	}
	goto st0;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
	switch( (*p) ) {
		case 69: goto st65;
		case 101: goto st65;
	}
	goto st0;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
	switch( (*p) ) {
		case 69: goto st75;
		case 101: goto st75;
	}
	goto st0;
st75:
	if ( ++p == pe )
		goto _test_eof75;
case 75:
	switch( (*p) ) {
		case 68: goto st65;
		case 100: goto st65;
	}
	goto st0;
	}
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 
	_test_eof75: cs = 75; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 176 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

TRfc822DateTimeParser::TRfc822DateTimeParser() {
    
/* #line 1370 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = RFC822DateParser_start;
	}

/* #line 181 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TRfc822DateTimeParser::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 1383 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
st1:
	if ( ++p == pe )
		goto _test_eof1;
case 1:
	switch( (*p) ) {
		case 32: goto st1;
		case 70: goto st63;
		case 77: goto st67;
		case 83: goto st69;
		case 84: goto st71;
		case 87: goto st74;
		case 102: goto st63;
		case 109: goto st67;
		case 115: goto st69;
		case 116: goto st71;
		case 119: goto st74;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr2;
	} else if ( (*p) >= 9 )
		goto st1;
	goto st0;
st0:
cs = 0;
	goto _out;
tr2:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 1431 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr8;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr9;
	} else if ( (*p) >= 9 )
		goto tr8;
	goto st0;
tr8:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 1448 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st3;
		case 65: goto st4;
		case 68: goto st37;
		case 70: goto st40;
		case 74: goto st43;
		case 77: goto st49;
		case 78: goto st53;
		case 79: goto st56;
		case 83: goto st59;
		case 97: goto st4;
		case 100: goto st37;
		case 102: goto st40;
		case 106: goto st43;
		case 109: goto st49;
		case 110: goto st53;
		case 111: goto st56;
		case 115: goto st59;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st3;
	goto st0;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	switch( (*p) ) {
		case 80: goto st5;
		case 85: goto st35;
		case 112: goto st5;
		case 117: goto st35;
	}
	goto st0;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	switch( (*p) ) {
		case 82: goto st6;
		case 114: goto st6;
	}
	goto st0;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	if ( (*p) == 32 )
		goto tr22;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr22;
	goto st0;
tr22:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st7;
tr63:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st7;
tr66:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st7;
tr69:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st7;
tr73:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st7;
tr76:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st7;
tr77:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st7;
tr81:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st7;
tr82:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st7;
tr85:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st7;
tr88:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st7;
tr91:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 1552 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st7;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr24;
	} else if ( (*p) >= 9 )
		goto st7;
	goto st0;
tr24:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 1577 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr25;
	goto st0;
tr25:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 1592 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr26;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr27;
	} else if ( (*p) >= 9 )
		goto tr26;
	goto st0;
tr26:
/* #line 82 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.SetLooseYear(I); }
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 1609 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st10;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr29;
	} else if ( (*p) >= 9 )
		goto st10;
	goto st0;
tr29:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 1634 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr30;
	goto st0;
tr30:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 1649 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr31;
	goto st0;
tr31:
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 1661 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr32;
	goto st0;
tr32:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 1681 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr33;
	goto st0;
tr33:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 1696 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto tr34;
		case 58: goto tr35;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr34;
	goto st0;
tr34:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st16;
tr60:
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 1716 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st16;
		case 43: goto tr37;
		case 45: goto tr37;
		case 67: goto tr39;
		case 69: goto tr40;
		case 71: goto tr41;
		case 77: goto tr42;
		case 80: goto tr43;
		case 85: goto tr44;
	}
	if ( (*p) < 75 ) {
		if ( (*p) > 13 ) {
			if ( 65 <= (*p) && (*p) <= 73 )
				goto tr38;
		} else if ( (*p) >= 9 )
			goto st16;
	} else if ( (*p) > 90 ) {
		if ( (*p) > 105 ) {
			if ( 107 <= (*p) && (*p) <= 122 )
				goto tr38;
		} else if ( (*p) >= 97 )
			goto tr38;
	} else
		goto tr38;
	goto st0;
tr37:
/* #line 155 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ Sign = (*p) == '+' ? 1 : -1; }
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 1751 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr45;
	goto st0;
tr45:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 1771 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr46;
	goto st0;
tr46:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 1786 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr47;
	goto st0;
tr47:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
/* #line 1801 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr48;
	goto st0;
tr38:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st76;
tr48:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 133 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    DateTimeFields.ZoneOffsetMinutes = Sign * (i32)(TDuration::Hours(I / 100) + TDuration::Minutes(I % 100)).Minutes();
}
	goto st76;
tr49:
/* #line 149 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(5).Minutes(); }
	goto st76;
tr50:
/* #line 148 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(6).Minutes();}
	goto st76;
tr51:
/* #line 147 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(4).Minutes(); }
	goto st76;
tr52:
/* #line 146 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(5).Minutes();}
	goto st76;
tr53:
/* #line 145 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st76;
tr54:
/* #line 151 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(6).Minutes(); }
	goto st76;
tr55:
/* #line 150 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(7).Minutes();}
	goto st76;
tr56:
/* #line 153 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(7).Minutes(); }
	goto st76;
tr57:
/* #line 152 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(8).Minutes();}
	goto st76;
tr110:
/* #line 144 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st76;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
/* #line 1880 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st76;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
tr39:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st77;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
/* #line 1910 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st21;
		case 83: goto st22;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
	if ( (*p) == 84 )
		goto tr49;
	goto st0;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	if ( (*p) == 84 )
		goto tr50;
	goto st0;
tr40:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st78;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
/* #line 1957 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st23;
		case 83: goto st24;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
	if ( (*p) == 84 )
		goto tr51;
	goto st0;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
	if ( (*p) == 84 )
		goto tr52;
	goto st0;
tr41:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st79;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
/* #line 2004 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 77: goto st25;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
	if ( (*p) == 84 )
		goto tr53;
	goto st0;
tr42:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st80;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
/* #line 2043 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st26;
		case 83: goto st27;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
	if ( (*p) == 84 )
		goto tr54;
	goto st0;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
	if ( (*p) == 84 )
		goto tr55;
	goto st0;
tr43:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st81;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
/* #line 2090 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 68: goto st28;
		case 83: goto st29;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	if ( (*p) == 84 )
		goto tr56;
	goto st0;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
	if ( (*p) == 84 )
		goto tr57;
	goto st0;
tr44:
/* #line 115 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    char c = (char)toupper((*p));
    if (c == 'Z')
        DateTimeFields.ZoneOffsetMinutes = 0;
    else {
        if (c <= 'M') {
            // ['A'..'M'] \ 'J'
            if (c < 'J')
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A' + 1).Minutes();
            else
                DateTimeFields.ZoneOffsetMinutes = (i32)TDuration::Hours(c - 'A').Minutes();
        } else {
            // ['N'..'Y']
            DateTimeFields.ZoneOffsetMinutes = -(i32)TDuration::Hours(c - 'N' + 1).Minutes();
        }
    }
}
	goto st82;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
/* #line 2137 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st76;
		case 84: goto tr110;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st76;
	goto st0;
tr35:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st30;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
/* #line 2153 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr58;
	goto st0;
tr58:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st31;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
/* #line 2173 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr59;
	goto st0;
tr59:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st32;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
/* #line 2188 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr60;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr60;
	goto st0;
tr27:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st33;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
/* #line 2205 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr61;
	goto st0;
tr61:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st34;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
/* #line 2220 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr26;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr26;
	goto st0;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
	switch( (*p) ) {
		case 71: goto st36;
		case 103: goto st36;
	}
	goto st0;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
	if ( (*p) == 32 )
		goto tr63;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr63;
	goto st0;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
	switch( (*p) ) {
		case 69: goto st38;
		case 101: goto st38;
	}
	goto st0;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
	switch( (*p) ) {
		case 67: goto st39;
		case 99: goto st39;
	}
	goto st0;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
	if ( (*p) == 32 )
		goto tr66;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr66;
	goto st0;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
	switch( (*p) ) {
		case 69: goto st41;
		case 101: goto st41;
	}
	goto st0;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	switch( (*p) ) {
		case 66: goto st42;
		case 98: goto st42;
	}
	goto st0;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 32 )
		goto tr69;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr69;
	goto st0;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
	switch( (*p) ) {
		case 65: goto st44;
		case 85: goto st46;
		case 97: goto st44;
		case 117: goto st46;
	}
	goto st0;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
	switch( (*p) ) {
		case 78: goto st45;
		case 110: goto st45;
	}
	goto st0;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
	if ( (*p) == 32 )
		goto tr73;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr73;
	goto st0;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
	switch( (*p) ) {
		case 76: goto st47;
		case 78: goto st48;
		case 108: goto st47;
		case 110: goto st48;
	}
	goto st0;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	if ( (*p) == 32 )
		goto tr76;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr76;
	goto st0;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
	if ( (*p) == 32 )
		goto tr77;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr77;
	goto st0;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
	switch( (*p) ) {
		case 65: goto st50;
		case 97: goto st50;
	}
	goto st0;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
	switch( (*p) ) {
		case 82: goto st51;
		case 89: goto st52;
		case 114: goto st51;
		case 121: goto st52;
	}
	goto st0;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
	if ( (*p) == 32 )
		goto tr81;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr81;
	goto st0;
st52:
	if ( ++p == pe )
		goto _test_eof52;
case 52:
	if ( (*p) == 32 )
		goto tr82;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr82;
	goto st0;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
	switch( (*p) ) {
		case 79: goto st54;
		case 111: goto st54;
	}
	goto st0;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
	switch( (*p) ) {
		case 86: goto st55;
		case 118: goto st55;
	}
	goto st0;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
	if ( (*p) == 32 )
		goto tr85;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr85;
	goto st0;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
	switch( (*p) ) {
		case 67: goto st57;
		case 99: goto st57;
	}
	goto st0;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
	switch( (*p) ) {
		case 84: goto st58;
		case 116: goto st58;
	}
	goto st0;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
	if ( (*p) == 32 )
		goto tr88;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr88;
	goto st0;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
	switch( (*p) ) {
		case 69: goto st60;
		case 101: goto st60;
	}
	goto st0;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
	switch( (*p) ) {
		case 80: goto st61;
		case 112: goto st61;
	}
	goto st0;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
	if ( (*p) == 32 )
		goto tr91;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr91;
	goto st0;
tr9:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st62;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
/* #line 2486 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr8;
	if ( 9 <= (*p) && (*p) <= 13 )
		goto tr8;
	goto st0;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
	switch( (*p) ) {
		case 82: goto st64;
		case 114: goto st64;
	}
	goto st0;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
	switch( (*p) ) {
		case 73: goto st65;
		case 105: goto st65;
	}
	goto st0;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
	if ( (*p) == 44 )
		goto st66;
	goto st0;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
	if ( (*p) == 32 )
		goto st66;
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr2;
	} else if ( (*p) >= 9 )
		goto st66;
	goto st0;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
	switch( (*p) ) {
		case 79: goto st68;
		case 111: goto st68;
	}
	goto st0;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
	switch( (*p) ) {
		case 78: goto st65;
		case 110: goto st65;
	}
	goto st0;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
	switch( (*p) ) {
		case 65: goto st70;
		case 85: goto st68;
		case 97: goto st70;
		case 117: goto st68;
	}
	goto st0;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
	switch( (*p) ) {
		case 84: goto st65;
		case 116: goto st65;
	}
	goto st0;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
	switch( (*p) ) {
		case 72: goto st72;
		case 85: goto st73;
		case 104: goto st72;
		case 117: goto st73;
	}
	goto st0;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
	switch( (*p) ) {
		case 85: goto st65;
		case 117: goto st65;
	}
	goto st0;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
	switch( (*p) ) {
		case 69: goto st65;
		case 101: goto st65;
	}
	goto st0;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
	switch( (*p) ) {
		case 69: goto st75;
		case 101: goto st75;
	}
	goto st0;
st75:
	if ( ++p == pe )
		goto _test_eof75;
case 75:
	switch( (*p) ) {
		case 68: goto st65;
		case 100: goto st65;
	}
	goto st0;
	}
	_test_eof1: cs = 1; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 
	_test_eof75: cs = 75; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 188 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}


/* #line 2707 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
static const int ISO8601DateTimeParser_start = 1;
static const int ISO8601DateTimeParser_first_final = 26;

static const int ISO8601DateTimeParser_en_main = 1;


/* #line 225 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */


TIso8601DateTimeParserDeprecated::TIso8601DateTimeParserDeprecated() {
    
/* #line 2719 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = ISO8601DateTimeParser_start;
	}

/* #line 229 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TIso8601DateTimeParserDeprecated::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 2732 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 2761 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 2776 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr3;
	goto st0;
tr3:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
/* #line 2791 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr4;
	goto st0;
tr4:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 83 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Year = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 2808 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 45 )
		goto st6;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr7;
	goto st0;
tr7:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 2837 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr8;
	goto st0;
tr8:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 2854 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 45 )
		goto st9;
	goto st0;
tr27:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 2871 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr10;
	goto st0;
tr10:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 2891 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr11;
	goto st0;
tr11:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st26;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
/* #line 2908 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st11;
		case 84: goto st11;
		case 116: goto st11;
	}
	goto st0;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr12;
	goto st0;
tr12:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 2938 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr13;
	goto st0;
tr13:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 2955 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto st20;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr14;
	goto st0;
tr14:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 2977 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr16;
	goto st0;
tr16:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st27;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
/* #line 2994 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr30;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
	switch( (*p) ) {
		case 43: goto tr17;
		case 45: goto tr17;
	}
	goto st0;
tr17:
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ Sign = (*p) == '+' ? 1 : -1; }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 3025 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr18;
	goto st0;
tr18:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 3045 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr19;
	goto st0;
tr19:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = Sign * (i32)TDuration::Hours(I).Minutes(); }
	goto st28;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
/* #line 3062 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto st30;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr32;
	goto st0;
tr32:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 3084 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr20;
	goto st0;
tr20:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes += I * Sign; }
	goto st29;
tr31:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st29;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
/* #line 3105 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr32;
	goto st0;
tr30:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 3130 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr21;
	goto st0;
tr21:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st31;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
/* #line 3147 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st15;
	goto st0;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr22;
	goto st0;
tr22:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st21;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
/* #line 3181 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr23;
	goto st0;
tr23:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st32;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
/* #line 3198 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 58: goto st22;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st15;
	goto st0;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr24;
	goto st0;
tr24:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st23;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
/* #line 3233 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr25;
	goto st0;
tr25:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st33;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
/* #line 3250 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 46: goto st24;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st15;
	goto st0;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr26;
	goto st0;
tr26:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st34;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
/* #line 3293 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr36;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr36:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st35;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
/* #line 3326 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr37;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr37:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st36;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
/* #line 3359 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr38;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr38:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st37;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
/* #line 3392 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr39;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr39:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st38;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
/* #line 3425 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr40;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr40:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st39;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
/* #line 3458 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st39;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr6:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st25;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
/* #line 3488 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr27;
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 236 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

TIso8601DateTimeParser::TIso8601DateTimeParser() {
    
/* #line 3542 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = ISO8601DateTimeParser_start;
	}

/* #line 241 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TIso8601DateTimeParser::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 3555 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 3584 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 3599 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr3;
	goto st0;
tr3:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
/* #line 3614 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr4;
	goto st0;
tr4:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 83 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Year = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 3631 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 45 )
		goto st6;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr7;
	goto st0;
tr7:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 3660 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr8;
	goto st0;
tr8:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 3677 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 45 )
		goto st9;
	goto st0;
tr27:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 3694 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr10;
	goto st0;
tr10:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 3714 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr11;
	goto st0;
tr11:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st26;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
/* #line 3731 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st11;
		case 84: goto st11;
		case 116: goto st11;
	}
	goto st0;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr12;
	goto st0;
tr12:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 3761 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr13;
	goto st0;
tr13:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 3778 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto st20;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr14;
	goto st0;
tr14:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 3800 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr16;
	goto st0;
tr16:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st27;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
/* #line 3817 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr30;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
	switch( (*p) ) {
		case 43: goto tr17;
		case 45: goto tr17;
	}
	goto st0;
tr17:
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ Sign = (*p) == '+' ? 1 : -1; }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 3848 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr18;
	goto st0;
tr18:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 3868 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr19;
	goto st0;
tr19:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = Sign * (i32)TDuration::Hours(I).Minutes(); }
	goto st28;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
/* #line 3885 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto st30;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr32;
	goto st0;
tr32:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 3907 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr20;
	goto st0;
tr20:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 213 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes += I * Sign; }
	goto st29;
tr31:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st29;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
/* #line 3928 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr32;
	goto st0;
tr30:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 3953 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr21;
	goto st0;
tr21:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st31;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
/* #line 3970 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st15;
	goto st0;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr22;
	goto st0;
tr22:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st21;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
/* #line 4004 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr23;
	goto st0;
tr23:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st32;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
/* #line 4021 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 58: goto st22;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st15;
	goto st0;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr24;
	goto st0;
tr24:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st23;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
/* #line 4056 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr25;
	goto st0;
tr25:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st33;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
/* #line 4073 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 46: goto st24;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( 9 <= (*p) && (*p) <= 13 )
		goto st15;
	goto st0;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr26;
	goto st0;
tr26:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st34;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
/* #line 4116 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr36;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr36:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st35;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
/* #line 4149 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr37;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr37:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st36;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
/* #line 4182 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr38;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr38:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st37;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
/* #line 4215 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr39;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr39:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st38;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
/* #line 4248 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto tr40;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr40:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 203 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    ui32 us = I;
    for (int k = Dc; k < 6; ++k) {
        us *= 10;
    }
    DateTimeFields.MicroSecond = us;
}
	goto st39;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
/* #line 4281 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 32: goto st15;
		case 43: goto tr17;
		case 45: goto tr17;
		case 90: goto tr31;
		case 122: goto tr31;
	}
	if ( (*p) > 13 ) {
		if ( 48 <= (*p) && (*p) <= 57 )
			goto st39;
	} else if ( (*p) >= 9 )
		goto st15;
	goto st0;
tr6:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st25;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
/* #line 4311 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr27;
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 248 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}


/* #line 269 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */



/* #line 4368 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
static const int HttpDateTimeParserStandalone_start = 1;
static const int HttpDateTimeParserStandalone_first_final = 161;

static const int HttpDateTimeParserStandalone_en_main = 1;


/* #line 281 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */


THttpDateTimeParserDeprecated::THttpDateTimeParserDeprecated() {
    
/* #line 4380 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = HttpDateTimeParserStandalone_start;
	}

/* #line 285 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool THttpDateTimeParserDeprecated::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 4393 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	switch( (*p) ) {
		case 70: goto st2;
		case 77: goto st144;
		case 83: goto st146;
		case 84: goto st151;
		case 87: goto st157;
		case 102: goto st2;
		case 109: goto st144;
		case 115: goto st146;
		case 116: goto st151;
		case 119: goto st157;
	}
	goto st0;
st0:
cs = 0;
	goto _out;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
	switch( (*p) ) {
		case 82: goto st3;
		case 114: goto st3;
	}
	goto st0;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	switch( (*p) ) {
		case 73: goto st4;
		case 105: goto st4;
	}
	goto st0;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 68: goto st105;
		case 100: goto st105;
	}
	goto st0;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	switch( (*p) ) {
		case 65: goto st6;
		case 68: goto st28;
		case 70: goto st31;
		case 74: goto st34;
		case 77: goto st40;
		case 78: goto st44;
		case 79: goto st47;
		case 83: goto st50;
		case 97: goto st6;
		case 100: goto st28;
		case 102: goto st31;
		case 106: goto st34;
		case 109: goto st40;
		case 110: goto st44;
		case 111: goto st47;
		case 115: goto st50;
	}
	goto st0;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	switch( (*p) ) {
		case 80: goto st7;
		case 85: goto st26;
		case 112: goto st7;
		case 117: goto st26;
	}
	goto st0;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
	switch( (*p) ) {
		case 82: goto st8;
		case 114: goto st8;
	}
	goto st0;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
	if ( (*p) == 32 )
		goto tr22;
	goto st0;
tr22:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st9;
tr42:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st9;
tr45:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st9;
tr48:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st9;
tr52:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st9;
tr55:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st9;
tr56:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st9;
tr60:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st9;
tr61:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st9;
tr64:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st9;
tr67:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st9;
tr70:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 4547 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st10;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr24;
	goto st0;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr25;
	goto st0;
tr25:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st11;
tr40:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 4583 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr26;
	goto st0;
tr26:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 4595 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr27;
	goto st0;
tr27:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 4615 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr28;
	goto st0;
tr28:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 4630 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr29;
	goto st0;
tr29:
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 4642 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr30;
	goto st0;
tr30:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 4662 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr31;
	goto st0;
tr31:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 4677 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr32;
	goto st0;
tr32:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 4689 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr33;
	goto st0;
tr33:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 4709 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr34;
	goto st0;
tr34:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
/* #line 4724 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr35;
	goto st0;
tr35:
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st21;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
/* #line 4736 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr36;
	goto st0;
tr36:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st22;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
/* #line 4756 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr37;
	goto st0;
tr37:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st23;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
/* #line 4771 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr38;
	goto st0;
tr38:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st24;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
/* #line 4786 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr39;
	goto st0;
tr39:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 82 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.SetLooseYear(I); }
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st161;
tr103:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st161;
st161:
	if ( ++p == pe )
		goto _test_eof161;
case 161:
/* #line 4809 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
tr24:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st25;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
/* #line 4827 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr40;
	goto st0;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
	switch( (*p) ) {
		case 71: goto st27;
		case 103: goto st27;
	}
	goto st0;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
	if ( (*p) == 32 )
		goto tr42;
	goto st0;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	switch( (*p) ) {
		case 69: goto st29;
		case 101: goto st29;
	}
	goto st0;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
	switch( (*p) ) {
		case 67: goto st30;
		case 99: goto st30;
	}
	goto st0;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	if ( (*p) == 32 )
		goto tr45;
	goto st0;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
	switch( (*p) ) {
		case 69: goto st32;
		case 101: goto st32;
	}
	goto st0;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
	switch( (*p) ) {
		case 66: goto st33;
		case 98: goto st33;
	}
	goto st0;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
	if ( (*p) == 32 )
		goto tr48;
	goto st0;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
	switch( (*p) ) {
		case 65: goto st35;
		case 85: goto st37;
		case 97: goto st35;
		case 117: goto st37;
	}
	goto st0;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
	switch( (*p) ) {
		case 78: goto st36;
		case 110: goto st36;
	}
	goto st0;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
	if ( (*p) == 32 )
		goto tr52;
	goto st0;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
	switch( (*p) ) {
		case 76: goto st38;
		case 78: goto st39;
		case 108: goto st38;
		case 110: goto st39;
	}
	goto st0;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
	if ( (*p) == 32 )
		goto tr55;
	goto st0;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
	if ( (*p) == 32 )
		goto tr56;
	goto st0;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
	switch( (*p) ) {
		case 65: goto st41;
		case 97: goto st41;
	}
	goto st0;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	switch( (*p) ) {
		case 82: goto st42;
		case 89: goto st43;
		case 114: goto st42;
		case 121: goto st43;
	}
	goto st0;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 32 )
		goto tr60;
	goto st0;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
	if ( (*p) == 32 )
		goto tr61;
	goto st0;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
	switch( (*p) ) {
		case 79: goto st45;
		case 111: goto st45;
	}
	goto st0;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
	switch( (*p) ) {
		case 86: goto st46;
		case 118: goto st46;
	}
	goto st0;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
	if ( (*p) == 32 )
		goto tr64;
	goto st0;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	switch( (*p) ) {
		case 67: goto st48;
		case 99: goto st48;
	}
	goto st0;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
	switch( (*p) ) {
		case 84: goto st49;
		case 116: goto st49;
	}
	goto st0;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
	if ( (*p) == 32 )
		goto tr67;
	goto st0;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
	switch( (*p) ) {
		case 69: goto st51;
		case 101: goto st51;
	}
	goto st0;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
	switch( (*p) ) {
		case 80: goto st52;
		case 112: goto st52;
	}
	goto st0;
st52:
	if ( ++p == pe )
		goto _test_eof52;
case 52:
	if ( (*p) == 32 )
		goto tr70;
	goto st0;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
	if ( (*p) == 32 )
		goto st54;
	goto st0;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr72;
	goto st0;
tr72:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st55;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
/* #line 5088 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr73;
	goto st0;
tr73:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st56;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
/* #line 5103 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr74;
	goto st0;
tr74:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st57;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
/* #line 5115 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 65: goto st58;
		case 68: goto st80;
		case 70: goto st83;
		case 74: goto st86;
		case 77: goto st92;
		case 78: goto st96;
		case 79: goto st99;
		case 83: goto st102;
		case 97: goto st58;
		case 100: goto st80;
		case 102: goto st83;
		case 106: goto st86;
		case 109: goto st92;
		case 110: goto st96;
		case 111: goto st99;
		case 115: goto st102;
	}
	goto st0;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
	switch( (*p) ) {
		case 80: goto st59;
		case 85: goto st78;
		case 112: goto st59;
		case 117: goto st78;
	}
	goto st0;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
	switch( (*p) ) {
		case 82: goto st60;
		case 114: goto st60;
	}
	goto st0;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
	if ( (*p) == 32 )
		goto tr86;
	goto st0;
tr86:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st61;
tr105:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st61;
tr108:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st61;
tr111:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st61;
tr115:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st61;
tr118:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st61;
tr119:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st61;
tr123:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st61;
tr124:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st61;
tr127:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st61;
tr130:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st61;
tr133:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st61;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
/* #line 5214 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr87;
	goto st0;
tr87:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st62;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
/* #line 5234 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr88;
	goto st0;
tr88:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st63;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
/* #line 5249 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr89;
	goto st0;
tr153:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st64;
tr89:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st64;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
/* #line 5276 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr90;
	goto st0;
tr90:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st65;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
/* #line 5291 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr91;
	goto st0;
tr91:
/* #line 82 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.SetLooseYear(I); }
	goto st66;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
/* #line 5303 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr92;
	goto st0;
tr92:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st67;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
/* #line 5323 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr93;
	goto st0;
tr93:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st68;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
/* #line 5338 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr94;
	goto st0;
tr94:
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st69;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
/* #line 5350 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr95;
	goto st0;
tr95:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st70;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
/* #line 5370 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr96;
	goto st0;
tr96:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st71;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
/* #line 5385 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr97;
	goto st0;
tr97:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st72;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
/* #line 5397 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr98;
	goto st0;
tr98:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st73;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
/* #line 5417 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr99;
	goto st0;
tr99:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st74;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
/* #line 5432 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr100;
	goto st0;
tr100:
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st75;
st75:
	if ( ++p == pe )
		goto _test_eof75;
case 75:
/* #line 5444 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 71: goto st76;
		case 103: goto st76;
	}
	goto st0;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
	switch( (*p) ) {
		case 77: goto st77;
		case 109: goto st77;
	}
	goto st0;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
	switch( (*p) ) {
		case 84: goto tr103;
		case 116: goto tr103;
	}
	goto st0;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
	switch( (*p) ) {
		case 71: goto st79;
		case 103: goto st79;
	}
	goto st0;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
	if ( (*p) == 32 )
		goto tr105;
	goto st0;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
	switch( (*p) ) {
		case 69: goto st81;
		case 101: goto st81;
	}
	goto st0;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
	switch( (*p) ) {
		case 67: goto st82;
		case 99: goto st82;
	}
	goto st0;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
	if ( (*p) == 32 )
		goto tr108;
	goto st0;
st83:
	if ( ++p == pe )
		goto _test_eof83;
case 83:
	switch( (*p) ) {
		case 69: goto st84;
		case 101: goto st84;
	}
	goto st0;
st84:
	if ( ++p == pe )
		goto _test_eof84;
case 84:
	switch( (*p) ) {
		case 66: goto st85;
		case 98: goto st85;
	}
	goto st0;
st85:
	if ( ++p == pe )
		goto _test_eof85;
case 85:
	if ( (*p) == 32 )
		goto tr111;
	goto st0;
st86:
	if ( ++p == pe )
		goto _test_eof86;
case 86:
	switch( (*p) ) {
		case 65: goto st87;
		case 85: goto st89;
		case 97: goto st87;
		case 117: goto st89;
	}
	goto st0;
st87:
	if ( ++p == pe )
		goto _test_eof87;
case 87:
	switch( (*p) ) {
		case 78: goto st88;
		case 110: goto st88;
	}
	goto st0;
st88:
	if ( ++p == pe )
		goto _test_eof88;
case 88:
	if ( (*p) == 32 )
		goto tr115;
	goto st0;
st89:
	if ( ++p == pe )
		goto _test_eof89;
case 89:
	switch( (*p) ) {
		case 76: goto st90;
		case 78: goto st91;
		case 108: goto st90;
		case 110: goto st91;
	}
	goto st0;
st90:
	if ( ++p == pe )
		goto _test_eof90;
case 90:
	if ( (*p) == 32 )
		goto tr118;
	goto st0;
st91:
	if ( ++p == pe )
		goto _test_eof91;
case 91:
	if ( (*p) == 32 )
		goto tr119;
	goto st0;
st92:
	if ( ++p == pe )
		goto _test_eof92;
case 92:
	switch( (*p) ) {
		case 65: goto st93;
		case 97: goto st93;
	}
	goto st0;
st93:
	if ( ++p == pe )
		goto _test_eof93;
case 93:
	switch( (*p) ) {
		case 82: goto st94;
		case 89: goto st95;
		case 114: goto st94;
		case 121: goto st95;
	}
	goto st0;
st94:
	if ( ++p == pe )
		goto _test_eof94;
case 94:
	if ( (*p) == 32 )
		goto tr123;
	goto st0;
st95:
	if ( ++p == pe )
		goto _test_eof95;
case 95:
	if ( (*p) == 32 )
		goto tr124;
	goto st0;
st96:
	if ( ++p == pe )
		goto _test_eof96;
case 96:
	switch( (*p) ) {
		case 79: goto st97;
		case 111: goto st97;
	}
	goto st0;
st97:
	if ( ++p == pe )
		goto _test_eof97;
case 97:
	switch( (*p) ) {
		case 86: goto st98;
		case 118: goto st98;
	}
	goto st0;
st98:
	if ( ++p == pe )
		goto _test_eof98;
case 98:
	if ( (*p) == 32 )
		goto tr127;
	goto st0;
st99:
	if ( ++p == pe )
		goto _test_eof99;
case 99:
	switch( (*p) ) {
		case 67: goto st100;
		case 99: goto st100;
	}
	goto st0;
st100:
	if ( ++p == pe )
		goto _test_eof100;
case 100:
	switch( (*p) ) {
		case 84: goto st101;
		case 116: goto st101;
	}
	goto st0;
st101:
	if ( ++p == pe )
		goto _test_eof101;
case 101:
	if ( (*p) == 32 )
		goto tr130;
	goto st0;
st102:
	if ( ++p == pe )
		goto _test_eof102;
case 102:
	switch( (*p) ) {
		case 69: goto st103;
		case 101: goto st103;
	}
	goto st0;
st103:
	if ( ++p == pe )
		goto _test_eof103;
case 103:
	switch( (*p) ) {
		case 80: goto st104;
		case 112: goto st104;
	}
	goto st0;
st104:
	if ( ++p == pe )
		goto _test_eof104;
case 104:
	if ( (*p) == 32 )
		goto tr133;
	goto st0;
st105:
	if ( ++p == pe )
		goto _test_eof105;
case 105:
	switch( (*p) ) {
		case 65: goto st106;
		case 97: goto st106;
	}
	goto st0;
st106:
	if ( ++p == pe )
		goto _test_eof106;
case 106:
	switch( (*p) ) {
		case 89: goto st107;
		case 121: goto st107;
	}
	goto st0;
st107:
	if ( ++p == pe )
		goto _test_eof107;
case 107:
	if ( (*p) == 44 )
		goto st108;
	goto st0;
st108:
	if ( ++p == pe )
		goto _test_eof108;
case 108:
	if ( (*p) == 32 )
		goto st109;
	goto st0;
st109:
	if ( ++p == pe )
		goto _test_eof109;
case 109:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr138;
	goto st0;
tr138:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st110;
st110:
	if ( ++p == pe )
		goto _test_eof110;
case 110:
/* #line 5750 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr139;
	goto st0;
tr139:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st111;
st111:
	if ( ++p == pe )
		goto _test_eof111;
case 111:
/* #line 5765 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 45 )
		goto tr140;
	goto st0;
tr140:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st112;
st112:
	if ( ++p == pe )
		goto _test_eof112;
case 112:
/* #line 5777 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 65: goto st113;
		case 68: goto st119;
		case 70: goto st122;
		case 74: goto st125;
		case 77: goto st131;
		case 78: goto st135;
		case 79: goto st138;
		case 83: goto st141;
		case 97: goto st113;
		case 100: goto st119;
		case 102: goto st122;
		case 106: goto st125;
		case 109: goto st131;
		case 110: goto st135;
		case 111: goto st138;
		case 115: goto st141;
	}
	goto st0;
st113:
	if ( ++p == pe )
		goto _test_eof113;
case 113:
	switch( (*p) ) {
		case 80: goto st114;
		case 85: goto st117;
		case 112: goto st114;
		case 117: goto st117;
	}
	goto st0;
st114:
	if ( ++p == pe )
		goto _test_eof114;
case 114:
	switch( (*p) ) {
		case 82: goto st115;
		case 114: goto st115;
	}
	goto st0;
st115:
	if ( ++p == pe )
		goto _test_eof115;
case 115:
	if ( (*p) == 45 )
		goto tr152;
	goto st0;
tr152:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st116;
tr155:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st116;
tr158:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st116;
tr161:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st116;
tr165:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st116;
tr168:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st116;
tr169:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st116;
tr173:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st116;
tr174:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st116;
tr177:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st116;
tr180:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st116;
tr183:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st116;
st116:
	if ( ++p == pe )
		goto _test_eof116;
case 116:
/* #line 5876 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr153;
	goto st0;
st117:
	if ( ++p == pe )
		goto _test_eof117;
case 117:
	switch( (*p) ) {
		case 71: goto st118;
		case 103: goto st118;
	}
	goto st0;
st118:
	if ( ++p == pe )
		goto _test_eof118;
case 118:
	if ( (*p) == 45 )
		goto tr155;
	goto st0;
st119:
	if ( ++p == pe )
		goto _test_eof119;
case 119:
	switch( (*p) ) {
		case 69: goto st120;
		case 101: goto st120;
	}
	goto st0;
st120:
	if ( ++p == pe )
		goto _test_eof120;
case 120:
	switch( (*p) ) {
		case 67: goto st121;
		case 99: goto st121;
	}
	goto st0;
st121:
	if ( ++p == pe )
		goto _test_eof121;
case 121:
	if ( (*p) == 45 )
		goto tr158;
	goto st0;
st122:
	if ( ++p == pe )
		goto _test_eof122;
case 122:
	switch( (*p) ) {
		case 69: goto st123;
		case 101: goto st123;
	}
	goto st0;
st123:
	if ( ++p == pe )
		goto _test_eof123;
case 123:
	switch( (*p) ) {
		case 66: goto st124;
		case 98: goto st124;
	}
	goto st0;
st124:
	if ( ++p == pe )
		goto _test_eof124;
case 124:
	if ( (*p) == 45 )
		goto tr161;
	goto st0;
st125:
	if ( ++p == pe )
		goto _test_eof125;
case 125:
	switch( (*p) ) {
		case 65: goto st126;
		case 85: goto st128;
		case 97: goto st126;
		case 117: goto st128;
	}
	goto st0;
st126:
	if ( ++p == pe )
		goto _test_eof126;
case 126:
	switch( (*p) ) {
		case 78: goto st127;
		case 110: goto st127;
	}
	goto st0;
st127:
	if ( ++p == pe )
		goto _test_eof127;
case 127:
	if ( (*p) == 45 )
		goto tr165;
	goto st0;
st128:
	if ( ++p == pe )
		goto _test_eof128;
case 128:
	switch( (*p) ) {
		case 76: goto st129;
		case 78: goto st130;
		case 108: goto st129;
		case 110: goto st130;
	}
	goto st0;
st129:
	if ( ++p == pe )
		goto _test_eof129;
case 129:
	if ( (*p) == 45 )
		goto tr168;
	goto st0;
st130:
	if ( ++p == pe )
		goto _test_eof130;
case 130:
	if ( (*p) == 45 )
		goto tr169;
	goto st0;
st131:
	if ( ++p == pe )
		goto _test_eof131;
case 131:
	switch( (*p) ) {
		case 65: goto st132;
		case 97: goto st132;
	}
	goto st0;
st132:
	if ( ++p == pe )
		goto _test_eof132;
case 132:
	switch( (*p) ) {
		case 82: goto st133;
		case 89: goto st134;
		case 114: goto st133;
		case 121: goto st134;
	}
	goto st0;
st133:
	if ( ++p == pe )
		goto _test_eof133;
case 133:
	if ( (*p) == 45 )
		goto tr173;
	goto st0;
st134:
	if ( ++p == pe )
		goto _test_eof134;
case 134:
	if ( (*p) == 45 )
		goto tr174;
	goto st0;
st135:
	if ( ++p == pe )
		goto _test_eof135;
case 135:
	switch( (*p) ) {
		case 79: goto st136;
		case 111: goto st136;
	}
	goto st0;
st136:
	if ( ++p == pe )
		goto _test_eof136;
case 136:
	switch( (*p) ) {
		case 86: goto st137;
		case 118: goto st137;
	}
	goto st0;
st137:
	if ( ++p == pe )
		goto _test_eof137;
case 137:
	if ( (*p) == 45 )
		goto tr177;
	goto st0;
st138:
	if ( ++p == pe )
		goto _test_eof138;
case 138:
	switch( (*p) ) {
		case 67: goto st139;
		case 99: goto st139;
	}
	goto st0;
st139:
	if ( ++p == pe )
		goto _test_eof139;
case 139:
	switch( (*p) ) {
		case 84: goto st140;
		case 116: goto st140;
	}
	goto st0;
st140:
	if ( ++p == pe )
		goto _test_eof140;
case 140:
	if ( (*p) == 45 )
		goto tr180;
	goto st0;
st141:
	if ( ++p == pe )
		goto _test_eof141;
case 141:
	switch( (*p) ) {
		case 69: goto st142;
		case 101: goto st142;
	}
	goto st0;
st142:
	if ( ++p == pe )
		goto _test_eof142;
case 142:
	switch( (*p) ) {
		case 80: goto st143;
		case 112: goto st143;
	}
	goto st0;
st143:
	if ( ++p == pe )
		goto _test_eof143;
case 143:
	if ( (*p) == 45 )
		goto tr183;
	goto st0;
st144:
	if ( ++p == pe )
		goto _test_eof144;
case 144:
	switch( (*p) ) {
		case 79: goto st145;
		case 111: goto st145;
	}
	goto st0;
st145:
	if ( ++p == pe )
		goto _test_eof145;
case 145:
	switch( (*p) ) {
		case 78: goto st4;
		case 110: goto st4;
	}
	goto st0;
st146:
	if ( ++p == pe )
		goto _test_eof146;
case 146:
	switch( (*p) ) {
		case 65: goto st147;
		case 85: goto st145;
		case 97: goto st147;
		case 117: goto st145;
	}
	goto st0;
st147:
	if ( ++p == pe )
		goto _test_eof147;
case 147:
	switch( (*p) ) {
		case 84: goto st148;
		case 116: goto st148;
	}
	goto st0;
st148:
	if ( ++p == pe )
		goto _test_eof148;
case 148:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 85: goto st149;
		case 117: goto st149;
	}
	goto st0;
st149:
	if ( ++p == pe )
		goto _test_eof149;
case 149:
	switch( (*p) ) {
		case 82: goto st150;
		case 114: goto st150;
	}
	goto st0;
st150:
	if ( ++p == pe )
		goto _test_eof150;
case 150:
	switch( (*p) ) {
		case 68: goto st105;
		case 100: goto st105;
	}
	goto st0;
st151:
	if ( ++p == pe )
		goto _test_eof151;
case 151:
	switch( (*p) ) {
		case 72: goto st152;
		case 85: goto st155;
		case 104: goto st152;
		case 117: goto st155;
	}
	goto st0;
st152:
	if ( ++p == pe )
		goto _test_eof152;
case 152:
	switch( (*p) ) {
		case 85: goto st153;
		case 117: goto st153;
	}
	goto st0;
st153:
	if ( ++p == pe )
		goto _test_eof153;
case 153:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 82: goto st154;
		case 114: goto st154;
	}
	goto st0;
st154:
	if ( ++p == pe )
		goto _test_eof154;
case 154:
	switch( (*p) ) {
		case 83: goto st150;
		case 115: goto st150;
	}
	goto st0;
st155:
	if ( ++p == pe )
		goto _test_eof155;
case 155:
	switch( (*p) ) {
		case 69: goto st156;
		case 101: goto st156;
	}
	goto st0;
st156:
	if ( ++p == pe )
		goto _test_eof156;
case 156:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 83: goto st150;
		case 115: goto st150;
	}
	goto st0;
st157:
	if ( ++p == pe )
		goto _test_eof157;
case 157:
	switch( (*p) ) {
		case 69: goto st158;
		case 101: goto st158;
	}
	goto st0;
st158:
	if ( ++p == pe )
		goto _test_eof158;
case 158:
	switch( (*p) ) {
		case 68: goto st159;
		case 100: goto st159;
	}
	goto st0;
st159:
	if ( ++p == pe )
		goto _test_eof159;
case 159:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 78: goto st160;
		case 110: goto st160;
	}
	goto st0;
st160:
	if ( ++p == pe )
		goto _test_eof160;
case 160:
	switch( (*p) ) {
		case 69: goto st154;
		case 101: goto st154;
	}
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof161: cs = 161; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 
	_test_eof75: cs = 75; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof83: cs = 83; goto _test_eof; 
	_test_eof84: cs = 84; goto _test_eof; 
	_test_eof85: cs = 85; goto _test_eof; 
	_test_eof86: cs = 86; goto _test_eof; 
	_test_eof87: cs = 87; goto _test_eof; 
	_test_eof88: cs = 88; goto _test_eof; 
	_test_eof89: cs = 89; goto _test_eof; 
	_test_eof90: cs = 90; goto _test_eof; 
	_test_eof91: cs = 91; goto _test_eof; 
	_test_eof92: cs = 92; goto _test_eof; 
	_test_eof93: cs = 93; goto _test_eof; 
	_test_eof94: cs = 94; goto _test_eof; 
	_test_eof95: cs = 95; goto _test_eof; 
	_test_eof96: cs = 96; goto _test_eof; 
	_test_eof97: cs = 97; goto _test_eof; 
	_test_eof98: cs = 98; goto _test_eof; 
	_test_eof99: cs = 99; goto _test_eof; 
	_test_eof100: cs = 100; goto _test_eof; 
	_test_eof101: cs = 101; goto _test_eof; 
	_test_eof102: cs = 102; goto _test_eof; 
	_test_eof103: cs = 103; goto _test_eof; 
	_test_eof104: cs = 104; goto _test_eof; 
	_test_eof105: cs = 105; goto _test_eof; 
	_test_eof106: cs = 106; goto _test_eof; 
	_test_eof107: cs = 107; goto _test_eof; 
	_test_eof108: cs = 108; goto _test_eof; 
	_test_eof109: cs = 109; goto _test_eof; 
	_test_eof110: cs = 110; goto _test_eof; 
	_test_eof111: cs = 111; goto _test_eof; 
	_test_eof112: cs = 112; goto _test_eof; 
	_test_eof113: cs = 113; goto _test_eof; 
	_test_eof114: cs = 114; goto _test_eof; 
	_test_eof115: cs = 115; goto _test_eof; 
	_test_eof116: cs = 116; goto _test_eof; 
	_test_eof117: cs = 117; goto _test_eof; 
	_test_eof118: cs = 118; goto _test_eof; 
	_test_eof119: cs = 119; goto _test_eof; 
	_test_eof120: cs = 120; goto _test_eof; 
	_test_eof121: cs = 121; goto _test_eof; 
	_test_eof122: cs = 122; goto _test_eof; 
	_test_eof123: cs = 123; goto _test_eof; 
	_test_eof124: cs = 124; goto _test_eof; 
	_test_eof125: cs = 125; goto _test_eof; 
	_test_eof126: cs = 126; goto _test_eof; 
	_test_eof127: cs = 127; goto _test_eof; 
	_test_eof128: cs = 128; goto _test_eof; 
	_test_eof129: cs = 129; goto _test_eof; 
	_test_eof130: cs = 130; goto _test_eof; 
	_test_eof131: cs = 131; goto _test_eof; 
	_test_eof132: cs = 132; goto _test_eof; 
	_test_eof133: cs = 133; goto _test_eof; 
	_test_eof134: cs = 134; goto _test_eof; 
	_test_eof135: cs = 135; goto _test_eof; 
	_test_eof136: cs = 136; goto _test_eof; 
	_test_eof137: cs = 137; goto _test_eof; 
	_test_eof138: cs = 138; goto _test_eof; 
	_test_eof139: cs = 139; goto _test_eof; 
	_test_eof140: cs = 140; goto _test_eof; 
	_test_eof141: cs = 141; goto _test_eof; 
	_test_eof142: cs = 142; goto _test_eof; 
	_test_eof143: cs = 143; goto _test_eof; 
	_test_eof144: cs = 144; goto _test_eof; 
	_test_eof145: cs = 145; goto _test_eof; 
	_test_eof146: cs = 146; goto _test_eof; 
	_test_eof147: cs = 147; goto _test_eof; 
	_test_eof148: cs = 148; goto _test_eof; 
	_test_eof149: cs = 149; goto _test_eof; 
	_test_eof150: cs = 150; goto _test_eof; 
	_test_eof151: cs = 151; goto _test_eof; 
	_test_eof152: cs = 152; goto _test_eof; 
	_test_eof153: cs = 153; goto _test_eof; 
	_test_eof154: cs = 154; goto _test_eof; 
	_test_eof155: cs = 155; goto _test_eof; 
	_test_eof156: cs = 156; goto _test_eof; 
	_test_eof157: cs = 157; goto _test_eof; 
	_test_eof158: cs = 158; goto _test_eof; 
	_test_eof159: cs = 159; goto _test_eof; 
	_test_eof160: cs = 160; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 292 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

THttpDateTimeParser::THttpDateTimeParser() {
    
/* #line 6444 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = HttpDateTimeParserStandalone_start;
	}

/* #line 297 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool THttpDateTimeParser::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 6457 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	switch( (*p) ) {
		case 70: goto st2;
		case 77: goto st144;
		case 83: goto st146;
		case 84: goto st151;
		case 87: goto st157;
		case 102: goto st2;
		case 109: goto st144;
		case 115: goto st146;
		case 116: goto st151;
		case 119: goto st157;
	}
	goto st0;
st0:
cs = 0;
	goto _out;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
	switch( (*p) ) {
		case 82: goto st3;
		case 114: goto st3;
	}
	goto st0;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	switch( (*p) ) {
		case 73: goto st4;
		case 105: goto st4;
	}
	goto st0;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 68: goto st105;
		case 100: goto st105;
	}
	goto st0;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
	switch( (*p) ) {
		case 65: goto st6;
		case 68: goto st28;
		case 70: goto st31;
		case 74: goto st34;
		case 77: goto st40;
		case 78: goto st44;
		case 79: goto st47;
		case 83: goto st50;
		case 97: goto st6;
		case 100: goto st28;
		case 102: goto st31;
		case 106: goto st34;
		case 109: goto st40;
		case 110: goto st44;
		case 111: goto st47;
		case 115: goto st50;
	}
	goto st0;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
	switch( (*p) ) {
		case 80: goto st7;
		case 85: goto st26;
		case 112: goto st7;
		case 117: goto st26;
	}
	goto st0;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
	switch( (*p) ) {
		case 82: goto st8;
		case 114: goto st8;
	}
	goto st0;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
	if ( (*p) == 32 )
		goto tr22;
	goto st0;
tr22:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st9;
tr42:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st9;
tr45:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st9;
tr48:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st9;
tr52:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st9;
tr55:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st9;
tr56:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st9;
tr60:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st9;
tr61:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st9;
tr64:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st9;
tr67:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st9;
tr70:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 6611 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto st10;
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr24;
	goto st0;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr25;
	goto st0;
tr25:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st11;
tr40:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 6647 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr26;
	goto st0;
tr26:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 6659 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr27;
	goto st0;
tr27:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 6679 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr28;
	goto st0;
tr28:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 6694 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr29;
	goto st0;
tr29:
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 6706 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr30;
	goto st0;
tr30:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 6726 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr31;
	goto st0;
tr31:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st17;
st17:
	if ( ++p == pe )
		goto _test_eof17;
case 17:
/* #line 6741 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr32;
	goto st0;
tr32:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st18;
st18:
	if ( ++p == pe )
		goto _test_eof18;
case 18:
/* #line 6753 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr33;
	goto st0;
tr33:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st19;
st19:
	if ( ++p == pe )
		goto _test_eof19;
case 19:
/* #line 6773 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr34;
	goto st0;
tr34:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st20;
st20:
	if ( ++p == pe )
		goto _test_eof20;
case 20:
/* #line 6788 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr35;
	goto st0;
tr35:
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st21;
st21:
	if ( ++p == pe )
		goto _test_eof21;
case 21:
/* #line 6800 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr36;
	goto st0;
tr36:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st22;
st22:
	if ( ++p == pe )
		goto _test_eof22;
case 22:
/* #line 6820 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr37;
	goto st0;
tr37:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st23;
st23:
	if ( ++p == pe )
		goto _test_eof23;
case 23:
/* #line 6835 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr38;
	goto st0;
tr38:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st24;
st24:
	if ( ++p == pe )
		goto _test_eof24;
case 24:
/* #line 6850 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr39;
	goto st0;
tr39:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 82 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.SetLooseYear(I); }
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st161;
tr103:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st161;
st161:
	if ( ++p == pe )
		goto _test_eof161;
case 161:
/* #line 6873 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
tr24:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st25;
st25:
	if ( ++p == pe )
		goto _test_eof25;
case 25:
/* #line 6891 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr40;
	goto st0;
st26:
	if ( ++p == pe )
		goto _test_eof26;
case 26:
	switch( (*p) ) {
		case 71: goto st27;
		case 103: goto st27;
	}
	goto st0;
st27:
	if ( ++p == pe )
		goto _test_eof27;
case 27:
	if ( (*p) == 32 )
		goto tr42;
	goto st0;
st28:
	if ( ++p == pe )
		goto _test_eof28;
case 28:
	switch( (*p) ) {
		case 69: goto st29;
		case 101: goto st29;
	}
	goto st0;
st29:
	if ( ++p == pe )
		goto _test_eof29;
case 29:
	switch( (*p) ) {
		case 67: goto st30;
		case 99: goto st30;
	}
	goto st0;
st30:
	if ( ++p == pe )
		goto _test_eof30;
case 30:
	if ( (*p) == 32 )
		goto tr45;
	goto st0;
st31:
	if ( ++p == pe )
		goto _test_eof31;
case 31:
	switch( (*p) ) {
		case 69: goto st32;
		case 101: goto st32;
	}
	goto st0;
st32:
	if ( ++p == pe )
		goto _test_eof32;
case 32:
	switch( (*p) ) {
		case 66: goto st33;
		case 98: goto st33;
	}
	goto st0;
st33:
	if ( ++p == pe )
		goto _test_eof33;
case 33:
	if ( (*p) == 32 )
		goto tr48;
	goto st0;
st34:
	if ( ++p == pe )
		goto _test_eof34;
case 34:
	switch( (*p) ) {
		case 65: goto st35;
		case 85: goto st37;
		case 97: goto st35;
		case 117: goto st37;
	}
	goto st0;
st35:
	if ( ++p == pe )
		goto _test_eof35;
case 35:
	switch( (*p) ) {
		case 78: goto st36;
		case 110: goto st36;
	}
	goto st0;
st36:
	if ( ++p == pe )
		goto _test_eof36;
case 36:
	if ( (*p) == 32 )
		goto tr52;
	goto st0;
st37:
	if ( ++p == pe )
		goto _test_eof37;
case 37:
	switch( (*p) ) {
		case 76: goto st38;
		case 78: goto st39;
		case 108: goto st38;
		case 110: goto st39;
	}
	goto st0;
st38:
	if ( ++p == pe )
		goto _test_eof38;
case 38:
	if ( (*p) == 32 )
		goto tr55;
	goto st0;
st39:
	if ( ++p == pe )
		goto _test_eof39;
case 39:
	if ( (*p) == 32 )
		goto tr56;
	goto st0;
st40:
	if ( ++p == pe )
		goto _test_eof40;
case 40:
	switch( (*p) ) {
		case 65: goto st41;
		case 97: goto st41;
	}
	goto st0;
st41:
	if ( ++p == pe )
		goto _test_eof41;
case 41:
	switch( (*p) ) {
		case 82: goto st42;
		case 89: goto st43;
		case 114: goto st42;
		case 121: goto st43;
	}
	goto st0;
st42:
	if ( ++p == pe )
		goto _test_eof42;
case 42:
	if ( (*p) == 32 )
		goto tr60;
	goto st0;
st43:
	if ( ++p == pe )
		goto _test_eof43;
case 43:
	if ( (*p) == 32 )
		goto tr61;
	goto st0;
st44:
	if ( ++p == pe )
		goto _test_eof44;
case 44:
	switch( (*p) ) {
		case 79: goto st45;
		case 111: goto st45;
	}
	goto st0;
st45:
	if ( ++p == pe )
		goto _test_eof45;
case 45:
	switch( (*p) ) {
		case 86: goto st46;
		case 118: goto st46;
	}
	goto st0;
st46:
	if ( ++p == pe )
		goto _test_eof46;
case 46:
	if ( (*p) == 32 )
		goto tr64;
	goto st0;
st47:
	if ( ++p == pe )
		goto _test_eof47;
case 47:
	switch( (*p) ) {
		case 67: goto st48;
		case 99: goto st48;
	}
	goto st0;
st48:
	if ( ++p == pe )
		goto _test_eof48;
case 48:
	switch( (*p) ) {
		case 84: goto st49;
		case 116: goto st49;
	}
	goto st0;
st49:
	if ( ++p == pe )
		goto _test_eof49;
case 49:
	if ( (*p) == 32 )
		goto tr67;
	goto st0;
st50:
	if ( ++p == pe )
		goto _test_eof50;
case 50:
	switch( (*p) ) {
		case 69: goto st51;
		case 101: goto st51;
	}
	goto st0;
st51:
	if ( ++p == pe )
		goto _test_eof51;
case 51:
	switch( (*p) ) {
		case 80: goto st52;
		case 112: goto st52;
	}
	goto st0;
st52:
	if ( ++p == pe )
		goto _test_eof52;
case 52:
	if ( (*p) == 32 )
		goto tr70;
	goto st0;
st53:
	if ( ++p == pe )
		goto _test_eof53;
case 53:
	if ( (*p) == 32 )
		goto st54;
	goto st0;
st54:
	if ( ++p == pe )
		goto _test_eof54;
case 54:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr72;
	goto st0;
tr72:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st55;
st55:
	if ( ++p == pe )
		goto _test_eof55;
case 55:
/* #line 7152 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr73;
	goto st0;
tr73:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st56;
st56:
	if ( ++p == pe )
		goto _test_eof56;
case 56:
/* #line 7167 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr74;
	goto st0;
tr74:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st57;
st57:
	if ( ++p == pe )
		goto _test_eof57;
case 57:
/* #line 7179 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 65: goto st58;
		case 68: goto st80;
		case 70: goto st83;
		case 74: goto st86;
		case 77: goto st92;
		case 78: goto st96;
		case 79: goto st99;
		case 83: goto st102;
		case 97: goto st58;
		case 100: goto st80;
		case 102: goto st83;
		case 106: goto st86;
		case 109: goto st92;
		case 110: goto st96;
		case 111: goto st99;
		case 115: goto st102;
	}
	goto st0;
st58:
	if ( ++p == pe )
		goto _test_eof58;
case 58:
	switch( (*p) ) {
		case 80: goto st59;
		case 85: goto st78;
		case 112: goto st59;
		case 117: goto st78;
	}
	goto st0;
st59:
	if ( ++p == pe )
		goto _test_eof59;
case 59:
	switch( (*p) ) {
		case 82: goto st60;
		case 114: goto st60;
	}
	goto st0;
st60:
	if ( ++p == pe )
		goto _test_eof60;
case 60:
	if ( (*p) == 32 )
		goto tr86;
	goto st0;
tr86:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st61;
tr105:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st61;
tr108:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st61;
tr111:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st61;
tr115:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st61;
tr118:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st61;
tr119:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st61;
tr123:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st61;
tr124:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st61;
tr127:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st61;
tr130:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st61;
tr133:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st61;
st61:
	if ( ++p == pe )
		goto _test_eof61;
case 61:
/* #line 7278 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr87;
	goto st0;
tr87:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st62;
st62:
	if ( ++p == pe )
		goto _test_eof62;
case 62:
/* #line 7298 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr88;
	goto st0;
tr88:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st63;
st63:
	if ( ++p == pe )
		goto _test_eof63;
case 63:
/* #line 7313 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr89;
	goto st0;
tr153:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st64;
tr89:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st64;
st64:
	if ( ++p == pe )
		goto _test_eof64;
case 64:
/* #line 7340 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr90;
	goto st0;
tr90:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st65;
st65:
	if ( ++p == pe )
		goto _test_eof65;
case 65:
/* #line 7355 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr91;
	goto st0;
tr91:
/* #line 82 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.SetLooseYear(I); }
	goto st66;
st66:
	if ( ++p == pe )
		goto _test_eof66;
case 66:
/* #line 7367 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr92;
	goto st0;
tr92:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st67;
st67:
	if ( ++p == pe )
		goto _test_eof67;
case 67:
/* #line 7387 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr93;
	goto st0;
tr93:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st68;
st68:
	if ( ++p == pe )
		goto _test_eof68;
case 68:
/* #line 7402 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr94;
	goto st0;
tr94:
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st69;
st69:
	if ( ++p == pe )
		goto _test_eof69;
case 69:
/* #line 7414 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr95;
	goto st0;
tr95:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st70;
st70:
	if ( ++p == pe )
		goto _test_eof70;
case 70:
/* #line 7434 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr96;
	goto st0;
tr96:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st71;
st71:
	if ( ++p == pe )
		goto _test_eof71;
case 71:
/* #line 7449 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 58 )
		goto tr97;
	goto st0;
tr97:
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st72;
st72:
	if ( ++p == pe )
		goto _test_eof72;
case 72:
/* #line 7461 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr98;
	goto st0;
tr98:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st73;
st73:
	if ( ++p == pe )
		goto _test_eof73;
case 73:
/* #line 7481 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr99;
	goto st0;
tr99:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st74;
st74:
	if ( ++p == pe )
		goto _test_eof74;
case 74:
/* #line 7496 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 32 )
		goto tr100;
	goto st0;
tr100:
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st75;
st75:
	if ( ++p == pe )
		goto _test_eof75;
case 75:
/* #line 7508 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 71: goto st76;
		case 103: goto st76;
	}
	goto st0;
st76:
	if ( ++p == pe )
		goto _test_eof76;
case 76:
	switch( (*p) ) {
		case 77: goto st77;
		case 109: goto st77;
	}
	goto st0;
st77:
	if ( ++p == pe )
		goto _test_eof77;
case 77:
	switch( (*p) ) {
		case 84: goto tr103;
		case 116: goto tr103;
	}
	goto st0;
st78:
	if ( ++p == pe )
		goto _test_eof78;
case 78:
	switch( (*p) ) {
		case 71: goto st79;
		case 103: goto st79;
	}
	goto st0;
st79:
	if ( ++p == pe )
		goto _test_eof79;
case 79:
	if ( (*p) == 32 )
		goto tr105;
	goto st0;
st80:
	if ( ++p == pe )
		goto _test_eof80;
case 80:
	switch( (*p) ) {
		case 69: goto st81;
		case 101: goto st81;
	}
	goto st0;
st81:
	if ( ++p == pe )
		goto _test_eof81;
case 81:
	switch( (*p) ) {
		case 67: goto st82;
		case 99: goto st82;
	}
	goto st0;
st82:
	if ( ++p == pe )
		goto _test_eof82;
case 82:
	if ( (*p) == 32 )
		goto tr108;
	goto st0;
st83:
	if ( ++p == pe )
		goto _test_eof83;
case 83:
	switch( (*p) ) {
		case 69: goto st84;
		case 101: goto st84;
	}
	goto st0;
st84:
	if ( ++p == pe )
		goto _test_eof84;
case 84:
	switch( (*p) ) {
		case 66: goto st85;
		case 98: goto st85;
	}
	goto st0;
st85:
	if ( ++p == pe )
		goto _test_eof85;
case 85:
	if ( (*p) == 32 )
		goto tr111;
	goto st0;
st86:
	if ( ++p == pe )
		goto _test_eof86;
case 86:
	switch( (*p) ) {
		case 65: goto st87;
		case 85: goto st89;
		case 97: goto st87;
		case 117: goto st89;
	}
	goto st0;
st87:
	if ( ++p == pe )
		goto _test_eof87;
case 87:
	switch( (*p) ) {
		case 78: goto st88;
		case 110: goto st88;
	}
	goto st0;
st88:
	if ( ++p == pe )
		goto _test_eof88;
case 88:
	if ( (*p) == 32 )
		goto tr115;
	goto st0;
st89:
	if ( ++p == pe )
		goto _test_eof89;
case 89:
	switch( (*p) ) {
		case 76: goto st90;
		case 78: goto st91;
		case 108: goto st90;
		case 110: goto st91;
	}
	goto st0;
st90:
	if ( ++p == pe )
		goto _test_eof90;
case 90:
	if ( (*p) == 32 )
		goto tr118;
	goto st0;
st91:
	if ( ++p == pe )
		goto _test_eof91;
case 91:
	if ( (*p) == 32 )
		goto tr119;
	goto st0;
st92:
	if ( ++p == pe )
		goto _test_eof92;
case 92:
	switch( (*p) ) {
		case 65: goto st93;
		case 97: goto st93;
	}
	goto st0;
st93:
	if ( ++p == pe )
		goto _test_eof93;
case 93:
	switch( (*p) ) {
		case 82: goto st94;
		case 89: goto st95;
		case 114: goto st94;
		case 121: goto st95;
	}
	goto st0;
st94:
	if ( ++p == pe )
		goto _test_eof94;
case 94:
	if ( (*p) == 32 )
		goto tr123;
	goto st0;
st95:
	if ( ++p == pe )
		goto _test_eof95;
case 95:
	if ( (*p) == 32 )
		goto tr124;
	goto st0;
st96:
	if ( ++p == pe )
		goto _test_eof96;
case 96:
	switch( (*p) ) {
		case 79: goto st97;
		case 111: goto st97;
	}
	goto st0;
st97:
	if ( ++p == pe )
		goto _test_eof97;
case 97:
	switch( (*p) ) {
		case 86: goto st98;
		case 118: goto st98;
	}
	goto st0;
st98:
	if ( ++p == pe )
		goto _test_eof98;
case 98:
	if ( (*p) == 32 )
		goto tr127;
	goto st0;
st99:
	if ( ++p == pe )
		goto _test_eof99;
case 99:
	switch( (*p) ) {
		case 67: goto st100;
		case 99: goto st100;
	}
	goto st0;
st100:
	if ( ++p == pe )
		goto _test_eof100;
case 100:
	switch( (*p) ) {
		case 84: goto st101;
		case 116: goto st101;
	}
	goto st0;
st101:
	if ( ++p == pe )
		goto _test_eof101;
case 101:
	if ( (*p) == 32 )
		goto tr130;
	goto st0;
st102:
	if ( ++p == pe )
		goto _test_eof102;
case 102:
	switch( (*p) ) {
		case 69: goto st103;
		case 101: goto st103;
	}
	goto st0;
st103:
	if ( ++p == pe )
		goto _test_eof103;
case 103:
	switch( (*p) ) {
		case 80: goto st104;
		case 112: goto st104;
	}
	goto st0;
st104:
	if ( ++p == pe )
		goto _test_eof104;
case 104:
	if ( (*p) == 32 )
		goto tr133;
	goto st0;
st105:
	if ( ++p == pe )
		goto _test_eof105;
case 105:
	switch( (*p) ) {
		case 65: goto st106;
		case 97: goto st106;
	}
	goto st0;
st106:
	if ( ++p == pe )
		goto _test_eof106;
case 106:
	switch( (*p) ) {
		case 89: goto st107;
		case 121: goto st107;
	}
	goto st0;
st107:
	if ( ++p == pe )
		goto _test_eof107;
case 107:
	if ( (*p) == 44 )
		goto st108;
	goto st0;
st108:
	if ( ++p == pe )
		goto _test_eof108;
case 108:
	if ( (*p) == 32 )
		goto st109;
	goto st0;
st109:
	if ( ++p == pe )
		goto _test_eof109;
case 109:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr138;
	goto st0;
tr138:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st110;
st110:
	if ( ++p == pe )
		goto _test_eof110;
case 110:
/* #line 7814 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr139;
	goto st0;
tr139:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st111;
st111:
	if ( ++p == pe )
		goto _test_eof111;
case 111:
/* #line 7829 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 45 )
		goto tr140;
	goto st0;
tr140:
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st112;
st112:
	if ( ++p == pe )
		goto _test_eof112;
case 112:
/* #line 7841 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 65: goto st113;
		case 68: goto st119;
		case 70: goto st122;
		case 74: goto st125;
		case 77: goto st131;
		case 78: goto st135;
		case 79: goto st138;
		case 83: goto st141;
		case 97: goto st113;
		case 100: goto st119;
		case 102: goto st122;
		case 106: goto st125;
		case 109: goto st131;
		case 110: goto st135;
		case 111: goto st138;
		case 115: goto st141;
	}
	goto st0;
st113:
	if ( ++p == pe )
		goto _test_eof113;
case 113:
	switch( (*p) ) {
		case 80: goto st114;
		case 85: goto st117;
		case 112: goto st114;
		case 117: goto st117;
	}
	goto st0;
st114:
	if ( ++p == pe )
		goto _test_eof114;
case 114:
	switch( (*p) ) {
		case 82: goto st115;
		case 114: goto st115;
	}
	goto st0;
st115:
	if ( ++p == pe )
		goto _test_eof115;
case 115:
	if ( (*p) == 45 )
		goto tr152;
	goto st0;
tr152:
/* #line 63 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  4; }
	goto st116;
tr155:
/* #line 67 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  8; }
	goto st116;
tr158:
/* #line 71 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 12; }
	goto st116;
tr161:
/* #line 61 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  2; }
	goto st116;
tr165:
/* #line 60 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  1; }
	goto st116;
tr168:
/* #line 66 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  7; }
	goto st116;
tr169:
/* #line 65 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  6; }
	goto st116;
tr173:
/* #line 62 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  3; }
	goto st116;
tr174:
/* #line 64 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  5; }
	goto st116;
tr177:
/* #line 70 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 11; }
	goto st116;
tr180:
/* #line 69 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = 10; }
	goto st116;
tr183:
/* #line 68 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month =  9; }
	goto st116;
st116:
	if ( ++p == pe )
		goto _test_eof116;
case 116:
/* #line 7940 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr153;
	goto st0;
st117:
	if ( ++p == pe )
		goto _test_eof117;
case 117:
	switch( (*p) ) {
		case 71: goto st118;
		case 103: goto st118;
	}
	goto st0;
st118:
	if ( ++p == pe )
		goto _test_eof118;
case 118:
	if ( (*p) == 45 )
		goto tr155;
	goto st0;
st119:
	if ( ++p == pe )
		goto _test_eof119;
case 119:
	switch( (*p) ) {
		case 69: goto st120;
		case 101: goto st120;
	}
	goto st0;
st120:
	if ( ++p == pe )
		goto _test_eof120;
case 120:
	switch( (*p) ) {
		case 67: goto st121;
		case 99: goto st121;
	}
	goto st0;
st121:
	if ( ++p == pe )
		goto _test_eof121;
case 121:
	if ( (*p) == 45 )
		goto tr158;
	goto st0;
st122:
	if ( ++p == pe )
		goto _test_eof122;
case 122:
	switch( (*p) ) {
		case 69: goto st123;
		case 101: goto st123;
	}
	goto st0;
st123:
	if ( ++p == pe )
		goto _test_eof123;
case 123:
	switch( (*p) ) {
		case 66: goto st124;
		case 98: goto st124;
	}
	goto st0;
st124:
	if ( ++p == pe )
		goto _test_eof124;
case 124:
	if ( (*p) == 45 )
		goto tr161;
	goto st0;
st125:
	if ( ++p == pe )
		goto _test_eof125;
case 125:
	switch( (*p) ) {
		case 65: goto st126;
		case 85: goto st128;
		case 97: goto st126;
		case 117: goto st128;
	}
	goto st0;
st126:
	if ( ++p == pe )
		goto _test_eof126;
case 126:
	switch( (*p) ) {
		case 78: goto st127;
		case 110: goto st127;
	}
	goto st0;
st127:
	if ( ++p == pe )
		goto _test_eof127;
case 127:
	if ( (*p) == 45 )
		goto tr165;
	goto st0;
st128:
	if ( ++p == pe )
		goto _test_eof128;
case 128:
	switch( (*p) ) {
		case 76: goto st129;
		case 78: goto st130;
		case 108: goto st129;
		case 110: goto st130;
	}
	goto st0;
st129:
	if ( ++p == pe )
		goto _test_eof129;
case 129:
	if ( (*p) == 45 )
		goto tr168;
	goto st0;
st130:
	if ( ++p == pe )
		goto _test_eof130;
case 130:
	if ( (*p) == 45 )
		goto tr169;
	goto st0;
st131:
	if ( ++p == pe )
		goto _test_eof131;
case 131:
	switch( (*p) ) {
		case 65: goto st132;
		case 97: goto st132;
	}
	goto st0;
st132:
	if ( ++p == pe )
		goto _test_eof132;
case 132:
	switch( (*p) ) {
		case 82: goto st133;
		case 89: goto st134;
		case 114: goto st133;
		case 121: goto st134;
	}
	goto st0;
st133:
	if ( ++p == pe )
		goto _test_eof133;
case 133:
	if ( (*p) == 45 )
		goto tr173;
	goto st0;
st134:
	if ( ++p == pe )
		goto _test_eof134;
case 134:
	if ( (*p) == 45 )
		goto tr174;
	goto st0;
st135:
	if ( ++p == pe )
		goto _test_eof135;
case 135:
	switch( (*p) ) {
		case 79: goto st136;
		case 111: goto st136;
	}
	goto st0;
st136:
	if ( ++p == pe )
		goto _test_eof136;
case 136:
	switch( (*p) ) {
		case 86: goto st137;
		case 118: goto st137;
	}
	goto st0;
st137:
	if ( ++p == pe )
		goto _test_eof137;
case 137:
	if ( (*p) == 45 )
		goto tr177;
	goto st0;
st138:
	if ( ++p == pe )
		goto _test_eof138;
case 138:
	switch( (*p) ) {
		case 67: goto st139;
		case 99: goto st139;
	}
	goto st0;
st139:
	if ( ++p == pe )
		goto _test_eof139;
case 139:
	switch( (*p) ) {
		case 84: goto st140;
		case 116: goto st140;
	}
	goto st0;
st140:
	if ( ++p == pe )
		goto _test_eof140;
case 140:
	if ( (*p) == 45 )
		goto tr180;
	goto st0;
st141:
	if ( ++p == pe )
		goto _test_eof141;
case 141:
	switch( (*p) ) {
		case 69: goto st142;
		case 101: goto st142;
	}
	goto st0;
st142:
	if ( ++p == pe )
		goto _test_eof142;
case 142:
	switch( (*p) ) {
		case 80: goto st143;
		case 112: goto st143;
	}
	goto st0;
st143:
	if ( ++p == pe )
		goto _test_eof143;
case 143:
	if ( (*p) == 45 )
		goto tr183;
	goto st0;
st144:
	if ( ++p == pe )
		goto _test_eof144;
case 144:
	switch( (*p) ) {
		case 79: goto st145;
		case 111: goto st145;
	}
	goto st0;
st145:
	if ( ++p == pe )
		goto _test_eof145;
case 145:
	switch( (*p) ) {
		case 78: goto st4;
		case 110: goto st4;
	}
	goto st0;
st146:
	if ( ++p == pe )
		goto _test_eof146;
case 146:
	switch( (*p) ) {
		case 65: goto st147;
		case 85: goto st145;
		case 97: goto st147;
		case 117: goto st145;
	}
	goto st0;
st147:
	if ( ++p == pe )
		goto _test_eof147;
case 147:
	switch( (*p) ) {
		case 84: goto st148;
		case 116: goto st148;
	}
	goto st0;
st148:
	if ( ++p == pe )
		goto _test_eof148;
case 148:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 85: goto st149;
		case 117: goto st149;
	}
	goto st0;
st149:
	if ( ++p == pe )
		goto _test_eof149;
case 149:
	switch( (*p) ) {
		case 82: goto st150;
		case 114: goto st150;
	}
	goto st0;
st150:
	if ( ++p == pe )
		goto _test_eof150;
case 150:
	switch( (*p) ) {
		case 68: goto st105;
		case 100: goto st105;
	}
	goto st0;
st151:
	if ( ++p == pe )
		goto _test_eof151;
case 151:
	switch( (*p) ) {
		case 72: goto st152;
		case 85: goto st155;
		case 104: goto st152;
		case 117: goto st155;
	}
	goto st0;
st152:
	if ( ++p == pe )
		goto _test_eof152;
case 152:
	switch( (*p) ) {
		case 85: goto st153;
		case 117: goto st153;
	}
	goto st0;
st153:
	if ( ++p == pe )
		goto _test_eof153;
case 153:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 82: goto st154;
		case 114: goto st154;
	}
	goto st0;
st154:
	if ( ++p == pe )
		goto _test_eof154;
case 154:
	switch( (*p) ) {
		case 83: goto st150;
		case 115: goto st150;
	}
	goto st0;
st155:
	if ( ++p == pe )
		goto _test_eof155;
case 155:
	switch( (*p) ) {
		case 69: goto st156;
		case 101: goto st156;
	}
	goto st0;
st156:
	if ( ++p == pe )
		goto _test_eof156;
case 156:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 83: goto st150;
		case 115: goto st150;
	}
	goto st0;
st157:
	if ( ++p == pe )
		goto _test_eof157;
case 157:
	switch( (*p) ) {
		case 69: goto st158;
		case 101: goto st158;
	}
	goto st0;
st158:
	if ( ++p == pe )
		goto _test_eof158;
case 158:
	switch( (*p) ) {
		case 68: goto st159;
		case 100: goto st159;
	}
	goto st0;
st159:
	if ( ++p == pe )
		goto _test_eof159;
case 159:
	switch( (*p) ) {
		case 32: goto st5;
		case 44: goto st53;
		case 78: goto st160;
		case 110: goto st160;
	}
	goto st0;
st160:
	if ( ++p == pe )
		goto _test_eof160;
case 160:
	switch( (*p) ) {
		case 69: goto st154;
		case 101: goto st154;
	}
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 
	_test_eof17: cs = 17; goto _test_eof; 
	_test_eof18: cs = 18; goto _test_eof; 
	_test_eof19: cs = 19; goto _test_eof; 
	_test_eof20: cs = 20; goto _test_eof; 
	_test_eof21: cs = 21; goto _test_eof; 
	_test_eof22: cs = 22; goto _test_eof; 
	_test_eof23: cs = 23; goto _test_eof; 
	_test_eof24: cs = 24; goto _test_eof; 
	_test_eof161: cs = 161; goto _test_eof; 
	_test_eof25: cs = 25; goto _test_eof; 
	_test_eof26: cs = 26; goto _test_eof; 
	_test_eof27: cs = 27; goto _test_eof; 
	_test_eof28: cs = 28; goto _test_eof; 
	_test_eof29: cs = 29; goto _test_eof; 
	_test_eof30: cs = 30; goto _test_eof; 
	_test_eof31: cs = 31; goto _test_eof; 
	_test_eof32: cs = 32; goto _test_eof; 
	_test_eof33: cs = 33; goto _test_eof; 
	_test_eof34: cs = 34; goto _test_eof; 
	_test_eof35: cs = 35; goto _test_eof; 
	_test_eof36: cs = 36; goto _test_eof; 
	_test_eof37: cs = 37; goto _test_eof; 
	_test_eof38: cs = 38; goto _test_eof; 
	_test_eof39: cs = 39; goto _test_eof; 
	_test_eof40: cs = 40; goto _test_eof; 
	_test_eof41: cs = 41; goto _test_eof; 
	_test_eof42: cs = 42; goto _test_eof; 
	_test_eof43: cs = 43; goto _test_eof; 
	_test_eof44: cs = 44; goto _test_eof; 
	_test_eof45: cs = 45; goto _test_eof; 
	_test_eof46: cs = 46; goto _test_eof; 
	_test_eof47: cs = 47; goto _test_eof; 
	_test_eof48: cs = 48; goto _test_eof; 
	_test_eof49: cs = 49; goto _test_eof; 
	_test_eof50: cs = 50; goto _test_eof; 
	_test_eof51: cs = 51; goto _test_eof; 
	_test_eof52: cs = 52; goto _test_eof; 
	_test_eof53: cs = 53; goto _test_eof; 
	_test_eof54: cs = 54; goto _test_eof; 
	_test_eof55: cs = 55; goto _test_eof; 
	_test_eof56: cs = 56; goto _test_eof; 
	_test_eof57: cs = 57; goto _test_eof; 
	_test_eof58: cs = 58; goto _test_eof; 
	_test_eof59: cs = 59; goto _test_eof; 
	_test_eof60: cs = 60; goto _test_eof; 
	_test_eof61: cs = 61; goto _test_eof; 
	_test_eof62: cs = 62; goto _test_eof; 
	_test_eof63: cs = 63; goto _test_eof; 
	_test_eof64: cs = 64; goto _test_eof; 
	_test_eof65: cs = 65; goto _test_eof; 
	_test_eof66: cs = 66; goto _test_eof; 
	_test_eof67: cs = 67; goto _test_eof; 
	_test_eof68: cs = 68; goto _test_eof; 
	_test_eof69: cs = 69; goto _test_eof; 
	_test_eof70: cs = 70; goto _test_eof; 
	_test_eof71: cs = 71; goto _test_eof; 
	_test_eof72: cs = 72; goto _test_eof; 
	_test_eof73: cs = 73; goto _test_eof; 
	_test_eof74: cs = 74; goto _test_eof; 
	_test_eof75: cs = 75; goto _test_eof; 
	_test_eof76: cs = 76; goto _test_eof; 
	_test_eof77: cs = 77; goto _test_eof; 
	_test_eof78: cs = 78; goto _test_eof; 
	_test_eof79: cs = 79; goto _test_eof; 
	_test_eof80: cs = 80; goto _test_eof; 
	_test_eof81: cs = 81; goto _test_eof; 
	_test_eof82: cs = 82; goto _test_eof; 
	_test_eof83: cs = 83; goto _test_eof; 
	_test_eof84: cs = 84; goto _test_eof; 
	_test_eof85: cs = 85; goto _test_eof; 
	_test_eof86: cs = 86; goto _test_eof; 
	_test_eof87: cs = 87; goto _test_eof; 
	_test_eof88: cs = 88; goto _test_eof; 
	_test_eof89: cs = 89; goto _test_eof; 
	_test_eof90: cs = 90; goto _test_eof; 
	_test_eof91: cs = 91; goto _test_eof; 
	_test_eof92: cs = 92; goto _test_eof; 
	_test_eof93: cs = 93; goto _test_eof; 
	_test_eof94: cs = 94; goto _test_eof; 
	_test_eof95: cs = 95; goto _test_eof; 
	_test_eof96: cs = 96; goto _test_eof; 
	_test_eof97: cs = 97; goto _test_eof; 
	_test_eof98: cs = 98; goto _test_eof; 
	_test_eof99: cs = 99; goto _test_eof; 
	_test_eof100: cs = 100; goto _test_eof; 
	_test_eof101: cs = 101; goto _test_eof; 
	_test_eof102: cs = 102; goto _test_eof; 
	_test_eof103: cs = 103; goto _test_eof; 
	_test_eof104: cs = 104; goto _test_eof; 
	_test_eof105: cs = 105; goto _test_eof; 
	_test_eof106: cs = 106; goto _test_eof; 
	_test_eof107: cs = 107; goto _test_eof; 
	_test_eof108: cs = 108; goto _test_eof; 
	_test_eof109: cs = 109; goto _test_eof; 
	_test_eof110: cs = 110; goto _test_eof; 
	_test_eof111: cs = 111; goto _test_eof; 
	_test_eof112: cs = 112; goto _test_eof; 
	_test_eof113: cs = 113; goto _test_eof; 
	_test_eof114: cs = 114; goto _test_eof; 
	_test_eof115: cs = 115; goto _test_eof; 
	_test_eof116: cs = 116; goto _test_eof; 
	_test_eof117: cs = 117; goto _test_eof; 
	_test_eof118: cs = 118; goto _test_eof; 
	_test_eof119: cs = 119; goto _test_eof; 
	_test_eof120: cs = 120; goto _test_eof; 
	_test_eof121: cs = 121; goto _test_eof; 
	_test_eof122: cs = 122; goto _test_eof; 
	_test_eof123: cs = 123; goto _test_eof; 
	_test_eof124: cs = 124; goto _test_eof; 
	_test_eof125: cs = 125; goto _test_eof; 
	_test_eof126: cs = 126; goto _test_eof; 
	_test_eof127: cs = 127; goto _test_eof; 
	_test_eof128: cs = 128; goto _test_eof; 
	_test_eof129: cs = 129; goto _test_eof; 
	_test_eof130: cs = 130; goto _test_eof; 
	_test_eof131: cs = 131; goto _test_eof; 
	_test_eof132: cs = 132; goto _test_eof; 
	_test_eof133: cs = 133; goto _test_eof; 
	_test_eof134: cs = 134; goto _test_eof; 
	_test_eof135: cs = 135; goto _test_eof; 
	_test_eof136: cs = 136; goto _test_eof; 
	_test_eof137: cs = 137; goto _test_eof; 
	_test_eof138: cs = 138; goto _test_eof; 
	_test_eof139: cs = 139; goto _test_eof; 
	_test_eof140: cs = 140; goto _test_eof; 
	_test_eof141: cs = 141; goto _test_eof; 
	_test_eof142: cs = 142; goto _test_eof; 
	_test_eof143: cs = 143; goto _test_eof; 
	_test_eof144: cs = 144; goto _test_eof; 
	_test_eof145: cs = 145; goto _test_eof; 
	_test_eof146: cs = 146; goto _test_eof; 
	_test_eof147: cs = 147; goto _test_eof; 
	_test_eof148: cs = 148; goto _test_eof; 
	_test_eof149: cs = 149; goto _test_eof; 
	_test_eof150: cs = 150; goto _test_eof; 
	_test_eof151: cs = 151; goto _test_eof; 
	_test_eof152: cs = 152; goto _test_eof; 
	_test_eof153: cs = 153; goto _test_eof; 
	_test_eof154: cs = 154; goto _test_eof; 
	_test_eof155: cs = 155; goto _test_eof; 
	_test_eof156: cs = 156; goto _test_eof; 
	_test_eof157: cs = 157; goto _test_eof; 
	_test_eof158: cs = 158; goto _test_eof; 
	_test_eof159: cs = 159; goto _test_eof; 
	_test_eof160: cs = 160; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 304 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}


/* #line 8507 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
static const int X509ValidityDateTimeParser_start = 1;
static const int X509ValidityDateTimeParser_first_final = 14;

static const int X509ValidityDateTimeParser_en_main = 1;


/* #line 327 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */


TX509ValidityDateTimeParserDeprecated::TX509ValidityDateTimeParserDeprecated() {
    
/* #line 8519 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = X509ValidityDateTimeParser_start;
	}

/* #line 331 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TX509ValidityDateTimeParserDeprecated::ParsePart(const char *input, size_t len) {
    const char *p = input;
    const char *pe = input + len;

    
/* #line 8532 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 8561 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 315 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Year = (I < 50 ? I + 2000 : I + 1900); }
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 8578 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr3;
	goto st0;
tr3:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
/* #line 8598 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr4;
	goto st0;
tr4:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 8615 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr5;
	goto st0;
tr5:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st6;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
/* #line 8635 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
tr6:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 8652 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr7;
	goto st0;
tr7:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 8672 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr8;
	goto st0;
tr8:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 8689 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr9;
	goto st0;
tr9:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 8709 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr10;
	goto st0;
tr10:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 8726 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr11;
	goto st0;
tr11:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 8746 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr12;
	goto st0;
tr12:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 8763 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 90 )
		goto tr13;
	goto st0;
tr13:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 8775 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 338 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

TX509ValidityDateTimeParser::TX509ValidityDateTimeParser() {
    
/* #line 8802 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = X509ValidityDateTimeParser_start;
	}

/* #line 343 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TX509ValidityDateTimeParser::ParsePart(const char *input, size_t len) {
    const char *p = input;
    const char *pe = input + len;

    
/* #line 8815 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 8844 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 315 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Year = (I < 50 ? I + 2000 : I + 1900); }
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 8861 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr3;
	goto st0;
tr3:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
/* #line 8881 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr4;
	goto st0;
tr4:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 8898 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr5;
	goto st0;
tr5:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st6;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
/* #line 8918 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
tr6:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 8935 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr7;
	goto st0;
tr7:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 8955 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr8;
	goto st0;
tr8:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 8972 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr9;
	goto st0;
tr9:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 8992 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr10;
	goto st0;
tr10:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 9009 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr11;
	goto st0;
tr11:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 9029 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr12;
	goto st0;
tr12:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 9046 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 90 )
		goto tr13;
	goto st0;
tr13:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 9058 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 350 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}


/* #line 9084 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
static const int X509Validity4yDateTimeParser_start = 1;
static const int X509Validity4yDateTimeParser_first_final = 16;

static const int X509Validity4yDateTimeParser_en_main = 1;


/* #line 372 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */


TX509Validity4yDateTimeParserDeprecated::TX509Validity4yDateTimeParserDeprecated() {
    
/* #line 9096 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = X509Validity4yDateTimeParser_start;
	}

/* #line 376 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TX509Validity4yDateTimeParserDeprecated::ParsePart(const char *input, size_t len) {
    const char *p = input;
    const char *pe = input + len;

    
/* #line 9109 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 9138 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 9153 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr3;
	goto st0;
tr3:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
/* #line 9168 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr4;
	goto st0;
tr4:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 359 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Year = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 9185 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr5;
	goto st0;
tr5:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st6;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
/* #line 9205 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
tr6:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 9222 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr7;
	goto st0;
tr7:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 9242 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr8;
	goto st0;
tr8:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 9259 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr9;
	goto st0;
tr9:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 9279 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr10;
	goto st0;
tr10:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 9296 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr11;
	goto st0;
tr11:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 9316 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr12;
	goto st0;
tr12:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 9333 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr13;
	goto st0;
tr13:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 9353 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr14;
	goto st0;
tr14:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 9370 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 90 )
		goto tr15;
	goto st0;
tr15:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 9382 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 383 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

TX509Validity4yDateTimeParser::TX509Validity4yDateTimeParser() {
    
/* #line 9411 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = X509Validity4yDateTimeParser_start;
	}

/* #line 388 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TX509Validity4yDateTimeParser::ParsePart(const char *input, size_t len) {
    const char *p = input;
    const char *pe = input + len;

    
/* #line 9424 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st2;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
/* #line 9453 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st3;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
/* #line 9468 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr3;
	goto st0;
tr3:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st4;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
/* #line 9483 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr4;
	goto st0;
tr4:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 359 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Year = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 9500 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr5;
	goto st0;
tr5:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st6;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
/* #line 9520 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
tr6:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 81 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Month = I; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 9537 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr7;
	goto st0;
tr7:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 9557 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr8;
	goto st0;
tr8:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 80 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Day = I; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 9574 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr9;
	goto st0;
tr9:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 9594 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr10;
	goto st0;
tr10:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 79 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Hour = I; }
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 9611 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr11;
	goto st0;
tr11:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 9631 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr12;
	goto st0;
tr12:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 78 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Minute = I; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 9648 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr13;
	goto st0;
tr13:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
	goto st14;
st14:
	if ( ++p == pe )
		goto _test_eof14;
case 14:
/* #line 9668 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr14;
	goto st0;
tr14:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 77 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.Second = I; }
	goto st15;
st15:
	if ( ++p == pe )
		goto _test_eof15;
case 15:
/* #line 9685 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 90 )
		goto tr15;
	goto st0;
tr15:
/* #line 84 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ DateTimeFields.ZoneOffsetMinutes = 0; }
	goto st16;
st16:
	if ( ++p == pe )
		goto _test_eof16;
case 16:
/* #line 9697 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
	}
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof14: cs = 14; goto _test_eof; 
	_test_eof15: cs = 15; goto _test_eof; 
	_test_eof16: cs = 16; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 395 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

TInstant TIso8601DateTimeParserDeprecated::GetResult(TInstant defaultValue) const {
    Y_UNUSED(ISO8601DateTimeParser_en_main);
    return TDateTimeParserBaseDeprecated::GetResult(ISO8601DateTimeParser_first_final, defaultValue);
}

TInstant TRfc822DateTimeParserDeprecated::GetResult(TInstant defaultValue) const {
    Y_UNUSED(RFC822DateParser_en_main);
    return TDateTimeParserBaseDeprecated::GetResult(RFC822DateParser_first_final, defaultValue);
}

TInstant THttpDateTimeParserDeprecated::GetResult(TInstant defaultValue) const {
    Y_UNUSED(HttpDateTimeParserStandalone_en_main);
    return TDateTimeParserBaseDeprecated::GetResult(HttpDateTimeParserStandalone_first_final, defaultValue);
}

TInstant TX509ValidityDateTimeParserDeprecated::GetResult(TInstant defaultValue) const {
    Y_UNUSED(X509ValidityDateTimeParser_en_main);
    return TDateTimeParserBaseDeprecated::GetResult(X509ValidityDateTimeParser_first_final, defaultValue);
}

TInstant TX509Validity4yDateTimeParserDeprecated::GetResult(TInstant defaultValue) const {
    Y_UNUSED(X509Validity4yDateTimeParser_en_main);
    return TDateTimeParserBaseDeprecated::GetResult(X509Validity4yDateTimeParser_first_final, defaultValue);
}

TInstant TIso8601DateTimeParser::GetResult(TInstant defaultValue) const {
    Y_UNUSED(ISO8601DateTimeParser_en_main);
    return TDateTimeParserBase::GetResult(ISO8601DateTimeParser_first_final, defaultValue);
}

TInstant TRfc822DateTimeParser::GetResult(TInstant defaultValue) const {
    Y_UNUSED(RFC822DateParser_en_main);
    return TDateTimeParserBase::GetResult(RFC822DateParser_first_final, defaultValue);
}

TInstant THttpDateTimeParser::GetResult(TInstant defaultValue) const {
    Y_UNUSED(HttpDateTimeParserStandalone_en_main);
    return TDateTimeParserBase::GetResult(HttpDateTimeParserStandalone_first_final, defaultValue);
}

TInstant TX509ValidityDateTimeParser::GetResult(TInstant defaultValue) const {
    Y_UNUSED(X509ValidityDateTimeParser_en_main);
    return TDateTimeParserBase::GetResult(X509ValidityDateTimeParser_first_final, defaultValue);
}

TInstant TX509Validity4yDateTimeParser::GetResult(TInstant defaultValue) const {
    Y_UNUSED(X509Validity4yDateTimeParser_en_main);
    return TDateTimeParserBase::GetResult(X509Validity4yDateTimeParser_first_final, defaultValue);
}

template<class TParser, class TResult>
static inline TResult Parse(const char* input, size_t len, TResult defaultValue) {
    TParser parser;
    if (!parser.ParsePart(input, len))
        return defaultValue;
    return parser.GetResult(defaultValue);
}

template<class TParser, class TResult, bool ThrowExceptionOnFailure = true>
static inline TResult ParseUnsafe(const char* input, size_t len) {
    TResult r = Parse<TParser, TResult>(input, len, TResult::Max());
    if (ThrowExceptionOnFailure && r == TResult::Max())
        ythrow TDateTimeParseException() << "error in datetime parsing. Input data: " << TStringBuf(input, len);
    return r;
}

TInstant TInstant::ParseIso8601Deprecated(const TStringBuf input) {
    return ParseUnsafe<TIso8601DateTimeParserDeprecated, TInstant>(input.data(), input.size());
}

TInstant TInstant::ParseRfc822Deprecated(const TStringBuf input) {
    return ParseUnsafe<TRfc822DateTimeParserDeprecated, TInstant>(input.data(), input.size());
}

TInstant TInstant::ParseHttpDeprecated(const TStringBuf input) {
    return ParseUnsafe<THttpDateTimeParserDeprecated, TInstant>(input.data(), input.size());
}

TInstant TInstant::ParseX509ValidityDeprecated(const TStringBuf input) {
    switch (input.size()) {
    case 13:
        return ParseUnsafe<TX509ValidityDateTimeParserDeprecated, TInstant>(input.data(), 13);
    case 15:
        return ParseUnsafe<TX509Validity4yDateTimeParserDeprecated, TInstant>(input.data(), 15);
    default:
        ythrow TDateTimeParseException();
    }
}

bool TInstant::TryParseIso8601Deprecated(const TStringBuf input, TInstant& instant) {
    const auto parsed = ParseUnsafe<TIso8601DateTimeParserDeprecated, TInstant, false>(input.data(), input.size());
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool TInstant::TryParseRfc822Deprecated(const TStringBuf input, TInstant& instant) {
    const auto parsed = ParseUnsafe<TRfc822DateTimeParserDeprecated, TInstant, false>(input.data(), input.size());
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool TInstant::TryParseHttpDeprecated(const TStringBuf input, TInstant& instant) {
    const auto parsed = ParseUnsafe<THttpDateTimeParserDeprecated, TInstant, false>(input.data(), input.size());
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool TInstant::TryParseX509Deprecated(const TStringBuf input, TInstant& instant) {
    TInstant parsed;
    switch (input.size()) {
        case 13:
            parsed = ParseUnsafe<TX509ValidityDateTimeParserDeprecated, TInstant, false>(input.data(), 13);
            break;
        case 15:
            parsed = ParseUnsafe<TX509Validity4yDateTimeParserDeprecated, TInstant, false>(input.data(), 15);
            break;
        default:
            return false;
    }
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

TInstant TInstant::ParseIso8601(const TStringBuf input) {
    return ParseUnsafe<TIso8601DateTimeParser, TInstant>(input.data(), input.size());
}

TInstant TInstant::ParseRfc822(const TStringBuf input) {
    return ParseUnsafe<TRfc822DateTimeParser, TInstant>(input.data(), input.size());
}

TInstant TInstant::ParseHttp(const TStringBuf input) {
    return ParseUnsafe<THttpDateTimeParser, TInstant>(input.data(), input.size());
}

TInstant TInstant::ParseX509Validity(const TStringBuf input) {
    switch (input.size()) {
    case 13:
        return ParseUnsafe<TX509ValidityDateTimeParser, TInstant>(input.data(), 13);
    case 15:
        return ParseUnsafe<TX509Validity4yDateTimeParser, TInstant>(input.data(), 15);
    default:
        ythrow TDateTimeParseException();
    }
}

bool TInstant::TryParseIso8601(const TStringBuf input, TInstant& instant) {
    const auto parsed = ParseUnsafe<TIso8601DateTimeParser, TInstant, false>(input.data(), input.size());
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool TInstant::TryParseRfc822(const TStringBuf input, TInstant& instant) {
    const auto parsed = ParseUnsafe<TRfc822DateTimeParser, TInstant, false>(input.data(), input.size());
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool TInstant::TryParseHttp(const TStringBuf input, TInstant& instant) {
    const auto parsed = ParseUnsafe<THttpDateTimeParser, TInstant, false>(input.data(), input.size());
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool TInstant::TryParseX509(const TStringBuf input, TInstant& instant) {
    TInstant parsed;
    switch (input.size()) {
        case 13:
            parsed = ParseUnsafe<TX509ValidityDateTimeParser, TInstant, false>(input.data(), 13);
            break;
        case 15:
            parsed = ParseUnsafe<TX509Validity4yDateTimeParser, TInstant, false>(input.data(), 15);
            break;
        default:
            return false;
    }
    if (TInstant::Max() == parsed) {
        return false;
    }
    instant = parsed;
    return true;
}

bool ParseRFC822DateTimeDeprecated(const char* input, time_t& utcTime) {
    return ParseRFC822DateTimeDeprecated(input, strlen(input), utcTime);
}

bool ParseISO8601DateTimeDeprecated(const char* input, time_t& utcTime) {
    return ParseISO8601DateTimeDeprecated(input, strlen(input), utcTime);
}

bool ParseHTTPDateTimeDeprecated(const char* input, time_t& utcTime) {
    return ParseHTTPDateTimeDeprecated(input, strlen(input), utcTime);
}

bool ParseX509ValidityDateTimeDeprecated(const char* input, time_t& utcTime) {
    return ParseX509ValidityDateTimeDeprecated(input, strlen(input), utcTime);
}

bool ParseRFC822DateTimeDeprecated(const char* input, size_t inputLen, time_t& utcTime) {
    try {
        utcTime = ParseUnsafe<TRfc822DateTimeParserDeprecated, TInstant>(input, inputLen).TimeT();
        return true;
    } catch (const TDateTimeParseException&) {
        return false;
    }
}

bool ParseISO8601DateTimeDeprecated(const char* input, size_t inputLen, time_t& utcTime) {
    try {
        utcTime = ParseUnsafe<TIso8601DateTimeParserDeprecated, TInstant>(input, inputLen).TimeT();
        return true;
    } catch (const TDateTimeParseException&) {
        return false;
    }
}

bool ParseHTTPDateTimeDeprecated(const char* input, size_t inputLen, time_t& utcTime) {
    try {
        utcTime = ParseUnsafe<THttpDateTimeParserDeprecated, TInstant>(input, inputLen).TimeT();
        return true;
    } catch (const TDateTimeParseException&) {
        return false;
    }
}

bool ParseX509ValidityDateTimeDeprecated(const char* input, size_t inputLen, time_t& utcTime) {
    TInstant r;
    switch (inputLen) {
    case 13:
        r = Parse<TX509ValidityDateTimeParserDeprecated, TInstant>(input, 13, TInstant::Max());
        break;
    case 15:
        r = Parse<TX509Validity4yDateTimeParserDeprecated, TInstant>(input, 15, TInstant::Max());
        break;
    default:
        return false;
    }
    if (r == TInstant::Max())
        return false;
    utcTime = r.TimeT();
    return true;
}

bool ParseRFC822DateTime(const char* input, time_t& utcTime) {
    return ParseRFC822DateTime(input, strlen(input), utcTime);
}

bool ParseISO8601DateTime(const char* input, time_t& utcTime) {
    return ParseISO8601DateTime(input, strlen(input), utcTime);
}

bool ParseHTTPDateTime(const char* input, time_t& utcTime) {
    return ParseHTTPDateTime(input, strlen(input), utcTime);
}

bool ParseX509ValidityDateTime(const char* input, time_t& utcTime) {
    return ParseX509ValidityDateTime(input, strlen(input), utcTime);
}

bool ParseRFC822DateTime(const char* input, size_t inputLen, time_t& utcTime) {
    try {
        utcTime = ParseUnsafe<TRfc822DateTimeParser, TInstant>(input, inputLen).TimeT();
        return true;
    } catch (const TDateTimeParseException&) {
        return false;
    }
}

bool ParseISO8601DateTime(const char* input, size_t inputLen, time_t& utcTime) {
    try {
        utcTime = ParseUnsafe<TIso8601DateTimeParser, TInstant>(input, inputLen).TimeT();
        return true;
    } catch (const TDateTimeParseException&) {
        return false;
    }
}

bool ParseHTTPDateTime(const char* input, size_t inputLen, time_t& utcTime) {
    try {
        utcTime = ParseUnsafe<THttpDateTimeParser, TInstant>(input, inputLen).TimeT();
        return true;
    } catch (const TDateTimeParseException&) {
        return false;
    }
}

bool ParseX509ValidityDateTime(const char* input, size_t inputLen, time_t& utcTime) {
    TInstant r;
    switch (inputLen) {
    case 13:
        r = Parse<TX509ValidityDateTimeParser, TInstant>(input, 13, TInstant::Max());
        break;
    case 15:
        r = Parse<TX509Validity4yDateTimeParser, TInstant>(input, 15, TInstant::Max());
        break;
    default:
        return false;
    }
    if (r == TInstant::Max())
        return false;
    utcTime = r.TimeT();
    return true;
}


/* #line 10051 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
static const int TDurationParser_start = 1;
static const int TDurationParser_first_final = 5;

static const int TDurationParser_en_main = 1;


/* #line 753 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */


TDurationParser::TDurationParser()
    : cs(0)
    , I(0)
    , Dc(0)
    , MultiplierPower(6)
    , Multiplier(1)
    , IntegerPart(0)
    , FractionPart(0)
    , FractionDigits(0)
{
    Y_UNUSED(TDurationParser_en_main);
    
/* #line 10073 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	cs = TDurationParser_start;
	}

/* #line 767 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
}

bool TDurationParser::ParsePart(const char* input, size_t len) {
    const char* p = input;
    const char* pe = input + len;

    
/* #line 10086 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	{
	if ( p == pe )
		goto _test_eof;
	switch ( cs )
	{
case 1:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr0;
	goto st0;
st0:
cs = 0;
	goto _out;
tr0:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 743 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ IntegerPart = I; }
	goto st5;
tr6:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 743 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ IntegerPart = I; }
	goto st5;
st5:
	if ( ++p == pe )
		goto _test_eof5;
case 5:
/* #line 10126 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 46: goto st2;
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr6;
	goto st0;
st2:
	if ( ++p == pe )
		goto _test_eof2;
case 2:
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr2;
	goto st0;
tr2:
/* #line 18 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = 0;
    Dc = 0;
}
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 745 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ FractionPart = I; FractionDigits = Dc; }
	goto st6;
st6:
	if ( ++p == pe )
		goto _test_eof6;
case 6:
/* #line 10165 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr14;
	goto st0;
tr14:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 745 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ FractionPart = I; FractionDigits = Dc; }
	goto st7;
st7:
	if ( ++p == pe )
		goto _test_eof7;
case 7:
/* #line 10191 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr15;
	goto st0;
tr15:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 745 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ FractionPart = I; FractionDigits = Dc; }
	goto st8;
st8:
	if ( ++p == pe )
		goto _test_eof8;
case 8:
/* #line 10217 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr16;
	goto st0;
tr16:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 745 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ FractionPart = I; FractionDigits = Dc; }
	goto st9;
st9:
	if ( ++p == pe )
		goto _test_eof9;
case 9:
/* #line 10243 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr17;
	goto st0;
tr17:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 745 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ FractionPart = I; FractionDigits = Dc; }
	goto st10;
st10:
	if ( ++p == pe )
		goto _test_eof10;
case 10:
/* #line 10269 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto tr18;
	goto st0;
tr18:
/* #line 23 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{
    I = I * 10 + ((*p) - '0');
    ++Dc;
}
/* #line 745 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ FractionPart = I; FractionDigits = Dc; }
	goto st11;
st11:
	if ( ++p == pe )
		goto _test_eof11;
case 11:
/* #line 10295 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	switch( (*p) ) {
		case 100: goto tr7;
		case 104: goto tr8;
		case 109: goto tr9;
		case 110: goto st3;
		case 115: goto tr11;
		case 117: goto st4;
		case 119: goto tr13;
	}
	if ( 48 <= (*p) && (*p) <= 57 )
		goto st11;
	goto st0;
tr3:
/* #line 740 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower = -3; Multiplier = 1; }
	goto st12;
tr4:
/* #line 739 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  0; Multiplier = 1; }
	goto st12;
tr7:
/* #line 734 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  6; Multiplier = 86400; }
	goto st12;
tr8:
/* #line 735 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  6; Multiplier = 3600; }
	goto st12;
tr11:
/* #line 737 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  6; Multiplier = 1; }
	goto st12;
tr13:
/* #line 733 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  6; Multiplier = 604800; }
	goto st12;
tr20:
/* #line 738 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  3; Multiplier = 1; }
	goto st12;
st12:
	if ( ++p == pe )
		goto _test_eof12;
case 12:
/* #line 10340 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	goto st0;
tr9:
/* #line 736 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
	{ MultiplierPower =  6; Multiplier = 60; }
	goto st13;
st13:
	if ( ++p == pe )
		goto _test_eof13;
case 13:
/* #line 10350 "/Users/makar/Documents/Course work/Repository/catboost/build_artifacts/util/parser.rl6.cpp" */
	if ( (*p) == 115 )
		goto tr20;
	goto st0;
st3:
	if ( ++p == pe )
		goto _test_eof3;
case 3:
	if ( (*p) == 115 )
		goto tr3;
	goto st0;
st4:
	if ( ++p == pe )
		goto _test_eof4;
case 4:
	if ( (*p) == 115 )
		goto tr4;
	goto st0;
	}
	_test_eof5: cs = 5; goto _test_eof; 
	_test_eof2: cs = 2; goto _test_eof; 
	_test_eof6: cs = 6; goto _test_eof; 
	_test_eof7: cs = 7; goto _test_eof; 
	_test_eof8: cs = 8; goto _test_eof; 
	_test_eof9: cs = 9; goto _test_eof; 
	_test_eof10: cs = 10; goto _test_eof; 
	_test_eof11: cs = 11; goto _test_eof; 
	_test_eof12: cs = 12; goto _test_eof; 
	_test_eof13: cs = 13; goto _test_eof; 
	_test_eof3: cs = 3; goto _test_eof; 
	_test_eof4: cs = 4; goto _test_eof; 

	_test_eof: {}
	_out: {}
	}

/* #line 774 "/Users/makar/Documents/Course work/Repository/catboost/util/datetime/parser.rl6" */
    return cs != 0;
}

static inline ui64 DecPower(ui64 part, i32 power) {
    if (power >= 0)
        return part * Power(10, power);
    return part / Power(10, -power);
}

TDuration TDurationParser::GetResult(TDuration defaultValue) const {
    if (cs < TDurationParser_first_final)
        return defaultValue;
    ui64 us = 0;
    us += Multiplier * DecPower(IntegerPart, MultiplierPower);
    us += Multiplier * DecPower(FractionPart, MultiplierPower - FractionDigits);
    return TDuration::MicroSeconds(us);
}

bool TDuration::TryParse(const TStringBuf input, TDuration& result) {
    TDuration r = ::Parse<TDurationParser, TDuration>(input.data(), input.size(), TDuration::Max());
    if (r == TDuration::Max())
        return false;
    result = r;
    return true;
}

TDuration TDuration::Parse(const TStringBuf input) {
    return ParseUnsafe<TDurationParser, TDuration>(input.data(), input.size());
}
