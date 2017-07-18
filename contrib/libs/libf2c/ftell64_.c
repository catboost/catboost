#include "f2c.h"
#include "fio.h"
#ifdef __cplusplus
extern "C" {
#endif

 static FILE *
#ifdef KR_headers
unit_chk(Unit, who) integer Unit; char *who;
#else
unit_chk(integer Unit, char *who)
#endif
{
	if (Unit >= MXUNIT || Unit < 0)
		f__fatal(101, who);
	return f__units[Unit].ufd;
	}

 longint
#ifdef KR_headers
ftell64_(Unit) integer *Unit;
#else
ftell64_(integer *Unit)
#endif
{
	FILE *f;
	return (f = unit_chk(*Unit, "ftell")) ? FTELL(f) : -1L;
	}

 int
#ifdef KR_headers
fseek64_(Unit, offset, whence) integer *Unit, *whence; longint *offset;
#else
fseek64_(integer *Unit, longint *offset, integer *whence)
#endif
{
	FILE *f;
	int w = (int)*whence;
#ifdef SEEK_SET
	static int wohin[3] = { SEEK_SET, SEEK_CUR, SEEK_END };
#endif
	if (w < 0 || w > 2)
		w = 0;
#ifdef SEEK_SET
	w = wohin[w];
#endif
	return	!(f = unit_chk(*Unit, "fseek"))
		|| FSEEK(f, (OFF_T)*offset, w) ? 1 : 0;
	}
#ifdef __cplusplus
}
#endif
