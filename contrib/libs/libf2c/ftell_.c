#include "f2c.h"
#include "fio.h"
#ifdef __cplusplus
extern "C" {
#endif

 static FILE *
#ifdef KR_headers
unit_chk(Unit, who) integer Unit; char *who;
#else
unit_chk(integer Unit, const char *who)
#endif
{
	if (Unit >= MXUNIT || Unit < 0)
		f__fatal(101, who);
	return f__units[Unit].ufd;
	}

 integer
#ifdef KR_headers
ftell_(Unit) integer *Unit;
#else
ftell_(integer *Unit)
#endif
{
	FILE *f;
	return (f = unit_chk(*Unit, "ftell")) ? ftell(f) : -1L;
	}

 int
#ifdef KR_headers
fseek_(Unit, offset, whence) integer *Unit, *offset, *whence;
#else
fseek_(integer *Unit, integer *offset, integer *whence)
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
		|| fseek(f, *offset, w) ? 1 : 0;
	}
#ifdef __cplusplus
}
#endif
