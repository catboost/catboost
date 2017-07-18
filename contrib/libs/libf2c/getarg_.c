#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

/*
 * subroutine getarg(k, c)
 * returns the kth unix command argument in fortran character
 * variable argument c
*/

#ifdef KR_headers
VOID getarg_(n, s, ls) ftnint *n; char *s; ftnlen ls;
#define Const /*nothing*/
#else
#define Const const
void getarg_(ftnint *n, char *s, ftnlen ls)
#endif
{
	extern int xargc;
	extern char **xargv;
	Const char *t;
	int i;
	
	if(*n>=0 && *n<xargc)
		t = xargv[*n];
	else
		t = "";
	for(i = 0; i<ls && *t!='\0' ; ++i)
		*s++ = *t++;
	for( ; i<ls ; ++i)
		*s++ = ' ';
	}
#ifdef __cplusplus
}
#endif
