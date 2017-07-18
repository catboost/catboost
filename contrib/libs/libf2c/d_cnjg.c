#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

 VOID
#ifdef KR_headers
d_cnjg(r, z) doublecomplex *r, *z;
#else
d_cnjg(doublecomplex *r, doublecomplex *z)
#endif
{
	doublereal zi = z->i;
	r->r = z->r;
	r->i = -zi;
	}
#ifdef __cplusplus
}
#endif
