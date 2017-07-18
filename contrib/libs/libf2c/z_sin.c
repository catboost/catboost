#include "f2c.h"

#ifdef KR_headers
double sin(), cos(), sinh(), cosh();
VOID z_sin(r, z) doublecomplex *r, *z;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
void z_sin(doublecomplex *r, doublecomplex *z)
#endif
{
	double zi = z->i, zr = z->r;
	r->r = sin(zr) * cosh(zi);
	r->i = cos(zr) * sinh(zi);
	}
#ifdef __cplusplus
}
#endif
