#include "f2c.h"

#ifdef KR_headers
double sin(), cos(), sinh(), cosh();
VOID z_cos(r, z) doublecomplex *r, *z;
#else
#undef abs
#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif
void z_cos(doublecomplex *r, doublecomplex *z)
#endif
{
	double zi = z->i, zr = z->r;
	r->r =   cos(zr) * cosh(zi);
	r->i = - sin(zr) * sinh(zi);
	}
#ifdef __cplusplus
}
#endif
