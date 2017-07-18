#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
double d_imag(z) doublecomplex *z;
#else
double d_imag(doublecomplex *z)
#endif
{
return(z->i);
}
#ifdef __cplusplus
}
#endif
