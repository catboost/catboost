#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
shortint h_dim(a,b) shortint *a, *b;
#else
shortint h_dim(shortint *a, shortint *b)
#endif
{
return( *a > *b ? *a - *b : 0);
}
#ifdef __cplusplus
}
#endif
