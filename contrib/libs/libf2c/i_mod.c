#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
integer i_mod(a,b) integer *a, *b;
#else
integer i_mod(integer *a, integer *b)
#endif
{
return( *a % *b);
}
#ifdef __cplusplus
}
#endif
