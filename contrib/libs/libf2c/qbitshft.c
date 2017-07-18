#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

 longint
#ifdef KR_headers
qbit_shift(a, b) longint a; integer b;
#else
qbit_shift(longint a, integer b)
#endif
{
	return b >= 0 ? a << b : (longint)((ulongint)a >> -b);
	}
#ifdef __cplusplus
}
#endif
