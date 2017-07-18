#include <wctype.h>
#include "libc.h"

int towupper_l(int c, locale_t l)
{
	return towupper(c);
}

weak_alias(towupper_l, __towupper_l);
