#include <wctype.h>
#include "libc.h"

int towlower_l(int c, locale_t l)
{
	return towlower(c);
}

weak_alias(towlower_l, __towlower_l);
