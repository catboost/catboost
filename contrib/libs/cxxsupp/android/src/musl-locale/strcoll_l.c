#include <string.h>
#include <locale.h>

int strcoll_l(const char *l, const char *r, locale_t loc)
{
	return strcoll(l, r);
}
