#include <string.h>
#include <locale.h>
#include "libc.h"

// ANDROID: was __strcoll_l
int strcoll_l(const char *l, const char *r, locale_t loc)
{
	return strcmp(l, r);
}

int strcoll(const char *l, const char *r)
{
	return strcoll_l(l, r, 0);
}

weak_alias(__strcoll_l, strcoll_l);
