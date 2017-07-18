#include <string.h>

size_t strxfrm_l(char *restrict dest, const char *restrict src, size_t n, locale_t l)
{
	return strxfrm(dest, src, n);
}
