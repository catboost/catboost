#include <locale.h>
#include <time.h>

size_t strftime_l(char *restrict s, size_t n, const char *restrict f, const struct tm *restrict tm, locale_t l)
{
	return strftime(s, n, f, tm);
}
