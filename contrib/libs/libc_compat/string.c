#include <ctype.h>
#include <string.h>

char* strupr(char* s) {
    char* d;
    for (d = s; *d; ++d)
        *d = (char)toupper((int)*d);
    return s;
}

char* strlwr(char* s) {
    char* d;
    for (d = s; *d; ++d)
        *d = (char)tolower((int)*d);
    return s;
}
