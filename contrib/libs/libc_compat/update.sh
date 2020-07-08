#!/bin/sh -e

unweak() {
    sed --in-place --expression 's/DEF_WEAK(.\+);//g' "$1"
}

curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/strlcpy.c" --output "strlcpy.c" && unweak "strlcpy.c"
curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/strlcat.c" --output "strlcat.c" && unweak "strlcat.c"

curl "https://raw.githubusercontent.com/freebsd/freebsd/master/include/glob.h" --output "glob.h"
curl "https://raw.githubusercontent.com/freebsd/freebsd/master/lib/libc/gen/glob.c" --output "glob.c"
curl "https://raw.githubusercontent.com/freebsd/freebsd/master/lib/libc/stdlib/reallocarray.c" --output "reallocarray.c"
> "collate.h"
> "stdlib.h"
> "unistd.h"
patch -i patches/glob.patch
