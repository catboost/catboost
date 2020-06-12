#!/bin/sh -e

unweak() {
    sed --in-place --expression 's/DEF_WEAK(.\+);//g' "$1"
}

curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/strlcpy.c" --output "strlcpy.c" && unweak "strlcpy.c"
curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/strlcat.c" --output "strlcat.c" && unweak "strlcat.c"
