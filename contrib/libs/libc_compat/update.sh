#!/bin/sh -e

unweak() {
    sed --in-place --expression 's/DEF_WEAK(.\+);//g' "$1"
}

curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/strlcpy.c" --output "strlcpy.c" && unweak "strlcpy.c"
curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/strlcat.c" --output "strlcat.c" && unweak "strlcat.c"
mkdir -p include/windows/sys
curl "https://raw.githubusercontent.com/openbsd/src/master/sys/sys/queue.h" --output "include/windows/sys/queue.h"
patch -p1 -i patches/sys-queue.patch

curl "https://raw.githubusercontent.com/freebsd/freebsd/master/include/glob.h" --output "glob.h"
curl "https://raw.githubusercontent.com/freebsd/freebsd/master/lib/libc/gen/glob.c" --output "glob.c"
curl "https://raw.githubusercontent.com/freebsd/freebsd/master/lib/libc/stdlib/reallocarray.c" --output "reallocarray.c"
> "collate.h"
> "stdlib.h"
> "unistd.h"
patch -i patches/glob.patch

mkdir -p include/uchar
curl "https://git.musl-libc.org/cgit/musl/plain/include/uchar.h" --output "include/uchar/uchar.h"
patch -p3 -i patches/uchar.patch
# TODO: provide c16rtomb, mbrtoc16, c32rtomb, mbrtoc32 implementations for uchar
# if any code actually needs them
