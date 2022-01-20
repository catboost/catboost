#!/bin/sh -e

unweak() {
    sed --in-place --expression 's/DEF_WEAK(.\+);//g' "$1"
}

get_string_method() {
   curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/string/$1" --output "$1" && unweak "$1"
}

fix_tabs() {
	sed --in-place --expression 's/\t/    /g' "$1"
}

fix_decls() {
	sed --in-place --expression 's/__BEGIN_DECLS/#ifdef __cplusplus\nextern "C" {\n#endif/g' "$1"
	sed --in-place --expression 's/__END_DECLS/#ifdef __cplusplus\n} \/\/ extern "C"\n#endif/g' "$1"
}

get_string_method "strlcpy.c"
get_string_method "strlcat.c"
get_string_method "strsep.c"
# strcasestr uses strncasecmp, which is platform dependent, so include local string.h
get_string_method "strcasestr.c" && sed --in-place 's/#include <string.h>/#include "string.h"/g' "strcasestr.c"
get_string_method "memrchr.c"
get_string_method "stpcpy.c"

mkdir -p include/windows/sys
curl "https://raw.githubusercontent.com/openbsd/src/master/sys/sys/queue.h" --output "include/windows/sys/queue.h"

mkdir -p include/readpassphrase
curl "https://raw.githubusercontent.com/openbsd/src/master/include/readpassphrase.h" --output "include/readpassphrase/readpassphrase.h" && fix_decls "include/readpassphrase/readpassphrase.h"
curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/gen/readpassphrase.c" --output "readpassphrase.c" && unweak "readpassphrase.c" && fix_tabs "readpassphrase.c"

curl "https://raw.githubusercontent.com/freebsd/freebsd/master/include/glob.h" --output "glob.h"
curl "https://raw.githubusercontent.com/freebsd/freebsd/master/lib/libc/gen/glob.c" --output "glob.c"
curl "https://raw.githubusercontent.com/openbsd/src/master/lib/libc/stdlib/reallocarray.c" --output "reallocarray.c" && unweak "reallocarray.c"
> "collate.h"
> "stdlib.h"
> "unistd.h"

mkdir -p include/uchar
curl "https://git.musl-libc.org/cgit/musl/plain/include/uchar.h" --output "include/uchar/uchar.h"
# TODO: provide c16rtomb, mbrtoc16, c32rtomb, mbrtoc32 implementations for uchar
# if any code actually needs them

mkdir -p include/random/sys
curl "https://git.musl-libc.org/cgit/musl/plain/include/sys/random.h" --output "include/random/sys/random.h"
curl "https://git.musl-libc.org/cgit/musl/plain/src/linux/getrandom.c" --output "getrandom.c"
curl "https://git.musl-libc.org/cgit/musl/plain/src/linux/memfd_create.c" --output "memfd_create.c"

# WARN: do not use github.com/morristech/android-ifaddrs, it is a long-ago abandoned fork
curl "https://raw.githubusercontent.com/oliviertilmans/android-ifaddrs/master/ifaddrs.c" --output "ifaddrs.c"
curl "https://raw.githubusercontent.com/oliviertilmans/android-ifaddrs/master/ifaddrs.h" --output "include/ifaddrs/ifaddrs.h"

# apply patches if necessary
for patch in patches/*.patch; do
	echo "Applying patch from $patch"
	patch -p1 < $patch
done
