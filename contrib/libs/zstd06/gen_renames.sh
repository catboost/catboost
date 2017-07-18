#!/bin/bash

ZSTD_ARCHIVE=${1-./libcontrib-libs-zstd06.a}
nm $ZSTD_ARCHIVE --defined-only -g | egrep '^00' | sed 's/Legacy06_//' | cut -d ' ' -f 3 | awk 'BEGIN{print "#pragma once"}{printf("#define %s Legacy06_%s\n", $1, $1)}' > renames.h

for f in $(find . -name '*.h' | grep -v renames); do
    cat "$f" | grep -v '#include "renames' | awk 'BEGIN{print "#include <contrib/libs/zstd06/renames.h>"}{print}' > "$f.tmp" && grep -q '#include' "$f.tmp" && mv "$f.tmp" "$f" || rm "$f.tmp"
done
