#!/usr/bin/env bash
set -euo pipefail

version="${1:-11.0.0}"
major="${version%%.*}"

arcadia="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
tmp="${TMPDIR:-/tmp}/clang"
exe="LLVM-${version}-win64.exe"
dir="clang-${version}-win"
tar="${dir}.tar.gz"

set -x

mkdir -p "$tmp"
cd "$tmp"
test -e "$exe" || wget "https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/${exe}"

rm -rf "$dir"
mkdir -p "$dir"
cd "$dir"
7z x ../"$exe"

"$arcadia"/ya \
    make "$arcadia"/contrib/libs/llvm${major}/tools/{llvm-as,llvm-link,opt} \
    -DNO_DEBUGINFO -r --target-platform=windows --no-src-links -I bin

tar czf "../$tar" *

printf '%q ' ya upload "$tmp/$tar" -d "Clang $version for Windows" --ttl inf --owner BUILD_TOOLS --type CLANG_TOOLKIT --attr platform=win32 --attr "version=$version"
echo
