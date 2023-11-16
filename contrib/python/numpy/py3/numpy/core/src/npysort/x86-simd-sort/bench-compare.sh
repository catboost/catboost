#~/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
branch=$(git rev-parse --abbrev-ref HEAD)
echo "Comparing main branch with $branch"

if [[ -z "${GBENCH}" ]]; then
    echo "Please set env variable GBENCH and re run"
    exit 1
fi

compare=$GBENCH/tools/compare.py
if [ ! -f $compare ]; then
    echo "Unable to locate $GBENCH/tools/compare.py"
    exit 1
fi

rm -rf .bench-compare
mkdir .bench-compare
cd .bench-compare
echo "Fetching and build $branch .."
git clone ${SCRIPT_DIR} -b $branch .
git fetch origin
meson setup --warnlevel 0 --buildtype plain builddir-${branch}
cd builddir-${branch}
ninja
echo "Fetching and build main .."
cd ..
git remote add upstream https://github.com/intel/x86-simd-sort.git
git fetch upstream
git checkout upstream/main
meson setup --warnlevel 0 --buildtype plain builddir-main
cd builddir-main
ninja
cd ..
echo "Running benchmarks .."
$compare benchmarks ./builddir-main/benchexe ./builddir-${branch}/benchexe
