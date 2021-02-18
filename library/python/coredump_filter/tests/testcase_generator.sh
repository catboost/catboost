#!/usr/bin/env bash

python="$(which python)"
cproc="$python ../core_proc.py"

echo "Generate ethalond data for testcases..."
rm data/*.html data/*.fp

$cproc || [ "$?" == "1" ]

# Test generic functionality
$cproc --version
$cproc -v

for i in data/test*.txt ; do
    echo "    Processing '$i' with core_proc"
    $cproc $i > $i.html
    # test fingerprints
    $cproc $i -f > $i.fp
done

echo "Done"
