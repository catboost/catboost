#!/usr/bin/env bash

python="$(which python)"

echo "Testing minidump2core..."
for i in data/md*.txt ; do
    echo "    Processing '$i' with minidump2core"
    $python ../minidump2core/src/minidump2core/minidump2core.py $i > $i.txtcore
    line_count="$(cat $i.txtcore | wc -l)"
    if [ "$line_count" -gt 5000 ] || [ "$line_count" -lt 10 ] ; then
        echo "Suspicious stacktrace generated ($line_count lines), test failed"
        exit 1
    fi
    echo "    Processing generated '$i.txtcore' with core_proc"
    $python ../core_proc.py $i.txtcore > $i.html
done

echo "Testing minidump2core as module..."
cd .. && rm -f *.pyc && PYTHONPATH=minidump2core/src/:$PYTHONPATH $python ./minidump2core/tests/test_minidump2core.py

echo "All tests passed OK"
