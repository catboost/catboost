#!/usr/bin/env bash

set -e

machines_list=/tmp/machines_list
cores_list=/tmp/cores_list

sky list i+a_kernel_test_dev > $machines_list
> $cores_list
for i in $(cat $machines_list) ; do
    echo "Processing $i"
    echo $i >> $cores_list
    ( ( rsync rsync://$i/coredumps/*.* || true ) 2>&1 ) >> $cores_list
done

/bin/grep -v core_watcher.timestamp $cores_list
