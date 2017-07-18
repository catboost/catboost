#!/bin/sh

nm ./libcontrib-libs-zstd.a | grep -v 'U ' | grep -v '\.L' | grep -v 'Legacy_' | grep 00 | awk '{print $3}' | ./gen.py > renames.h
