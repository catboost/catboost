#!/usr/bin/env bash

for NAME in Tensor SparseTensor Schema Message File
do
	curl https://raw.githubusercontent.com/apache/arrow/master/format/$NAME.fbs -o $NAME.fbs
done

planus --ignore-docstring-errors rust *.fbs -o src/ipc/generated.rs
