#!/usr/bin/env bash

./catboost fit \
	--use-best-model false \
	--loss-function Cox \
	-f ~/catboost/catboost/pytest/data/veteran/train \
	-t ~/catboost/catboost/pytest/data/veteran/test  \
	--column-description ~/catboost/catboost/pytest/data/veteran/train.cd \
	--boosting-type Plain \
	-i 100 \
	-T 2 \
	--bootstrap-type No \
	--random-strength 0 \
	-m mopl.bin \
	--eval-file test.eval \