#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = "noxoomo"
__email__ = "noxoomo@yandex-team.ru"

import json
import subprocess
import os
import sys
from subprocess import Popen
import subprocess
import argparse
import os
import os.path
import pandas as pd
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--metric',
                        required=True)
    parser.add_argument('--output',
                        required=True)

    args = parser.parse_args(sys.argv[1:])

    def get_xgboost_times(log_path):
        elapsed_times = []
        with open(log_path, 'r') as input:
            for line in input.readlines():
                if 'boosting round' in line and 'sec elapsed' in line:
                    elapsed_times.append(float(line.split(',')[1].strip().split(' ')[0]) * 1000)
        return np.array(elapsed_times)

    def get_metric_from_xgboost(log_path, metric):
        scores = []
        with open(log_path, 'r') as input:
            for line in input.readlines():
                if 'test-' in line and metric in line:
                    scores.append(float(line.split(metric+':')[1].split('\t')[0]))
        return scores


    def get_xgboost_times(log_path):
        elapsed_times = []
        with open(log_path, 'r') as input:
            for line in input.readlines():
                if 'boosting round' in line and 'sec elapsed' in line:
                    elapsed_times.append(float(line.split(',')[1].strip().split(' ')[0]) * 1000)
        return np.array(elapsed_times)



    quality_log_suffix = ".quality.log"
    speed_log_suffix = ".speed.log"
    times = get_xgboost_times("xgboost_stderr" + speed_log_suffix)
    scores = get_metric_from_xgboost("xgboost_stderr" + quality_log_suffix,args.metric)[0:len(times)]
    iters = [i for i in range(0, len(scores))]
    method =["XGBoost"] * len(iters)
    dt = pd.DataFrame({"Iteration" : iters, "Time" : times, "Score" : scores, "Method" : method})
    dt.to_csv(args.output, sep='\t',index=False)



