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

    def get_lightgbm_times(log_path):
        elapsed_times = []
        with open(log_path, 'r') as input:
            for line in input.readlines():
                if 'seconds elapsed, finished' in line:
                    elapsed_times.append(float(line.split('seconds')[0].strip().split(' ')[-1]) * 1000)
        return np.array(elapsed_times)

    def estimate_fit_time(log_path, iter):
        times = get_lightgbm_times(log_path)
        return times[iter]

    def get_metric_from_lightgbm(log_path, metric):
        scores = []
        metric_name = "l2" if metric == "mse" else metric
        with open(log_path, 'r') as input:
            for line in input.readlines():
                if "Info" in line and 'valid_1' in line and metric_name in line:
                    score=  float(line.split("valid_1 "+ metric_name+' :')[-1])
                    scores.append(score)
        scores = np.array(scores)
        if metric == "mse":
            return scores ** 0.5 #we need rmse, lightgbm computes l2
        else:
            return scores


    def is_max_optimal(metric):
        return "mse" not in metric

    def get_best_score(score_values, metric):
        if is_max_optimal(metric):
            best = max(score_values)
        else:
            best = min(score_values)
        return best

    def get_best_iter(scores, metric):
        if is_max_optimal(metric):
            return np.argmax(scores)
        else:
            return np.argmin(scores)

    def get_best_iter_from_log(dir, metric, suffix):
        scores = get_metric_from_lightgbm(os.path.join(dir, "lightgbm_stdout" + suffix), metric)
        return get_best_iter(scores, metric)

    def get_best_score_from_log(dir, metric, suffix):
        scores = get_metric_from_lightgbm(os.path.join(dir, "lightgbm_stdout" + suffix), metric)
        return get_best_score(scores, metric)

    def get_fit_time_from_log(dir, iter, suffix):
        return estimate_fit_time(os.path.join(dir, "lightgbm_stdout" + suffix), iter)


    quality_log_suffix = ".quality.log"
    speed_log_suffix = ".speed.log"
    times = get_lightgbm_times("lightgbm_stdout" + speed_log_suffix)
    scores = get_metric_from_lightgbm("lightgbm_stdout" + quality_log_suffix, args.metric)[0:len(times)]
    iters = [i for i in range(0, len(scores))]
    method =["LightGBM"] * len(iters)
    dt = pd.DataFrame({"Iteration" : iters, "Time" : times, "Score" : scores, "Method" : method})
    dt.to_csv(args.output, sep='\t',index=False)

