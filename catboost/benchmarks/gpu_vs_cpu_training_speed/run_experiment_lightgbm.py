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

lightgbm_path = "./lightgbm"

fit_template = " sparse_threshold=1.0 num_threads=16 device=gpu early_stopping_rounds=300"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--base-step',
                        type=float,
                        required=True)
    parser.add_argument('--base-iter',
                        type=float,
                        required=True)
    parser.add_argument('--mult',
                        type=float,
                        required=True)
    parser.add_argument('--count',
                        type=int,
                        required=True)
    parser.add_argument('--device',
                        required=True)
    parser.add_argument('--target',
                        required=True)
    parser.add_argument('--test-metric',
                        required=True)
    parser.add_argument('--pool-dir',
                        required=True)
    parser.add_argument('--read_pool_args')
    parser.add_argument('--other-args')

    parser.add_argument('--bins',type=int,default=15)
    parser.add_argument('--leaves',type=int,default=64)

    parser.add_argument('--skip-up-tasks', action='store_true')
    parser.add_argument('--skip-down-tasks', action='store_true')

    args = parser.parse_args(sys.argv[1:])
    pool_dir = args.pool_dir

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

    def get_features_abs_path():
        return os.path.join(pool_dir, "features.train")


    def get_test_abs_path():
        return os.path.join(pool_dir, "features.test")


    def set_pool_path(cmd, metric=None):
        result = cmd + " data={__features}".format(__features=get_features_abs_path())
        if metric is not None:
            result = result + " valid={__test} metric={__metric}".format(__test=get_test_abs_path(), __metric=metric)
        if args.read_pool_args:
            result = result + " " + args.read_pool_args.sp
        return result


    def get_algo_name():
        return "lightgbm_{__bins}_{__leaves}".format(__bins=args.bins, __leaves=args.leaves)


    def set_tree_params(cmd, leaves, bins):
        return cmd + " max_bin={__bins} num_leaves={__leaves}".format(__bins=bins, __leaves=leaves)

    def set_params(cmd, step, iters, target, leaves, bins):
        result = cmd + " learning_rate={__step} num_iterations={__iters} objective={__target}".format(__iters=iters, __step=step, __target=target)
        if args.other_args is not None:
            result = result + " " + args.other_args

        return set_tree_params(result, leaves, bins)

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

    def update_plot_log(step,  iter, best_score, best_iter, time, output):
        with open(output, 'a+') as outfile:
            outfile.write("{__iter}\t{__score}\t{__step}\t{__time}\t{__total}\t{__name}\n".format(__iter=best_iter,
                                                                                                  __score=best_score,
                                                                                                  __step=step,
                                                                                                  __time=time,
                                                                                                  __name=get_algo_name(),
                                                                                                  __total=iter))

    def run_on_device(args, device_id, task_dir, log_suffix):
        ensure_dir_exists(task_dir)
        cmd = args.split(' ')
        print("Running command: {}\n".format(" ".join(cmd)))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(device_id)
        full_path = os.path.join(os.getcwd(), task_dir)

        stderr = open(os.path.join(task_dir, "lightgbm_stderr" + log_suffix), "w")
        stdout = open(os.path.join(task_dir, "lightgbm_stdout" + log_suffix), "w")
        return subprocess.call(cmd, stdout=stdout, stderr=stderr, env=env, cwd=full_path)


    def run_on_device_relative(args, device_id, task_dir, log_suffix):
        return run_on_device(args, device_id, os.path.join(os.getcwd(), task_dir), log_suffix)


    def ensure_dir_exists(dir):
        if not os.path.exists(dir):
            os.mkdir(dir)


    def get_steps(str):
        return str.split(",")


    def get_dir_for_step(step):
        return "step_{__step}".format(__step=step)


    def generate_tasks(base_step, base_iter, mult, count, add_up, add_down):
        tasks = []
        for i in range(0, count):
            if add_up:
                tasks.append({"step": base_step * (mult ** i), "iter": int(base_iter / (mult ** i))})
            if add_down:
                tasks.append({"step": base_step / (mult ** i), "iter": int(base_iter * (mult ** i))})
        return tasks


    add_up = not args.skip_up_tasks
    add_down = not args.skip_down_tasks
    tasks = generate_tasks(args.base_step, args.base_iter, args.mult, args.count, add_up, add_down)
    #fit
    for task in tasks:
        step_ = task["step"]
        iter_ = task["iter"]
        task_dir = get_dir_for_step(step_)
        base_cmd = lightgbm_path + fit_template
        if args.other_args:
            base_cmd = base_cmd + " " + args.other_args
        run_with_quality_calc = set_pool_path(base_cmd, args.test_metric)
        run_with_quality_calc = set_params(run_with_quality_calc, step_, iter_, args.target, args.leaves, args.bins)
        quality_log_suffix = ".quality.log"
        speed_log_suffix = ".speed.log"
        run_on_device_relative(run_with_quality_calc, args.device, task_dir, quality_log_suffix)
        best_score = get_best_score_from_log(task_dir, args.test_metric, quality_log_suffix)
        best_iter = int(get_best_iter_from_log(task_dir, args.test_metric, quality_log_suffix))
        run_for_speed = set_pool_path(base_cmd)
        run_for_speed = set_params(run_for_speed, step_, best_iter + 1, args.target, args.leaves, args.bins)
        run_on_device_relative(run_for_speed, args.device, task_dir, speed_log_suffix)
        best_time = get_fit_time_from_log(task_dir, best_iter, speed_log_suffix)
        update_plot_log(step_, iter_, best_score, best_iter, best_time, "scores_plot.tsv")

