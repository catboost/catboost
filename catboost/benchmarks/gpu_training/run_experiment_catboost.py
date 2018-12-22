#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Run eval on nirvana"""
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

cb_cuda_path = "./cb_cuda"
catboost_path = "./catboost"

fit_template = " fit --auto-stop-pval 1.0 --overfitting-detector-iterations-wait 500 --use-best-model"

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

    parser.add_argument('--skip-up-tasks', action='store_true')
    parser.add_argument('--skip-down-tasks', action='store_true')

    args = parser.parse_args(sys.argv[1:])
    pool_dir = args.pool_dir


    def set_matrixnet_mode(cmd):
        return cmd + " --dev-disable-dontlookahead -p1"

    def get_features_abs_path():
        return os.path.join(pool_dir, "train_full3")


    def get_test_abs_path():
        return os.path.join(pool_dir, "test3")


    def get_cd_abs_path():
        return os.path.join(pool_dir, "train_full3.cd")

    def set_pool_path(cmd):
        result = cmd + " -f {__features} -t {__test} --cd {__cd}".format(__features=get_features_abs_path(), __test=get_test_abs_path(),__cd=get_cd_abs_path())
        if args.read_pool_args:
            result = result + " " + args.read_pool_args.sp
        return result


    def set_params(cmd, step, iters, target, complexity=1):
        result = cmd + " -i {__iters} -w {__step}  --loss-function {__target} --max-ctr-complexity={__complexity}".format(__iters=iters, __step=step, __target=target, __complexity=complexity)
        if args.other_args is not None:
            result = result + " " + args.other_args
        return result

    def update_plot_log(logs_dir, step, total_iters, output):
        scores = pd.read_csv(os.path.join(logs_dir, "test_metric.tsv"), sep="\t")
        score_values = scores.iloc[:, 1]
        if 'RMSE' in "".join(scores.columns):
            best = min(score_values)
        else:
            best = max(score_values)
        best_iter = int(scores.iloc[int(max(scores.index[score_values == best])),0])
        time_to_best = pd.read_csv(os.path.join(logs_dir, "time_left.tsv"), header=None, sep="\t").iloc[
            best_iter - 1, 2]
        with open(output, 'a+') as outfile:
            outfile.write("{__iter}\t{__score}\t{__step}\t{__time}\t{__total}\tcatboost\n".format(__iter=best_iter,
                                                                                        __score=best,
                                                                                        __step=step,
                                                                                        __time=time_to_best,
                                                                                        __total=total_iters))

    def compute_metric(metric, dir, plot_step, test_path, cd_path):
        cmd = catboost_path + " plot --input-path {__test_path} --cd {__cd_path} -T 16 " \
                              "-m {__dir}/catboost.bin -o test_metric.tsv --eval-metric {__metric} --verbose --step {__step}".format(__test_path=test_path,
                                                                                                                                     __cd_path=cd_path,
                                                                                                                                     __dir=dir,
                                                                                                                                     __metric=metric,
                                                                                                                                     __step=plot_step)
        print("Running command: {}".format(cmd))
        return subprocess.call(cmd.split(' '), cwd=dir)



    def compute_metric_relative(metric,
                                dir,
                                plot_step,
                                test_path=get_test_abs_path(),
                                cd_path=get_cd_abs_path()):
        return compute_metric(metric, os.path.join(os.getcwd(), dir), plot_step, test_path, cd_path)


    def run_on_device(args, deviceId, task_dir):
        ensure_dir_exists(task_dir)
        cmd = args.split(' ')
        print("Running command: {}".format(cmd))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(deviceId)
        # cmd = ['bash', '-o', 'pipefail', '-uec', cmd]
        full_path = os.path.join(os.getcwd(), task_dir)
        stderr = open(os.path.join(task_dir, "matrixnet_stderr.log"), "w")
        stdout = open(os.path.join(task_dir, "matrixnet_stdout.log"), "w")
        return subprocess.call(cmd, stdout=stdout, stderr=stderr, env=env, cwd=full_path)


    def run_on_device_relative(args, deviceId, task_dir):
        return run_on_device(args, deviceId, os.path.join(os.getcwd(), task_dir))


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

    def get_plot_step(iter):
        if iter > 2500:
            return 25
        elif iter > 500:
            return 10
        elif iter > 250:
            return 2
        else:
            return 1

    add_up = not args.skip_up_tasks
    add_down = not args.skip_down_tasks
    tasks = generate_tasks(args.base_step, args.base_iter, args.mult, args.count, add_up, add_down)
    #fit
    for task in tasks:
        cmd = set_pool_path(cb_cuda_path + fit_template)
        step_ = task["step"]
        iter_ = task["iter"]
        cmd = set_params(cmd, step_, iter_, args.target)
        task_dir = get_dir_for_step(step_)
        run_on_device_relative(cmd, args.device, task_dir)
    #compute
    for task in tasks:
        cmd = set_pool_path(cb_cuda_path + fit_template)
        step_ = task["step"]
        iter_ = task["iter"]
        task_dir = get_dir_for_step(step_)
        compute_metric_relative(args.test_metric,
                                task_dir,
                                get_plot_step(iter_)
                                )
        update_plot_log(task_dir, step_, iter_, "scores_plot.tsv")

        #
