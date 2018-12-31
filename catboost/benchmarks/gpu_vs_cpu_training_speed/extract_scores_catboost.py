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

    parser.add_argument('--output',
                        required=True)

    args = parser.parse_args(sys.argv[1:])


    scores = pd.read_csv("test_metric.tsv", sep="\t")
    times= pd.read_csv("time_left.tsv", sep="\t")

    iters = pd.Series([i for i in range(0, times.shape[0])])
    method =["CatBoost"] * scores.shape[0]


    dt = pd.DataFrame({"Iteration" : np.array(iters[scores.iloc[:,0]]), "Time" : np.array(times.iloc[scores.iloc[:,0],2]), "Score" : np.array(scores.iloc[:,1]), "Method" : method})
    dt.to_csv(args.output, sep='\t',index=False)

