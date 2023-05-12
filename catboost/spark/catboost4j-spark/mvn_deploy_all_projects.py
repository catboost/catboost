#!/usr/bin/env python
#
# All additional arguments passed to this script are forwarded to the `mvn deploy` commands for each project
#

import os
import subprocess
import sys


current_dir = os.path.dirname(os.path.realpath(__file__))
mvn_deploy_args = ' '.join(sys.argv[1:])

projects_dir = os.path.join(current_dir, 'projects')

commands = [
    'cd ' + current_dir + ' && ./generate_projects/generate.py'
]
commands += [
    'cd ' + os.path.join(projects_dir, mvn_project_subdir) + ' && mvn deploy ' + mvn_deploy_args
    for mvn_project_subdir in [
        'spark_2.3_2.11',
        os.path.join('spark_2.4_2.11', 'core'),
        'spark_2.4_2.12',
        os.path.join('spark_3.0_2.12', 'core'),
        os.path.join('spark_3.1_2.12', 'core'),
        os.path.join('spark_3.2_2.12', 'core'),
        'spark_3.2_2.13',
        os.path.join('spark_3.3_2.12', 'core'),
        os.path.join('spark_3.3_2.13', 'core'),
        os.path.join('spark_3.4_2.12', 'core'),
        os.path.join('spark_3.4_2.13', 'core'),
    ]
]


for command in commands:
    print ('Executing ', command)
    subprocess.check_call(command, shell=True)
