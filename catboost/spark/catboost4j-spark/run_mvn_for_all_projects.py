#!/usr/bin/env python
#
# All additional arguments passed to this script are forwarded to the `mvn` commands for each project
#

import os
import subprocess
import sys


current_dir = os.path.dirname(os.path.realpath(__file__))
mvn_args = ' '.join(sys.argv[1:])

sys.path = [os.path.join(current_dir, 'generate_projects')] + sys.path

import generate

projects_dir = os.path.join(current_dir, 'projects')

commands = [
    'cd ' + current_dir + ' && ./generate_projects/generate.py'
]

for config in generate.configs:
    project_dir=os.path.realpath(os.path.join(current_dir, 'generate_projects', config['dst_dir']))
    commands.append('cd ' + project_dir + ' && mvn ' + mvn_args)

for command in commands:
    print ('Executing ', command)
    subprocess.check_call(command, shell=True)
