#!/usr/bin/env python3.8

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == 'cli':
    if len(sys.argv) > 2 and sys.argv[2] == 'fit':
        args = ['fit', '--task-type', 'GPU'] + sys.argv[3:]
    else:
        args =  sys.argv[2:]
    os.execvp('/catboost/catboost', ['cli'] + args)
else:
    print('Starting jupyter notebook on 8888 port')
    os.execvp('jupyter-notebook', ['jupyter-notebook', '--port', '8888', '--ip', '0.0.0.0', '--allow-root', '--NotebookApp.token', os.environ.get('JUPYTER_TOKEN', "")])
