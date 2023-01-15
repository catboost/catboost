import errno
import os
import shutil
import socket
import socketserver
import subprocess
import string
import tempfile
import time

import catboost as cb

from config import CATBOOST_APP_PATH


def object_list_to_tsv(object_list, file_path):
    with open(file_path, 'w') as f:
        for object_data in object_list:
            f.write('\t'.join(map(str, object_data)) + '\n')


def get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]

def is_port_free(port, sock_type=socket.SOCK_STREAM):
    sock = socket.socket(socket.AF_INET6, sock_type)
    try:
        sock.bind(('::', port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            return False
        raise
    finally:
        sock.close()
    return True


# returns CatBoost object
def run_dist_train(cmd_line_params_list, model_class=cb.CatBoost, worker_count=1):
    hosts_path = tempfile.mkstemp(prefix='catboost_dist_train_hosts_')[1]
    model_path = tempfile.mkstemp(prefix='catboost_model_')[1]
    train_dir = tempfile.mkdtemp(prefix='catboost_train_dir_')
    try:
        ports = []
        workers = []
        for i in range(worker_count):
            ports.append(get_free_port())

        with open(hosts_path, 'w') as hosts:
            for port in ports:
                hosts.write('localhost:' + str(port) + '\n')

        for port in ports:
            workers.append(subprocess.Popen([CATBOOST_APP_PATH, 'run-worker', '--node-port', str(port)]))

        while any([is_port_free(port) for port in ports]):
            time.sleep(1)

        cmd = (
            [CATBOOST_APP_PATH, 'fit']
             + cmd_line_params_list
             + ['--node-type', 'Master',
                '--file-with-hosts', hosts_path,
                '--train-dir', train_dir,
                '--model-file', model_path
            ]
        )
        subprocess.check_call(cmd)

        for worker in workers:
            worker.wait()

        model = model_class()
        model.load_model(model_path)
        return model
    finally:
        os.remove(hosts_path)
        os.remove(model_path)
        shutil.rmtree(train_dir)

# returns CatBoost object
def run_local_train(cmd_line_params_list, model_class=cb.CatBoost):
    model_path = tempfile.mkstemp(prefix='catboost_model_')[1]
    train_dir = tempfile.mkdtemp(prefix='catboost_train_dir_')
    try:
        cmd = (
            [CATBOOST_APP_PATH, 'fit']
             + cmd_line_params_list
             + ['--train-dir', train_dir,
                '--model-file', model_path
            ]
        )
        subprocess.check_call(cmd)

        model = model_class()
        model.load_model(model_path)
        return model
    finally:
        os.remove(model_path)
        shutil.rmtree(train_dir)
