import sys

import library.python.resource as lpr


def parse_v():
    for ll in lpr.find('/GrpcUtil.java').decode('utf-8').split('\n'):
        if 'CURRENT_GRPC_VERSION' in ll:
            return ll.split('"')[1]


assert sys.argv[1] == parse_v(), 'version mismatch'
