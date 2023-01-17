import os
import sys


class RestartTestException(Exception):
    def __init__(self, *args, **kwargs):
        super(RestartTestException, self).__init__(*args, **kwargs)
        sys.stderr.write("##restart-test##\n")
        sys.stderr.flush()
        os.environ["FORCE_EXIT_TESTSFAILED"] = "1"


class InfrastructureException(Exception):
    def __init__(self, *args, **kwargs):
        super(InfrastructureException, self).__init__(*args, **kwargs)
        sys.stderr.write("##infrastructure-error##\n")
        sys.stderr.flush()
        os.environ["FORCE_EXIT_TESTSFAILED"] = "1"
