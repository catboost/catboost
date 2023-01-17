class TsError(RuntimeError):
    pass


class TsValidationError(TsError):
    def __init__(self, path, errors):
        self.path = path
        self.errors = errors

        super(TsValidationError, self).__init__("Invalid tsconfig {}:\n{}".format(path, "\n".join(errors)))


class TsCompilationError(TsError):
    def __init__(self, code, stdout, stderr):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

        super(TsCompilationError, self).__init__("tsc exited with code {}:\n{}\n{}".format(code, stdout, stderr))
