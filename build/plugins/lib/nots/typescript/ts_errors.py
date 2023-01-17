class TsError(RuntimeError):
    pass


class TsValidationError(TsError):
    def __init__(self, path, errors):
        self.path = path
        self.errors = errors

        super(TsValidationError, self).__init__("Invalid tsconfig {}:\n{}".format(path, "\n".join(errors)))
