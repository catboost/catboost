def is_cmdfile_arg(arg):
    # type: (str) -> bool
    return arg.startswith('@')


def cmdfile_path(arg):
    # type: (str) -> str
    return arg[1:]


def read_from_command_file(arg):
    # type: (str) -> list[str]
    with open(arg) as afile:
        return afile.read().splitlines()


def skip_markers(args):
    # type: (list[str]) -> list[str]
    res = []
    for arg in args:
        if arg == '--ya-start-command-file' or arg == '--ya-end-command-file':
            continue
        res.append(arg)
    return res


def iter_args(
        args,  # type: list[str]
        ):
    for arg in args:
        if not is_cmdfile_arg(arg):
            if arg == '--ya-start-command-file' or arg == '--ya-end-command-file':
                continue
            yield arg
        else:
            for cmdfile_arg in read_from_command_file(cmdfile_path(arg)):
                yield cmdfile_arg


def get_args(args):
    # type: (list[str]) -> list[str]
    return list(iter_args(args))
