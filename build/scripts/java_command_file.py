import platform
import subprocess as sp


def _java_cmd_file_quote(s):
    """ Wrap argument based on https://docs.oracle.com/en/java/javase/21/docs/specs/man/java.html#java-command-line-argument-files """
    if not s:
        return "''"

    if not any(char.isspace() for char in s):
        return s

    return f'"{s.replace('\\', '\\\\')}"'


def call_java_with_command_file(cmd, wrapped_args, **kwargs):
    is_win = platform.system() == 'Windows'

    args = cmd
    args_to_wrap = wrapped_args
    if is_win:
        args = [cmd[0]]
        args_to_wrap = cmd[1:] + args_to_wrap

    commands_file = 'wrapped.args'
    with open(commands_file, 'w') as f:
        f.write(' '.join(_java_cmd_file_quote(arg) for arg in args_to_wrap))

    if is_win:
        # Some Windows machines has troubles with running cmd lines with `@` without shell=True
        kwargs['shell'] = True

    try:
        return sp.check_output(
            args + ["@" + commands_file],
            **kwargs
        )
    except Exception as e:
        if hasattr(e, "add_note"):
            e.add_note(f"Original command: {cmd} {wrapped_args}")
            e.add_note(f"Wrapped part: {wrapped_args}")

        raise
