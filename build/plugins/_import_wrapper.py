try:
    from ymake import CustomCommand as RealCustomCommand
    from ymake import addrule
    from ymake import addparser
    from ymake import subst

    class CustomCommand(RealCustomCommand):
        def __init__(self, *args, **kwargs):
            RealCustomCommand.__init__(*args, **kwargs)

        def resolve_path(self, path):
            return subst(path)

except ImportError:
    from _custom_command import CustomCommand  # noqa
    from _custom_command import addrule  # noqa
    from _custom_command import addparser  # noqa


try:
    from ymake import engine_version
except ImportError:
    def engine_version():
        return -1
