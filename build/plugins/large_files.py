import os
import ymake
from _common import strip_roots

PLACEHOLDER_EXT = "external"


def onlarge_files(unit, *args):
    """
        @usage LARGE_FILES(Files...)

        Use alrge file ether from working copy or from remote storage via placeholder <File>.remote
        If <File> is presemt locally (and not a symlink!) it will be copied to build directory.
        Otherwise macro will try to locate <File>.remote, parse it retrieve ot during build phase.
    """
    args = list(args)
    for arg in args:
        src = unit.resolve_arc_path(arg)
        if src.startswith("$S"):
            msg = "Used local large file {}. Don't forget to run 'ya upload --update-external' and commit {}.{}".format(src, src, PLACEHOLDER_EXT)
            unit.message(["warn", msg])
            unit.oncopy_file([arg, arg])
        else:
            out_file = strip_roots(os.path.join(unit.path(), arg))
            external = "{}.{}".format(arg, PLACEHOLDER_EXT)
            unit.on_from_external([external, out_file, 'OUT_NOAUTO', arg])
            unit.onadd_check(['check.external', external])

