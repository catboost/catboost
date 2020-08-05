import os
import ymake
from _common import strip_roots

PLACEHOLDER_EXT = "external"


def onlarge_files(unit, *args):
    """
        @usage LARGE_FILES([AUTOUPDATED] Files...)

        Use large file ether from working copy or from remote storage via placeholder <File>.external
        If <File> is present locally (and not a symlink!) it will be copied to build directory.
        Otherwise macro will try to locate <File>.external, parse it retrieve ot during build phase.
    """
    args = list(args)

    if args and args[0] == 'AUTOUPDATED':
        args = args[1:]

    for arg in args:
        if arg == 'AUTOUPDATED':
            unit.message(["warn", "Please set AUTOUPDATED argument before other file names"])
            continue

        src = unit.resolve_arc_path(arg)
        if src.startswith("$S"):
            msg = "Used local large file {}. Don't forget to run 'ya upload --update-external' and commit {}.{}".format(src, src, PLACEHOLDER_EXT)
            unit.message(["warn", msg])
            unit.oncopy_file([arg, arg])
        else:
            out_file = strip_roots(os.path.join(unit.path(), arg))
            external = "{}.{}".format(arg, PLACEHOLDER_EXT)
            from_external_cmd = [external, out_file, 'OUT_NOAUTO', arg]
            if os.path.dirname(arg):
                from_external_cmd.extend(("RENAME", os.path.basename(arg)))
            unit.on_from_external(from_external_cmd)
            unit.onadd_check(['check.external', external])

