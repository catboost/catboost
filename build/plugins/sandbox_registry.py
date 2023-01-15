import os

import ymake


def onregister_sandbox_import(unit, *args):
    args = iter(args)
    for path in args:
        path = os.path.normpath(path)
        source = unit.resolve_arc_path(path)
        abs_source = unit.resolve(source)
        if not os.path.exists(abs_source):
            ymake.report_configure_error('REGISTER_SANDBOX_IMPORT: File or directory {} does not exists'.format(path))
        splited_path = path.split(os.sep)
        l, r = 0, len(splited_path)
        if splited_path[-1] == "__init__.py":
            r -= 1
        if not splited_path[0]:
            l += 1
        path = ".".join(splited_path[l:r])
        unit.onresource(["-", "{}.{}={}".format("SANDBOX_TASK_REGISTRY", path, path)])
