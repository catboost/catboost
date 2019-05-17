import os


def onbundle(unit, *args):
    """
        @usage BUNDLE(<Dir [NAME Name]>...)

        Brings build artefact from module Dir under optional Name to the current module (e.g. UNION)
        If NAME is not specified, the name of the Dir's build artefact will be preserved
        It makes little sense to specify BUNDLE on non-final targets and so this may stop working without prior notice.
        Bundle on multimodule will select final target among multimodule variants and will fail if there are none or more than one.
    """
    i = 0
    while i < len(args):
        if i + 2 < len(args) and args[i + 1] == "NAME":
            target, name = args[i], args[i + 2]
            i += 3
        else:
            target, name = args[i], os.path.basename(args[i])
            i += 1

        unit.on_bundle_target([target, name])
