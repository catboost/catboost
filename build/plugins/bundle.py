import os


def onbundle(unit, *args):
    i = 0
    while i < len(args):
        if i + 2 < len(args) and args[i + 1] == "NAME":
            target, name = args[i], args[i + 2]
            i += 3
        else:
            target, name = args[i], os.path.basename(args[i])
            i += 1

        unit.onbundle_target([target, name])
