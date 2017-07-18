import os


def onbundle(unit, *args):
    for arg in args:
        unit.onbundle_program([arg, os.path.basename(arg)])
