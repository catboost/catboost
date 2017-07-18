
def onaddinclself(unit, *args):
    path = unit.path()
    if path.startswith("$S/") or path.startswith("$B/"):
        path = path[3:]
    unit.onaddincl(path)
