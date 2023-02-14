import os


def which(name, flags=os.X_OK):
    exts = [f for f in os.environ.get('PATHEXT', '').split(os.pathsep) if f]
    path = os.environ.get('PATH', None)
    if path is None:
        return None
    for p in os.environ.get('PATH', '').split(os.pathsep):
        p = os.path.join(p, name)
        if os.access(p, flags):
            return p
        for e in exts:
            pext = p + e
            if os.access(pext, flags):
                return pext
    return None
