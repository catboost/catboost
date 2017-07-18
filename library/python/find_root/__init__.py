import os


def is_root(path):
    return os.path.exists(os.path.join(path, ".arcadia.root")) or os.path.exists(os.path.join(path, 'devtools', 'ya', 'ya.conf.json'))


def detect_root(path, detector=is_root):
    return _find_path(path, detector)


def _find_path(starts_from, check):
    p = os.path.realpath(starts_from)
    while True:
        if check(p):
            return p
        next_p = os.path.dirname(p)
        if next_p == p:
            return None
        p = next_p
