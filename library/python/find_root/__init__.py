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


def try_get_arcadia_root(start_path=os.path.abspath(__file__)):
    """
    Extended Arcadia root lookup: use various env vars
    :param start_path: initial path for lookup
    Obtain Arcadia root or **empty string** when arcadia root cannot be found.
    """
    env_root = os.getenv("ARCADIA_ROOT")
    if env_root:
        return env_root

    path = start_path
    while path != "/" and not os.path.exists(os.path.join(path, ".arcadia.root")):
        path = os.path.dirname(path)

    # if after all, we reached root, try to check up from ya path
    # env variable "_" contains path to ya
    if path == "/" and start_path == os.path.abspath(__file__):
        path = try_get_arcadia_root(os.path.dirname(os.getenv("_")))

    return path if path != "/" else ""


def get_arcadia_root(start_path=os.path.abspath(__file__)):
    """
    :param start_path: initial path for lookup
    Obtain Arcadia root or raise exception when root cannot be found.
    """
    arcadia_root = try_get_arcadia_root(start_path=start_path)
    if not arcadia_root:
        raise Exception("Cannot find Arcadia root")
    return arcadia_root
