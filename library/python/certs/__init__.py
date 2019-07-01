_builtin_cadata = None


def builtin_cadata():
    global _builtin_cadata
    if _builtin_cadata is None:
        import __res
        data = __res.find(b'/builtin/cacert')
        # load_verify_locations expects PEM cadata to be an ASCII-only unicode
        # object, so we discard unicode in comments.
        _builtin_cadata = data.decode('ASCII', errors='ignore')
    return _builtin_cadata
