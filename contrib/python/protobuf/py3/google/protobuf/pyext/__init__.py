import warnings

with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
    import google.protobuf.pyext._message
