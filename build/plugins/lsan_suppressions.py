def onsuppressions(unit, *args):
    """
        SUPPRESSIONS() - allows to specify files with suppression notation which will be used by
        address, leak or thread sanitizer runtime by default.
        See https://clang.llvm.org/docs/AddressSanitizer.html#suppressing-memory-leaks
        for details.
    """
    if unit.get("SANITIZER_TYPE") in ("leak", "address", "thread"):
        unit.onsrcs(["GLOBAL"] + list(args))
