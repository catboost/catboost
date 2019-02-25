def onlsan_suppressions(unit, *args):
    """
        LSAN_SUPPRESSIONS() - allows to specify files with suppression notation which will be used by default.
        See https://clang.llvm.org/docs/AddressSanitizer.html#suppressing-memory-leaks
        for details.
    """
    if unit.get("SANITIZER_TYPE") in ("leak", "address"):
        unit.onsrcs(["GLOBAL"] + list(args))
