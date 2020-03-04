def onsuppressions(unit, *args):
    """
        SUPPRESSIONS() - allows to specify files with suppression notation which will be used by
        address, leak or thread sanitizer runtime by default.
        Use asan.supp filename for address sanitizer, lsan.supp for leak sanitizer
        and tsan.supp for thread sanitizer suppressions respectively.
        See https://clang.llvm.org/docs/AddressSanitizer.html#suppressing-memory-leaks
        for details.
    """
    import os

    valid = ("asan.supp", "tsan.supp", "lsan.supp")

    if unit.get("SANITIZER_TYPE") in ("leak", "address", "thread"):
        for x in args:
            if os.path.basename(x) not in valid:
                unit.message(['error', "Invalid suppression filename: {} (any of the following is expected: {})".format(x, valid)])
                return
        unit.onsrcs(["GLOBAL"] + list(args))
