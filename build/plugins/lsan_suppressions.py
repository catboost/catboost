def onlsan_suppressions(unit, *args):
    if unit.get("SANITIZER_TYPE") in ("leak", "address"):
        unit.onsrcs(["GLOBAL"] + list(args))
