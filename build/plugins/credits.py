def oncredits_disclaimer(unit, *args):
    if unit.get('WITH_CREDITS'):
        unit.message(["warn", "CREDITS WARNING: {}".format(' '.join(args))])
