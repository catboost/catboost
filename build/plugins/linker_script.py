def onlinker_script(unit, *args):
    """
        @usage: LINKER_SCRIPT(Files...)

        Specify files to be used as a linker script
    """
    for arg in args:
        if not arg.endswith(".ld") and not arg.endswith(".ld.in"):
            unit.message(['error', "Invalid linker script extension: {}".format(arg)])
            return

    unit.onglobal_srcs(list(args))
