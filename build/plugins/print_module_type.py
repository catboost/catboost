def onprint_module_type(unit, *args):
    filepath = unit.get('KIWI_OUT_FILE')
    if len(args) >= 2 and filepath is not None:
        with open(filepath, "a") as file_handler:
            print >>file_handler, "{0} {1} {2}".format(args[0], args[1], unit.path())
