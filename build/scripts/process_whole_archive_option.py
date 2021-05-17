
def process_whole_archive_for_global_libs(args):
    start_wa = '--start-wa'
    end_wa = '--end-wa'
    is_inside_wa = False
    cmd = []
    for arg in args:
        if arg == start_wa:
            is_inside_wa = True
        elif arg == end_wa:
            is_inside_wa = False
        elif is_inside_wa:
            cmd.append('-Wl,-force_load,' + arg)
        else:
            cmd.append(arg)
    return cmd

