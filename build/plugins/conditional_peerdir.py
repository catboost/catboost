from _common import join_intl_paths


def onconditional_peerdir(unit, *args):
    dict_name = args[0].upper()
    use_var = "USE_" + dict_name
    make_var = "MAKE_" + dict_name + "_FROM_SOURCE"
    use_var_value = unit.get(use_var)
    make_var_value = unit.get(make_var)
    if use_var_value is None:
        unit.set([use_var, 'yes'])
        use_var_value = 'yes'

    if make_var_value is None:
        unit.set([make_var, 'no'])
        make_var_value = 'no'

    if use_var_value == 'yes':
        peer = join_intl_paths(args[1], 'source' if make_var_value == 'yes' else 'generated')
        unit.onpeerdir([peer])
