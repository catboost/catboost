from _common import rootrel_arc_src


def oncredits_disclaimer(unit, *args):
    if unit.get('WITH_CREDITS'):
        unit.message(["warn", "CREDITS WARNING: {}".format(' '.join(args))])

def oncheck_contrib_credits(unit, *args):
    module_path = rootrel_arc_src(unit.path(), unit)
    excepts = set()
    if 'EXCEPT' in args:
        args = list(args)
        except_pos = args.index('EXCEPT')
        excepts = set(args[except_pos + 1:])
        args = args[:except_pos]
    for arg in args:
        if module_path.startswith(arg) and not unit.get('CREDITS_TEXTS_FILE') and not unit.get('NO_CREDITS_TEXTS_FILE'):
            for ex in excepts:
                if module_path.startswith(ex):
                    break
            else:
                unit.message(["error", "License texts not found. See https://st.yandex-team.ru/DTCC-324"])
