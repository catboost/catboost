from _common import rootrel_arc_src


def oncredits_disclaimer(unit, *args):
    if unit.get('WITH_CREDITS'):
        unit.message(["warn", "CREDITS WARNING: {}".format(' '.join(args))])

def oncheck_contrib_credits(unit, *args):
    module_path = rootrel_arc_src(unit.path(), unit)
    for arg in args:
        if module_path.startswith(arg) and not unit.get('CREDITS_TEXTS_FILE') and not unit.get('NO_CREDITS_TEXTS_FILE'):
            unit.message(["error", "License texts not found. See https://st.yandex-team.ru/DTCC-324"])
