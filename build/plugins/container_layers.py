from _common import rootrel_arc_src

def oncheck_allowed_path(unit, *args):
    module_path = rootrel_arc_src(unit.path(), unit)
    if not (module_path.startswith("junk") or module_path.startswith("base_layers")):
        unit.message(["error", "Cannot create container layer in this directory. See https://st.yandex-team.ru/DTCC-1123"])
