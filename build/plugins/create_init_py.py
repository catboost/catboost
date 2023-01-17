import os

from _common import sort_by_keywords


def oncreate_init_py_structure(unit, *args):
    if unit.get('DISTBUILD'):
        return
    target_dir = unit.get('PY_PROTOS_FOR_DIR')
    path_list = target_dir.split(os.path.sep)[1:]
    inits = [os.path.join("${ARCADIA_BUILD_ROOT}", '__init__.py')]
    for i in range(1, len(path_list) + 1):
        inits.append(os.path.join("${ARCADIA_BUILD_ROOT}", os.path.join(*path_list[0:i]), '__init__.py'))
    unit.ontouch(inits)

