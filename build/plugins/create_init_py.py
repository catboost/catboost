import os

from _common import sort_by_keywords


def oncreate_init_py(unit, *args):
    keywords = {"DESTINATION": 1, "INCLUDING_DEST_DIR": 0, "RESULT": 1}

    flat_args, spec_args = sort_by_keywords(keywords, args)
    generated = []

    dest_dir = spec_args["DESTINATION"][0] if "DESTINATION" in spec_args else "$ARCADIA_BUILD_ROOT"
    if "INCLUDING_DEST_DIR" in spec_args:
        generated.append(os.path.join(dest_dir, "__init__.py"))

    for proto_file in flat_args:
        path_list = proto_file.split(os.sep)[:-1]

        for idx, val in enumerate(path_list):
            generated.append(os.path.join(dest_dir, os.path.join(*path_list[0:len(path_list) - idx]), "__init__.py"))

    generated = list(set(generated))

    unit.ontouch(generated)
    if "RESULT" in spec_args:
        unit.set([spec_args["RESULT"][0], " ".join(generated)])


def oncreate_init_py_structure(unit, *args):
    if unit.get('DISTBUILD'):
        return
    target_dir = unit.get('PY_PROTOS_FOR_DIR')
    path_list = target_dir.split(os.path.sep)[1:]
    inits = [os.path.join("${ARCADIA_BUILD_ROOT}", '__init__.py')]
    for i in range(1, len(path_list) + 1):
        inits.append(os.path.join("${ARCADIA_BUILD_ROOT}", os.path.join(*path_list[0:i]), '__init__.py'))
    unit.ontouch(inits)


def oncreate_init_py_for(unit, *args):
    paths = set()
    for arg in args:
        path_list = arg.split(os.sep)[:-1]

        for idx, val in enumerate(path_list):
            paths.add(os.path.join('${ARCADIA_BUILD_ROOT}', os.path.join(*path_list[0:len(path_list) - idx]), "__init__.py"))

    unit.ontouch(list(paths))
