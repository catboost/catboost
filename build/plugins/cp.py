import os

from _common import sort_by_keywords


def oncopy(unit, *args):
    keywords = {'RESULT': 1, 'KEEP_DIR_STRUCT': 0, 'DESTINATION': 1, 'FROM': 1}

    flat_args, spec_args = sort_by_keywords(keywords, args)

    dest_dir = spec_args['DESTINATION'][0] if 'DESTINATION' in spec_args else ''
    from_dir = spec_args['FROM'][0] if 'FROM' in spec_args else ''
    keep_struct = 'KEEP_DIR_STRUCT' in spec_args
    save_in_var = 'RESULT' in spec_args
    targets = []

    for source in flat_args:
        rel_path = ''
        path_list = source.split(os.sep)
        filename = path_list[-1]
        if keep_struct:
            if path_list[:-1]:
                rel_path = os.path.join(*path_list[:-1])
        source_path = os.path.join(from_dir, rel_path, filename)
        target_path = os.path.join(dest_dir, rel_path, filename)
        if save_in_var:
            targets.append(target_path)
        unit.oncopy_file([source_path, target_path])
    if save_in_var:
        unit.set([spec_args["RESULT"][0], " ".join(targets)])
