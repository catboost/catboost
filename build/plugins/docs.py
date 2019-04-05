import base64
import json
import os

DELIM = '================================'


def extract_macro_calls(unit, macro_value_name):
    if not unit.get(macro_value_name):
        return {}

    def split_args(arg):
        if arg is None:
            return None

        kv = filter(None, arg.split('='))
        if len(kv) != 2:
            unit.message(['error', 'Invalid variables specification "{}": value expected to be in form %name%=%value% (with no spaces)'.format(arg)])
            return None

        return kv

    return dict(filter(None, map(split_args, unit.get(macro_value_name).replace('$' + macro_value_name, '').split())))


def onprocess_docslib(unit, *args):
    generate_dart(unit, as_lib=True)


def onprocess_docs(unit, *args):
    generate_dart(unit)


def generate_dart(unit, as_lib=False):

    module_dir = os.path.normpath(unit.path()[3:])
    docs_dir = (unit.get('DOCSDIR') or '').rstrip('/')
    if docs_dir:
        docs_dir = os.path.normpath(docs_dir)
        unit.onsrcdir(docs_dir)
    else:
        docs_dir = module_dir

    docs_config = os.path.normpath(unit.get('DOCSCONFIG') or 'mkdocs.yml')
    if os.path.sep not in docs_config:
        docs_config = os.path.join(module_dir, docs_config)
    elif not docs_config.startswith(docs_dir + os.path.sep):
        unit.message(['error', 'DOCS_CONFIG value "{}" is outside the project directory and DOCS_DIR'.format(docs_config)])
        return

    if not os.path.exists(unit.resolve('$S/' + docs_config)):
        unit.message(['error', 'DOCS_CONFIG value "{}" does not exist'.format(docs_config)])
        return

    data = {
        'PATH': module_dir,
        'MODULE_TAG': unit.get('MODULE_TAG'),
        'DOCSDIR': docs_dir,
        'DOCSCONFIG': docs_config,
        'DOCSVARS': extract_macro_calls(unit, 'DOCSVARS'),
        'DOCSLIB': as_lib,
        'PEERDIRS': [d[3:] for d in unit.get_module_dirs('PEERDIRS')],
    }

    dart = 'DOCS_DART: ' + base64.b64encode(json.dumps(data)) + '\n' + DELIM + '\n'

    unit.set_property(['DOCS_DART_DATA', dart])
