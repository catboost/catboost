import base64
import json
import os
import ymake

DELIM = '================================'


def extract_macro_calls(unit, macro_value_name):
    if not unit.get(macro_value_name):
        return []

    return filter(None, unit.get(macro_value_name).replace('$' + macro_value_name, '').split())


def macro_calls_to_dict(unit, calls):
    def split_args(arg):
        if arg is None:
            return None

        kv = filter(None, arg.split('='))
        if len(kv) != 2:
            unit.message(['error', 'Invalid variables specification "{}": value expected to be in form %name%=%value% (with no spaces)'.format(arg)])
            return None

        return kv

    return dict(filter(None, map(split_args, calls)))


def onprocess_docslib(unit, *args):
    generate_dart(unit, as_lib=True)


def onprocess_docs(unit, *args):
    generate_dart(unit)


def generate_dart(unit, as_lib=False):
    module_dir = os.path.normpath(unit.path()[3:])
    docs_dir = (unit.get('DOCSDIR') or '').rstrip('/')
    if docs_dir:
        docs_dir = os.path.normpath(docs_dir)
        unit.set(['SRCDIR', docs_dir])
    else:
        docs_dir = module_dir

    build_tool = unit.get('DOCSBUILDER') or 'mkdocs'

    if build_tool not in ['mkdocs', 'yfm']:
        unit.message(['error', 'Unsupported build tool {}'.format(build_tool)])

    docs_config = unit.get('DOCSCONFIG')
    if not docs_config:
        docs_config = 'mkdocs.yml' if build_tool == 'mkdocs' else '.yfm'

    docs_config = os.path.normpath(docs_config)
    if os.path.sep not in docs_config:
        docs_config = os.path.join(module_dir if build_tool == 'mkdocs' else docs_dir, docs_config)

    if not docs_config.startswith(docs_dir + os.path.sep) and not docs_config.startswith(module_dir + os.path.sep) :
        unit.message(['error', 'DOCS_CONFIG value "{}" is outside the project directory and DOCS_DIR'.format(docs_config)])
        return

    if not os.path.exists(unit.resolve('$S/' + docs_config)):
        unit.message(['error', 'DOCS_CONFIG value "{}" does not exist'.format(docs_config)])
        return

    includes = extract_macro_calls(unit, 'DOCSINCLUDESOURCES')

    data = {
        'DOCS_NAME': unit.name(),
        'PATH': module_dir,
        'MODULE_TAG': unit.get('MODULE_TAG'),
        'DOCSDIR': docs_dir,
        'DOCSCONFIG': docs_config,
        'DOCSVARS': macro_calls_to_dict(unit, extract_macro_calls(unit, 'DOCSVARS')),
        'DOCSINCLUDESOURCES': includes,
        'DOCSLIB': as_lib,
        'PEERDIRS': '${PEERDIR}',
        'DOCSBUILDER': build_tool,
    }

    dart = 'DOCS_DART: ' + base64.b64encode(json.dumps(data)) + '\n' + DELIM + '\n'

    unit.set_property(['DOCS_DART_DATA', dart])

    for i in includes:
        include_path = unit.resolve('$S/' + i)

        if not os.path.exists(include_path):
            ymake.report_configure_error('DOCS_INCLUDE_SOURCES value "{}" does not exist'.format(i))

        elif not os.path.isfile(include_path):
            ymake.report_configure_error('DOCS_INCLUDE_SOURCES value "{}" must be a file'.format(i))
