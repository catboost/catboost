import base64
import json
import os

DELIM = '================================'


def onprocess_docs(unit, *args):
    smart_mode = unit.get('DOCS_SMART')
    if not smart_mode:
        return

    module_dir = os.path.normpath(unit.path()[3:])
    docs_dir = (unit.get('DOCSDIR') or '').rstrip('/')
    if docs_dir:
        docs_dir = os.path.normpath(docs_dir)
        unit.onsrcdir(docs_dir)
    else:
        docs_dir = module_dir

    docs_config = os.path.normpath(unit.get('DOCSCONFIG') or os.path.join(docs_dir, 'mkdocs.yml'))
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
        'DOCSDIR': docs_dir,
        'DOCSCONFIG': docs_config,
    }

    for k, v in data.items():
        if not v:
            data.pop(k)

    dart = 'DOCS_DART: ' + base64.b64encode(json.dumps(data)) + '\n' + DELIM + '\n'

    unit.set_property(['DOCS_DART_DATA', dart])
