import json


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


def onprocess_docs(unit, *args):
    build_tool = unit.get('_DOCS_BUILDER_VALUE')
    if build_tool:
        if build_tool not in ['mkdocs', 'yfm']:
            unit.message(['error', 'Unsupported build tool {}'.format(build_tool)])
    else:
        build_tool = 'yfm'
        unit.ondocs_builder([build_tool])
    if build_tool == 'yfm' and unit.enabled('_DOCS_USE_PLANTUML'):
        unit.on_docs_yfm_use_plantuml([])
    orig_variables = macro_calls_to_dict(unit, extract_macro_calls(unit, '_DOCS_VARS_VALUE'))
    variables = {k: unit.get(k) or v for k, v in orig_variables.items()}
    if variables:
        if build_tool == 'mkdocs':
            unit.set(['_DOCS_VARS_FLAG', ' '.join(['--var {}={}'.format(k, v) for k, v in variables.items()])])
        elif build_tool == 'yfm':
            unit.set(['_DOCS_VARS_FLAG', '--vars {}'.format(json.dumps(json.dumps(variables, sort_keys=True)))])
        else:
            assert False, 'Unexpected build_tool value: [{}]'.format(build_tool)
