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


def get_variables(unit):
    orig_variables = macro_calls_to_dict(unit, extract_macro_calls(unit, '_DOCS_VARS_VALUE'))
    return {k: unit.get(k) or v for k, v in orig_variables.items()}


def onprocess_docs(unit, *args):
    if unit.enabled('_DOCS_USE_PLANTUML'):
        unit.on_docs_yfm_use_plantuml([])

    if unit.get('_DOCS_DIR_VALUE') == '':
        unit.on_yfm_docs_dir([unit.get('_YFM_DOCS_DIR_DEFAULT_VALUE')])

    variables = get_variables(unit)
    if variables:
        unit.set(['_DOCS_VARS_FLAG', '--vars {}'.format(json.dumps(json.dumps(variables, sort_keys=True)))])


def onprocess_mkdocs(unit, *args):
    variables = get_variables(unit)
    if variables:
        unit.set(['_DOCS_VARS_FLAG', ' '.join(['--var {}={}'.format(k, v) for k, v in variables.items()])])
