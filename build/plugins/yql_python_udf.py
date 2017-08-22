from _common import sort_by_keywords


def get_or_default(kv, name, default):
    if name in kv:
        return kv[name][0]
    return default


def onregister_yql_python_udf(unit, *args):
    flat, kv = sort_by_keywords({'NAME': 1, 'RESOURCE_NAME': 1}, args)
    assert len(flat) == 0
    name = get_or_default(kv, 'NAME', 'CustomPython')
    resource_name = get_or_default(kv, 'RESOURCE_NAME', name)

    use_arcadia_python = unit.get('USE_ARCADIA_PYTHON') == 'yes'

    unit.onyql_abi_version(['2', '0', '0'])
    unit.onpeerdir(['yql/udfs/common/python/python_udf'])

    if use_arcadia_python:
        flavor = 'Arcadia'
        unit.onpeerdir([
            'library/python/runtime',
            'yql/udfs/common/python/main',
        ])
    else:
        flavor = 'System'

    path = name + '.yql_python_udf.cpp'
    unit.onbuiltin_python([
        'build/scripts/gen_yql_python_udf.py',
        flavor, name, resource_name, path,
        'OUT', path,
    ])
