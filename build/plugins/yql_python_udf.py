from _common import sort_by_keywords


def get_or_default(kv, name, default):
    if name in kv:
        return kv[name][0]
    return default


def onregister_yql_python_udf(unit, *args):
    flat, kv = sort_by_keywords({'NAME': 1, 'RESOURCE_NAME': 1, 'ADD_LIBRA_MODULES': 1}, args)
    assert len(flat) == 0
    name = get_or_default(kv, 'NAME', 'CustomPython')
    resource_name = get_or_default(kv, 'RESOURCE_NAME', name)
    add_libra_modules = get_or_default(kv, 'ADD_LIBRA_MODULES', 'no') == 'yes'

    use_arcadia_python = unit.get('USE_ARCADIA_PYTHON') == 'yes'
    py3 = unit.get('PYTHON3') == 'yes'

    unit.onyql_abi_version(['2', '9', '0'])
    unit.onpeerdir(['yql/udfs/common/python/python_udf'])
    unit.onpeerdir(['yql/library/udf'])

    if add_libra_modules:
        unit.onpeerdir(['quality/user_sessions/libra_arc/noyql'])
        unit.onpeerdir(['yql/udfs/quality/libra/module'])

    if use_arcadia_python:
        flavor = 'Arcadia'
        unit.onpeerdir([
            'library/python/runtime',
            'yql/udfs/common/python/main'
        ] if not py3 else [
            'library/python/runtime_py3',
            'yql/udfs/common/python/main_py3'
        ])
    else:
        flavor = 'System'

    output_includes = [
        'yql/udfs/common/python/python_udf/python_udf.h',
        'yql/library/udf/udf_registrator.h',
    ]
    if add_libra_modules:
        output_includes.append('yql/udfs/quality/libra/module/module.h')

    path = name + '.yql_python_udf.cpp'
    libra_flag = '1' if add_libra_modules else '0'
    unit.onpython([
        'build/scripts/gen_yql_python_udf.py',
        flavor, name, resource_name, path, libra_flag,
        'OUT', path,
        'OUTPUT_INCLUDES',
    ] + output_includes
    )
