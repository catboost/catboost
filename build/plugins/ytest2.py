import os
import _common


def dir_stmts(unit, dir):
    unit.onpeerdir(dir)
    unit.onsrcdir(os.sep.join([dir, 'tests']))


def pytest_base(unit, args):
    related_prj_dir = args[0]
    related_prj_name = args[1]
    dir_stmts(unit, related_prj_dir)
    ytest_base(unit, related_prj_dir, related_prj_name, args[2:])
    unit.set(['ADDITIONAL_PATH', '--test-related-path ${ARCADIA_ROOT}/test'])


def ytest_base(unit, related_prj_dir, related_prj_name, args):
    keywords = {"DEPENDS": -1, "DATA": -1}
    flat_args, spec_args = _common.sort_by_keywords(keywords, args)
    unit.set(['TEST-NAME', os.path.basename(flat_args[0])])
    unit.set(['SCRIPT-REL-PATH', flat_args[1]])
    unit.set(['SOURCE-FOLDER-PATH', related_prj_dir])
    unit.set(['BUILD-FOLDER-PATH', os.path.join('$B', related_prj_dir)])
    unit.set(['TESTED-BINARY-PATH', flat_args[0]])

    custom_deps = ' '.join(spec_args["DEPENDS"]) if "DEPENDS" in spec_args else ''
    unit.set(['CUSTOM-DEPENDENCIES', custom_deps])
    data_lst = spec_args.get('DATA', []) + (unit.get(['__test_data']) or '').split(' ')
    data_lst.sort()
    data = '\"' + ';'.join(data_lst) + '\"' if data_lst else ''
    unit.set(['TEST-DATA', data])
    ya_root = unit.get('YA_ROOT')
    unit.set(['TEST_RUN_SCRIPT', 'devtools/{}/test/node/run_test.py'.format(ya_root)])

    related_dirs_list = ['{ARCADIA_ROOT}/devtools/${YA_ROOT}', '${ARCADIA_ROOT}/devtools/${YA_ROOT}', '$RELATED_TARGET_SRCDIR']
    related_dirs_value = []
    for rel in related_dirs_list:
        related_dirs_value.extend(['--test-related-path', rel])
    unit.set(['RELATED_DIRS', ' '.join(related_dirs_value)])
    unit.set(['TEST_KV', '${{kv;hide:"test_related_dirs {}"}}'.format(' '.join(related_dirs_list))])


def on_unittest(unit, *args):
    related_prj_name = args[0]
    related_prj_dir = args[1][3:]
    ya_root = unit.get('YA_ROOT')
    unit.set(['SPECIFIC_RUN_SCRIPT', 'devtools/{}/test/scripts/run_ut.py'.format(ya_root)])
    unit.set(['TEST_TYPE', '${kv;hide:"test-type unittest"}'])
    ytest_base(unit, related_prj_dir, related_prj_name, args)


def on_ytest(unit, *args):
    pytest_base(unit, args)


def on_py_test(unit, *args):
    pytest_base(unit, args)
