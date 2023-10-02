import pytest
import yatest.common
import shutil
import os
import zipfile


PYTHON_PACKAGE_DIR = os.path.join("catboost", "python-package")


def test_wheel():
    try:
        import catboost_pytest_lib  # noqa
    except ImportError:
        pytest.skip('test_wheel requires YA build')

    shutil.copy(yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "mk_wheel.py")), 'mk_wheel.py')
    from mk_wheel import BUILD_SYSTEM, PythonTrait, make_wheel

    py_trait = PythonTrait(BUILD_SYSTEM.YA, '', '', [])
    so_modules = {}
    for module_name, arc_path, dst_subdir in (
        ('_catboost', os.path.join('catboost', 'python-package', 'catboost', 'no_cuda'), ''),
        ('_hnsw', os.path.join('library', 'python', 'hnsw', 'hnsw'), 'hnsw')
    ):
        src = yatest.common.binary_path(os.path.join(arc_path, py_trait.built_so_name(module_name)))
        so_modules[module_name] = (src, dst_subdir)

    wheel_name = 'catboost.whl'

    make_wheel(wheel_name, 'catboost', '0.0.0', BUILD_SYSTEM.YA, yatest.common.source_path('.'), so_modules, should_build_widget=False)

    with zipfile.ZipFile(wheel_name, 'r') as f:
        f.extractall('catboost')

    python_binary = yatest.common.binary_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "python_binary", "catboost-python"))

    catboost_test_script = yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "run_catboost.py"))

    catboost_source_data_path = yatest.common.source_path(os.path.join("catboost", "pytest", "data", "adult"))
    catboost_temp_data_path = os.path.join(yatest.common.test_output_path(), "data", "adult")
    shutil.copytree(catboost_source_data_path, catboost_temp_data_path)

    yatest.common.execute(
        [python_binary, catboost_test_script],
        env={'PYTHONPATH': os.path.join(os.getcwd(), 'catboost')},
        cwd=yatest.common.test_output_path()
    )

    hnsw_test_script = yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "run_catboost_hnsw.py"))

    hnsw_source_data_path = yatest.common.source_path(os.path.join("library", "python", "hnsw", "ut", "data", "floats_60000"))
    hnsw_temp_data_path = os.path.join(yatest.common.test_output_path(), "hnsw_data")
    os.mkdir(hnsw_temp_data_path)
    shutil.copy(hnsw_source_data_path, hnsw_temp_data_path)

    yatest.common.execute(
        [python_binary, hnsw_test_script],
        env={'PYTHONPATH': os.path.join(os.getcwd(), 'catboost')},
        cwd=yatest.common.test_output_path()
    )
