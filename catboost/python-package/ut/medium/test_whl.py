import yatest.common
import shutil
import os
import zipfile


PYTHON_PACKAGE_DIR = os.path.join("catboost", "python-package")


def test_wheel():
    shutil.copy(yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "mk_wheel.py")), 'mk_wheel.py')
    from mk_wheel import PythonTrait, make_wheel
    cpu_so_name = PythonTrait('', '', []).so_name()
    cpu_so_path = yatest.common.binary_path(os.path.join(PYTHON_PACKAGE_DIR, "catboost", "no_cuda", cpu_so_name))
    wheel_name = 'catboost.whl'

    make_wheel(wheel_name, 'catboost', '0.0.0', yatest.common.source_path('.'), cpu_so_path)

    with zipfile.ZipFile(wheel_name, 'r') as f:
        f.extractall('catboost')

    python_binary = yatest.common.binary_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "python_binary", "catboost-python"))
    test_script = yatest.common.source_path(os.path.join(PYTHON_PACKAGE_DIR, "ut", "medium", "run_catboost.py"))

    source_data_path = yatest.common.source_path(os.path.join("catboost", "pytest", "data", "adult"))
    temp_data_path = os.path.join(yatest.common.test_output_path(), "data", "adult")
    shutil.copytree(source_data_path, temp_data_path)

    yatest.common.execute(
        [python_binary, test_script],
        env={'PYTHONPATH': os.path.join(os.getcwd(), 'catboost')},
        cwd=yatest.common.test_output_path()
    )
