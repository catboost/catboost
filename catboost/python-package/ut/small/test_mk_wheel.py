import os
import shutil
import sys

import pytest


PYTHON_PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


@pytest.fixture(scope="module")
def mk_wheel():
    """Import the standalone mk_wheel build script."""
    sys.path.insert(0, PYTHON_PACKAGE_DIR)
    try:
        import mk_wheel

        return mk_wheel
    finally:
        sys.path.pop(0)


def test_add_license_file_copies_license_into_dist_info(mk_wheel, tmp_path):
    arc_root = tmp_path / "arc_root"
    arc_root.mkdir()
    (arc_root / "LICENSE").write_text("Apache License, Version 2.0\n")

    dist_info_dir = tmp_path / "catboost-1.2.10.dist-info"
    dist_info_dir.mkdir()

    mk_wheel.add_license_file(str(arc_root), str(dist_info_dir))

    license_path = dist_info_dir / "licenses" / "LICENSE"
    assert license_path.exists()
    assert license_path.read_text() == "Apache License, Version 2.0\n"


def test_make_record_includes_license_file(mk_wheel, tmp_path):
    arc_root = tmp_path / "arc_root"
    arc_root.mkdir()
    (arc_root / "LICENSE").write_text("Apache License, Version 2.0\n")

    dist_info_dir = tmp_path / "catboost-1.2.10.dist-info"
    dist_info_dir.mkdir()
    mk_wheel.add_license_file(str(arc_root), str(dist_info_dir))

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    shutil.copytree(dist_info_dir, wheel_dir / dist_info_dir.name)

    mk_wheel.make_record(str(wheel_dir), str(wheel_dir / dist_info_dir.name))

    record = (wheel_dir / dist_info_dir.name / "RECORD").read_text()
    assert "catboost-1.2.10.dist-info/licenses/LICENSE" in record
    assert ",sha256=" in record


def test_add_license_file_missing_license_raises(mk_wheel, tmp_path):
    arc_root = tmp_path / "arc_root"
    arc_root.mkdir()  # no LICENSE file

    dist_info_dir = tmp_path / "catboost-1.2.10.dist-info"
    dist_info_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        mk_wheel.add_license_file(str(arc_root), str(dist_info_dir))
