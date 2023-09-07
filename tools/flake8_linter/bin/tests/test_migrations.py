import pytest
from configparser import ConfigParser

from . import util

FLAKE8_CONFIG_DATA = """
    [flake8]
    select = E, W, F
    ignore =
        E122,
        E743,
        F403,
        W605,
"""
MIGRATION_CONFIG_DATA = """
    flake8:
        F401:
            ignore:
            - F401
            prefixes:
            - project401
        SKIP:
            ignore:
            - "*"
            prefixes:
            - project_skip
"""
FILE_WITHOUT_EXCEPTIONS = "project/file1.py"
FILE_IGNORE_F401 = "project401/file2.py"
FILE_SKIPPED = "project_skip/file3.py"


@pytest.mark.parametrize(
    "test_file, added_ignore, disable_migrations",
    [
        (FILE_WITHOUT_EXCEPTIONS, "", False),
        (FILE_IGNORE_F401, "\nF401,", False),
        (FILE_IGNORE_F401, "", True),
    ],
)
def test_ignore(test_file, added_ignore, disable_migrations):
    test_files = [test_file]

    runner = util.LinterRunner()
    runner.create_source_file(util.FLAKE8_CONFIG_FILE, FLAKE8_CONFIG_DATA)
    runner.create_source_file(util.MIGRATIONS_CONFIG_FILE, MIGRATION_CONFIG_DATA)
    runner.create_source_tree(test_files)

    disable_migrations_params = {"extra_params": {"DISABLE_FLAKE8_MIGRATIONS": "yes"}} if disable_migrations else {}
    run_result = runner.run_test(test_files, custom_params=disable_migrations_params)

    assert len(run_result.flake8_launches) == 1

    launch = run_result.flake8_launches[0]
    got_config = ConfigParser()
    got_config.read_string(launch.config_data)
    expected_config = ConfigParser()
    expected_config.read_string(FLAKE8_CONFIG_DATA)
    expected_config["flake8"]["ignore"] += added_ignore

    util.assert_configs(got_config, expected_config)

    assert launch.rel_file_paths == test_files


def test_skipped():
    test_files = [FILE_SKIPPED]

    runner = util.LinterRunner()
    runner.create_source_file(util.FLAKE8_CONFIG_FILE, FLAKE8_CONFIG_DATA)
    runner.create_source_file(util.MIGRATIONS_CONFIG_FILE, MIGRATION_CONFIG_DATA)
    runner.create_source_tree(test_files)

    run_result = runner.run_test(test_files)

    assert len(run_result.flake8_launches) == 0

    abs_test_file_path = runner.abs_source_file_path(FILE_SKIPPED)
    assert run_result.report_data["report"][abs_test_file_path]["status"] == "SKIPPED"


def test_group_files_by_config():
    test_files = [FILE_WITHOUT_EXCEPTIONS, FILE_IGNORE_F401]

    runner = util.LinterRunner()
    runner.create_source_file(util.FLAKE8_CONFIG_FILE, FLAKE8_CONFIG_DATA)
    runner.create_source_file(util.MIGRATIONS_CONFIG_FILE, MIGRATION_CONFIG_DATA)
    runner.create_source_tree(test_files)

    run_result = runner.run_test(test_files)

    assert len(run_result.flake8_launches) == 2

    for launch in run_result.flake8_launches:
        rel_file_paths = launch.rel_file_paths
        got_config = ConfigParser()
        got_config.read_string(launch.config_data)
        # Relaxed check if config is matched with a checked file
        # Thorough config check is done in test_ignore()
        if rel_file_paths == [FILE_WITHOUT_EXCEPTIONS]:
            assert "F401" not in got_config["flake8"]["ignore"]
        elif rel_file_paths == [FILE_IGNORE_F401]:
            assert "F401" in got_config["flake8"]["ignore"]
        else:
            pytest.fail("Unexpected file paths passed to flake8 binary: {}".format(rel_file_paths))


@pytest.mark.parametrize(
    "migrations_file, expected_ignore",
    [
        (None, "F777"),
        ("", None),
        ("build/config/other_migration.yaml", "F888"),
    ],
)
def test_migration_file_from_env(migrations_file, expected_ignore):
    # Env var _YA_TEST_FLAKE8_CONFIG overrides file name from configs parameter. _YA_TEST_FLAKE8_CONFIG:
    # - is not defined - use migrations file from configs parameter
    # - is empty - don't use migrations file at all
    # - not empty - use variable value as migrations file name
    config_migrations = """
        flake8:
            ignore:
                ignore:
                - F777
                prefixes:
                - project
    """
    test_files = ["project/test.py"]

    runner = util.LinterRunner()
    runner.create_source_file(util.FLAKE8_CONFIG_FILE, FLAKE8_CONFIG_DATA)
    runner.create_source_file(util.MIGRATIONS_CONFIG_FILE, config_migrations)
    runner.create_source_tree(test_files)
    env = {}
    if migrations_file is not None:
        if migrations_file:
            env_var_migrations = """
                flake8:
                    ignore:
                        ignore:
                        - F888
                        prefixes:
                        - project
            """
            runner.create_source_file(migrations_file, env_var_migrations)
            env_var_value = runner.abs_source_file_path(migrations_file)
        else:
            env_var_value = ""
        env["_YA_TEST_FLAKE8_CONFIG"] = env_var_value

    run_result = runner.run_test(test_files, env=env)

    assert len(run_result.flake8_launches) == 1

    launch = run_result.flake8_launches[0]
    got_config = ConfigParser()
    got_config.read_string(launch.config_data)
    ignores = got_config["flake8"]["ignore"]
    if expected_ignore:
        assert expected_ignore in ignores
    else:
        assert "F777" not in ignores
        assert "F888" not in ignores
