from build.plugins.lib.nots.package_manager.base import utils


def test_extract_package_name_from_path():
    happy_checklist = [
        ("@yandex-int/foo-bar-baz/some/path/inside/the/package", "@yandex-int/foo-bar-baz"),
        ("@yandex-int/foo-bar-buzz", "@yandex-int/foo-bar-buzz"),
        ("package-wo-scope", "package-wo-scope"),
        ("p", "p"),
        ("", ""),
    ]

    for item in happy_checklist:
        package_name = utils.extract_package_name_from_path(item[0])
        assert package_name == item[1]
