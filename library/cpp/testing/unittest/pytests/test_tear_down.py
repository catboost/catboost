import yatest.common as ya_common


def test_sanity():
    test_path = ya_common.binary_path("library/cpp/testing/unittest/pytests/test_subject/library-cpp-testing-unittest-pytests-test_subject")

    tests = ya_common.execute([test_path, '-A']).std_out \
        .decode().strip().split('\n')

    assert tests

    for test in tests:
        name = test.split('::')[1]
        res = ya_common.execute([test_path, test], check_exit_code=False)
        assert res.exit_code != 0
        lines = res.std_err.decode().split('\n')
        assert f"{name}: TearDown is ran" in lines
