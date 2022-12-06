from build.plugins.lib.nots.semver import Version, Operator, VersionRange


def test_from_str():
    # arrange
    range_str = ">= 1.2.3"

    # act
    range = VersionRange.from_str(range_str)

    # assert
    assert isinstance(range, VersionRange)
    assert range.operator == Operator.GE


def test_from_str_no_operator():
    # arrange
    range_str = r"¯\_(ツ)_/¯"
    error = None

    # act
    try:
        VersionRange.from_str(range_str)
    except Exception as exception:
        error = exception

    # assert
    assert isinstance(error, ValueError)
    assert str(error) == "Unsupported version range: '{}'. Currently we only support ranges formatted like so: '>= 1.2.3'".format(range_str)


def test_init():
    # arrange
    operator = Operator.GE
    version = Version.from_str("1.2.3")

    # act
    range = VersionRange(operator, version)

    # assert
    assert range.operator == Operator.GE
    assert range.version == Version(1, 2, 3)


def test_is_satisfied_by_starts_with():
    # arrange
    version = Version.from_str("1.2.3")
    range = VersionRange.from_str(">= 1.2.3")

    # act + assert
    assert range.is_satisfied_by(version)


def test_is_satisfied_by_includes():
    # arrange
    version = Version.from_str("5.8.2")
    range = VersionRange.from_str(">= 1.2.3")

    # act + assert
    assert range.is_satisfied_by(version)


def test_is_satisfied_by_not_includes():
    # arrange
    version = Version.from_str("1.2.2")
    range = VersionRange.from_str(">= 1.2.3")

    # act + assert
    assert not range.is_satisfied_by(version)
