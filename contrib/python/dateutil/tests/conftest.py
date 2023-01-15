import os
import pytest


# Configure pytest to ignore xfailing tests
# See: https://stackoverflow.com/a/53198349/467366
def pytest_collection_modifyitems(items):
    for item in items:
        marker_getter = getattr(item, 'get_closest_marker', None)

        # Python 3.3 support
        if marker_getter is None:
            marker_getter = item.get_marker

        marker = marker_getter('xfail')

        # Need to query the args because conditional xfail tests still have
        # the xfail mark even if they are not expected to fail
        if marker and (not marker.args or marker.args[0]):
            item.add_marker(pytest.mark.no_cover)


def set_tzpath():
    """
    Sets the TZPATH variable if it's specified in an environment variable.
    """
    tzpath = os.environ.get('DATEUTIL_TZPATH', None)

    if tzpath is None:
        return

    path_components = tzpath.split(':')

    print("Setting TZPATH to {}".format(path_components))

    from dateutil import tz
    tz.TZPATHS.clear()
    tz.TZPATHS.extend(path_components)


set_tzpath()
