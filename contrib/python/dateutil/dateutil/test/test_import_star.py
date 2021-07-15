"""Test for the "import *" functionality.

As import * can be only done at module level, it has been added in a separate file
"""
import pytest

prev_locals = list(locals())
from dateutil import *
new_locals = {name:value for name,value in locals().items()
              if name not in prev_locals}
new_locals.pop('prev_locals')


@pytest.mark.import_star
def test_imported_modules():
    """ Test that `from dateutil import *` adds modules in __all__ locally """
    import dateutil.easter
    import dateutil.parser
    import dateutil.relativedelta
    import dateutil.rrule
    import dateutil.tz
    import dateutil.utils
    import dateutil.zoneinfo

    assert dateutil.easter == new_locals.pop("easter")
    assert dateutil.parser == new_locals.pop("parser")
    assert dateutil.relativedelta == new_locals.pop("relativedelta")
    assert dateutil.rrule == new_locals.pop("rrule")
    assert dateutil.tz == new_locals.pop("tz")
    assert dateutil.utils == new_locals.pop("utils")
    assert dateutil.zoneinfo == new_locals.pop("zoneinfo")

    assert not new_locals
