"""Test for the "import *" functionality.

As imort * can be only done at module level, it has been added in a separate file
"""
import unittest

prev_locals = list(locals())
from dateutil import *
new_locals = {name:value for name,value in locals().items()
              if name not in prev_locals}
new_locals.pop('prev_locals')

class ImportStarTest(unittest.TestCase):
    """ Test that `from dateutil import *` adds the modules in __all__ locally"""

    def testImportedModules(self):
        import dateutil.easter
        import dateutil.parser
        import dateutil.relativedelta
        import dateutil.rrule
        import dateutil.tz
        import dateutil.utils
        import dateutil.zoneinfo

        self.assertEquals(dateutil.easter, new_locals.pop("easter"))
        self.assertEquals(dateutil.parser, new_locals.pop("parser"))
        self.assertEquals(dateutil.relativedelta, new_locals.pop("relativedelta"))
        self.assertEquals(dateutil.rrule, new_locals.pop("rrule"))
        self.assertEquals(dateutil.tz, new_locals.pop("tz"))
        self.assertEquals(dateutil.utils, new_locals.pop("utils"))
        self.assertEquals(dateutil.zoneinfo, new_locals.pop("zoneinfo"))

        self.assertFalse(new_locals)
