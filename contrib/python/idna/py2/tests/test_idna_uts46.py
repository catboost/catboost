"""Tests for TR46 code."""

import gzip
import os.path
import re
import sys
import unittest

import idna

if sys.version_info[0] >= 3:
    unichr = chr
    unicode = str

_RE_UNICODE = re.compile(u"\\\\u([0-9a-fA-F]{4})")
_RE_SURROGATE = re.compile(u"[\uD800-\uDBFF][\uDC00-\uDFFF]")
_SKIP_TESTS = [
    # These appear to be errors in the test vectors. All relate to incorrectly applying
    # bidi rules across label boundaries. Appears independently confirmed
    # at http://www.alvestrand.no/pipermail/idna-update/2017-January/007946.html
    u'0\u00E0.\u05D0', u'0a\u0300.\u05D0', u'0A\u0300.\u05D0', u'0\u00C0.\u05D0', 'xn--0-sfa.xn--4db',
    u'\u00E0\u02c7.\u05D0', u'a\u0300\u02c7.\u05D0', u'A\u0300\u02c7.\u05D0', u'\u00C0\u02c7.\u05D0',
    'xn--0ca88g.xn--4db', u'0A.\u05D0', u'0a.\u05D0', '0a.xn--4db', 'c.xn--0-eha.xn--4db',
    u'c.0\u00FC.\u05D0', u'c.0u\u0308.\u05D0', u'C.0U\u0308.\u05D0', u'C.0\u00DC.\u05D0',
    u'\u06B6\u06DF\u3002\u2087\uA806', u'\u06B6\u06DF\u30027\uA806', 'xn--pkb6f.xn--7-x93e',
    u'\u06B6\u06DF.7\uA806', u'1.\uAC7E6.\U00010C41\u06D0', u'1.\u1100\u1165\u11B56.\U00010C41\u06D0',
    '1.xn--6-945e.xn--glb1794k',

    # These are transitional strings that compute to NV8 and thus are not supported
    # in IDNA 2008.
    u'\U000102F7\u3002\u200D',
    u'\U0001D7F5\u9681\u2BEE\uFF0E\u180D\u200C',
    u'9\u9681\u2BEE.\u180D\u200C',
    u'\u00DF\u200C\uAAF6\u18A5.\u22B6\u2D21\u2D16',
    u'ss\u200C\uAAF6\u18A5.\u22B6\u2D21\u2D16',
    u'\u00DF\u200C\uAAF6\u18A5\uFF0E\u22B6\u2D21\u2D16',
    u'ss\u200C\uAAF6\u18A5\uFF0E\u22B6\u2D21\u2D16',
    u'\U00010A57\u200D\u3002\u2D09\u2D15',
    u'\U00010A57\u200D\uFF61\u2D09\u2D15',
    u'\U0001D7CF\U0001DA19\u2E16.\u200D',
    u'1\U0001DA19\u2E16.\u200D',
    u'\U0001D7E04\U000E01D7\U0001D23B\uFF0E\u200D\U000102F5\u26E7\u200D',
    u'84\U000E01D7\U0001D23B.\u200D\U000102F5\u26E7\u200D',
]

def unicode_fixup(string):
    """Replace backslash-u-XXXX with appropriate unicode characters."""
    return _RE_SURROGATE.sub(lambda match: unichr(
        (ord(match.group(0)[0]) - 0xd800) * 0x400 +
        ord(match.group(0)[1]) - 0xdc00 + 0x10000),
        _RE_UNICODE.sub(lambda match: unichr(int(match.group(1), 16)), string))


def parse_idna_test_table(inputstream):
    """Parse IdnaTest.txt and return a list of tuples."""
    tests = []
    for lineno, line in enumerate(inputstream):
        line = line.decode("utf-8").strip()
        if "#" in line:
            line = line.split("#", 1)[0]
        if not line:
            continue
        tests.append((lineno + 1, tuple(field.strip()
            for field in line.split(u";"))))
    return tests


class TestIdnaTest(unittest.TestCase):
    """Run one of the IdnaTest.txt test lines."""
    def __init__(self, lineno=None, fields=None):
        super(TestIdnaTest, self).__init__()
        self.lineno = lineno
        self.fields = fields

    def id(self):
        return "%s.%d" % (super(TestIdnaTest, self).id(), self.lineno)

    def shortDescription(self):
        if not self.fields:
            return ""
        return "IdnaTest.txt line %d: %r" % (self.lineno,
            u"; ".join(self.fields))

    def runTest(self):
        if not self.fields:
            return
        try:
            types, source, to_unicode, to_ascii = (unicode_fixup(field)
                for field in self.fields[:4])
            if (unicode_fixup(u"\\uD804\\uDC39") in source and
                    sys.version_info[0] < 3):
                raise unittest.SkipTest(
                    "Python 2's Unicode support is too old for this test")
        except ValueError:
            raise unittest.SkipTest(
                "Test requires Python wide Unicode support")
        if source in _SKIP_TESTS:
            return
        if not to_unicode:
            to_unicode = source
        if not to_ascii:
            to_ascii = to_unicode
        nv8 = (len(self.fields) > 4 and self.fields[4])
        try:
            output = idna.decode(source, uts46=True, strict=True)
            if to_unicode[0] == u"[":
                self.fail("decode() did not emit required error {0} for {1}".format(to_unicode, repr(source)))
            self.assertEqual(output, to_unicode, "unexpected decode() output")
        except (idna.IDNAError, UnicodeError, ValueError) as exc:
            if unicode(exc).startswith(u"Unknown"):
                raise unittest.SkipTest("Test requires support for a newer"
                    " version of Unicode than this Python supports")
            if to_unicode[0] != u"[" and not nv8:
                raise
        for transitional in {
                u"B": (True, False),
                u"T": (True,),
                u"N": (False,),
                }[types]:
            try:
                output = idna.encode(source, uts46=True, strict=True,
                    transitional=transitional).decode("ascii")
                if to_ascii[0] == u"[":
                    self.fail(
                        "encode(transitional={0}) did not emit required error {1} for {2}".
                        format(transitional, to_ascii, repr(source)))
                self.assertEqual(output, to_ascii,
                    "unexpected encode(transitional={0}) output".
                    format(transitional))
            except (idna.IDNAError, UnicodeError, ValueError) as exc:
                if unicode(exc).startswith(u"Unknown"):
                    raise unittest.SkipTest("Test requires support for a newer"
                                            " version of Unicode than this Python supports")
                if to_ascii[0] != u"[" and not nv8:
                    raise


def load_tests(loader, tests, pattern):
    """Create a suite of all the individual tests."""
    suite = unittest.TestSuite()
    with gzip.open(os.path.join(os.path.dirname(__file__),
            "IdnaTest.txt.gz"), "rb") as tests_file:
        suite.addTests(TestIdnaTest(lineno, fields)
            for lineno, fields in parse_idna_test_table(tests_file))
    return suite
