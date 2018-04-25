from build.plugins import flatc


def test_include_parser():
    text = '''
aaaaa
include "incl1.fbs";
include   "incl2.fbs";
// include "none.fbs";
'''
    includes, induced = flatc.FlatcParser.parse_includes(text.split('\n'))
    assert includes == ['incl1.fbs', 'incl2.fbs', ]
    assert induced == ['incl1.fbs.h', 'incl2.fbs.h', ]


def test_include_parser64():
    text = '''
aaaaa
include "incl1.fbs64";
include   "incl2.fbs64";
// include "none.fbs64";
'''
    includes, induced = flatc.FlatcParser64.parse_includes(text.split('\n'))
    assert includes == ['incl1.fbs64', 'incl2.fbs64', ]
    assert induced == ['incl1.fbs64.h', 'incl2.fbs64.h', ]
