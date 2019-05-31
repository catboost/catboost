import xml.etree.cElementTree as ET

from build.plugins import ssqls


example = '''\
<?xml version="1.0" encoding="utf-8"?>
<DbObject>
    <include path="a.ssqls">&lt;a.h&gt;</include>
    <include>"b.h"</include>

    <ancestors>
        <ancestor path="c.ssqls"/>
    </ancestors>
</DbObject>
'''


def test_include_parser():
    doc = ET.fromstring(example)
    xmls, headers = ssqls.SSQLSParser.parse_doc(doc)
    assert headers == ['a.h', 'b.h']
    assert xmls == ['a.ssqls', 'c.ssqls']
