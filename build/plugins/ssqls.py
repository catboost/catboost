import _import_wrapper as iw
from _common import resolve_src, resolve_includes


class SSQLSParser(object):
    def __init__(self, path, unit):
        s = resolve_src(unit, path)
        assert s.startswith('$S/') and s.endswith('.ssqls'), s
        h = '$B/' + s[3:-6] + '.h'

        import xml.etree.cElementTree as ET
        try:
            doc = ET.parse(path)
        except ET.ParseError as e:
            unit.message(['error', 'malformed XML {}: {}'.format(path, e)])
            doc = ET.Element('DbObject')
        xmls, headers = self.parse_doc(doc)
        self._includes = resolve_includes(unit, s, xmls)
        self._induced = {'cpp': [h], 'h': resolve_includes(unit, h, headers)}

    @staticmethod
    def parse_doc(doc):
        nodes = doc.findall('include')
        headers = [e.text.strip('<>""') for e in nodes]
        nodes += doc.findall('ancestors/ancestor')
        xmls = filter(None, (e.get('path') for e in nodes))
        return xmls, headers

    def includes(self):
        return self._includes

    def induced_deps(self):
        return self._induced


def init():
    iw.addparser('ssqls', SSQLSParser)
