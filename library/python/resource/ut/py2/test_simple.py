import library.python.resource as rs

text = 'na gorshke sidel korol\n'


def test_simple():
    assert rs.find('/qw.txt') == text


def test_iter():
    assert set(rs.iterkeys()).issuperset({'/qw.txt', '/prefix/1.txt', '/prefix/2.txt'})
    assert set(rs.iterkeys(prefix='/prefix/')) == {'/prefix/1.txt', '/prefix/2.txt'}
    assert set(rs.iterkeys(prefix='/prefix/', strip_prefix=True)) == {'1.txt', '2.txt'}
    assert set(rs.iteritems(prefix='/prefix')) == {
        ('/prefix/1.txt', text),
        ('/prefix/2.txt', text),
    }
    assert set(rs.iteritems(prefix='/prefix', strip_prefix=True)) == {
        ('/1.txt', text),
        ('/2.txt', text),
    }
