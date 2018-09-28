import library.python.resource as rs

text = b'na gorshke sidel korol\n'


def test_simple():
    assert(rs.find(b'/qw.txt') == text)


def test_iter():
    assert set(rs.iterkeys()).issuperset({b'/qw.txt', b'/prefix/1.txt', b'/prefix/2.txt'})
    assert set(rs.iterkeys(prefix=b'/prefix/')) == {b'/prefix/1.txt', b'/prefix/2.txt'}
    assert set(rs.iterkeys(prefix=b'/prefix/', strip_prefix=True)) == {b'1.txt', b'2.txt'}
    assert set(rs.iteritems(prefix=b'/prefix')) == {
        (b'/prefix/1.txt', text),
        (b'/prefix/2.txt', text),
    }
    assert set(rs.iteritems(prefix=b'/prefix', strip_prefix=True)) == {
        (b'/1.txt', text),
        (b'/2.txt', text),
    }
