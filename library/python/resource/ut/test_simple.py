import six  # noqa

import library.python.resource as rs

text = b"na gorshke sidel korol\n"


def test_find():
    assert rs.find("/qw.txt") == text


def test_iter():
    assert set(rs.iterkeys()).issuperset({"/qw.txt", "/prefix/1.txt", "/prefix/2.txt"})
    assert set(rs.iterkeys(prefix="/prefix/")) == {"/prefix/1.txt", "/prefix/2.txt"}
    assert set(rs.iterkeys(prefix="/prefix/", strip_prefix=True)) == {"1.txt", "2.txt"}
    assert set(rs.iteritems(prefix="/prefix")) == {
        ("/prefix/1.txt", text),
        ("/prefix/2.txt", text),
    }
    assert set(rs.iteritems(prefix="/prefix", strip_prefix=True)) == {
        ("/1.txt", text),
        ("/2.txt", text),
    }


def test_resfs_files():
    metadata_path = "contrib/python/py/py{}/.dist-info/METADATA".format("2" if six.PY2 else "3")
    assert metadata_path in set(rs.resfs_files())


def test_resfs_read():
    metadata_path = "contrib/python/py/py{}/.dist-info/METADATA".format("2" if six.PY2 else "3")
    assert "Metadata-Version" in rs.resfs_read(metadata_path).decode("utf-8")
