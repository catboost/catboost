from io import BytesIO
import sys

import pytest

from pandas.io.msgpack import ExtType, OutOfData, Unpacker, packb


class TestUnpack(object):

    def test_unpack_array_header_from_file(self):
        f = BytesIO(packb([1, 2, 3, 4]))
        unpacker = Unpacker(f)
        assert unpacker.read_array_header() == 4
        assert unpacker.unpack() == 1
        assert unpacker.unpack() == 2
        assert unpacker.unpack() == 3
        assert unpacker.unpack() == 4
        msg = "No more data to unpack"
        with pytest.raises(OutOfData, match=msg):
            unpacker.unpack()

    def test_unpacker_hook_refcnt(self):
        if not hasattr(sys, 'getrefcount'):
            pytest.skip('no sys.getrefcount()')
        result = []

        def hook(x):
            result.append(x)
            return x

        basecnt = sys.getrefcount(hook)

        up = Unpacker(object_hook=hook, list_hook=hook)

        assert sys.getrefcount(hook) >= basecnt + 2

        up.feed(packb([{}]))
        up.feed(packb([{}]))
        assert up.unpack() == [{}]
        assert up.unpack() == [{}]
        assert result == [{}, [{}], {}, [{}]]

        del up

        assert sys.getrefcount(hook) == basecnt

    def test_unpacker_ext_hook(self):
        class MyUnpacker(Unpacker):

            def __init__(self):
                super(MyUnpacker, self).__init__(ext_hook=self._hook,
                                                 encoding='utf-8')

            def _hook(self, code, data):
                if code == 1:
                    return int(data)
                else:
                    return ExtType(code, data)

        unpacker = MyUnpacker()
        unpacker.feed(packb({'a': 1}, encoding='utf-8'))
        assert unpacker.unpack() == {'a': 1}
        unpacker.feed(packb({'a': ExtType(1, b'123')}, encoding='utf-8'))
        assert unpacker.unpack() == {'a': 123}
        unpacker.feed(packb({'a': ExtType(2, b'321')}, encoding='utf-8'))
        assert unpacker.unpack() == {'a': ExtType(2, b'321')}
