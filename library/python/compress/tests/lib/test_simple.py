from io import open

import library.python.compress as lpc


def test_simple():
    data = b'12345' * 10000000

    with open('data', 'wb') as f:
        f.write(data)

    lpc.compress('data', 'data.zstd_1', threads=5)
    lpc.decompress('data.zstd_1', 'data.new', threads=3)

    with open('data.new', 'rb') as f:
        assert f.read() == data
