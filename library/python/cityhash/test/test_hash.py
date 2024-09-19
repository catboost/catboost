import cityhash
import yatest.common as yc


def test_data1():
    path = yc.test_source_path('test_data_1')
    assert cityhash.filehash128high64(path) == 7988416450439545646
    assert cityhash.filehash64(path) == 3870718426980431168


def test_data2():
    path = yc.test_source_path('test_data_2')
    assert cityhash.filehash128high64(path) == 6406929700245303324
    assert cityhash.filehash64(path) == 5442203742544462300


def test_hash64():
    assert cityhash.hash64('0123456789') == 12631666426400459317
    assert cityhash.hash64('abacaba') == 12549660514692179516
    assert cityhash.hash64('') == 11160318154034397263


def test_hash128():
    assert cityhash.hash128('0123456789') == (5058155155124848858, 17393408752974585106)
    assert cityhash.hash128('abacaba') == (4599629899855957408, 4104518117632749755)
    assert cityhash.hash128('') == (18085479540095642321, 11079402499652051579)


def test_hash64seed():
    assert cityhash.hash64seed('', 0) == 0
    assert cityhash.hash64seed('', 117) == 7102524123839304709
    assert cityhash.hash64seed('test', 12345) == 14900027982776226655


# def test_hash_large_encrypted_files():
#     path = yc.source_path('devtools/dummy_arcadia/arc_special_files/encrypted/encrypted_files/1mb_file.txt')
#     try:
#         cityhash.filehash64(path)
#     except RuntimeError:
#         # This is expected to fail on local and in Sandbox but run normally in distbuild
#         # Replace with 'with raises' once YA-1099 is done
#         pass
