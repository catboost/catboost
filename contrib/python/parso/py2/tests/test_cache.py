"""
Test all things related to the ``jedi.cache`` module.
"""

import os
import os.path

import pytest
import time

from parso.cache import (_CACHED_FILE_MAXIMUM_SURVIVAL, _VERSION_TAG,
                         _get_cache_clear_lock, _get_hashed_path,
                         _load_from_file_system, _NodeCacheItem,
                         _remove_cache_and_update_lock, _save_to_file_system,
                         load_module, parser_cache, try_to_save_module)
from parso._compatibility import is_pypy, PermissionError
from parso import load_grammar
from parso import cache
from parso import file_io
from parso import parse

skip_pypy = pytest.mark.skipif(
    is_pypy,
    reason="pickling in pypy is slow, since we don't pickle,"
           "we never go into path of auto-collecting garbage"
)


@pytest.fixture()
def isolated_parso_cache(monkeypatch, tmpdir):
    """Set `parso.cache._default_cache_path` to a temporary directory
    during the test. """
    cache_path = str(os.path.join(str(tmpdir), "__parso_cache"))
    monkeypatch.setattr(cache, '_default_cache_path', cache_path)
    monkeypatch.setattr(cache, '_get_default_cache_path', lambda *args, **kwargs: cache_path)
    return cache_path


@pytest.mark.skip("SUBBOTNIK-2721 Disable load cache from disk")
def test_modulepickling_change_cache_dir(tmpdir):
    """
    ParserPickling should not save old cache when cache_directory is changed.

    See: `#168 <https://github.com/davidhalter/jedi/pull/168>`_
    """
    dir_1 = str(tmpdir.mkdir('first'))
    dir_2 = str(tmpdir.mkdir('second'))

    item_1 = _NodeCacheItem('bla', [])
    item_2 = _NodeCacheItem('bla', [])
    path_1 = 'fake path 1'
    path_2 = 'fake path 2'

    hashed_grammar = load_grammar()._hashed
    _save_to_file_system(hashed_grammar, path_1, item_1, cache_path=dir_1)
    parser_cache.clear()
    cached = load_stored_item(hashed_grammar, path_1, item_1, cache_path=dir_1)
    assert cached == item_1.node

    _save_to_file_system(hashed_grammar, path_2, item_2, cache_path=dir_2)
    cached = load_stored_item(hashed_grammar, path_1, item_1, cache_path=dir_2)
    assert cached is None


def load_stored_item(hashed_grammar, path, item, cache_path):
    """Load `item` stored at `path` in `cache`."""
    item = _load_from_file_system(hashed_grammar, path, item.change_time - 1, cache_path)
    return item


@pytest.mark.usefixtures("isolated_parso_cache")
def test_modulepickling_simulate_deleted_cache(tmpdir):
    """
    Tests loading from a cache file after it is deleted.
    According to macOS `dev docs`__,

        Note that the system may delete the Caches/ directory to free up disk
        space, so your app must be able to re-create or download these files as
        needed.

    It is possible that other supported platforms treat cache files the same
    way.

    __ https://developer.apple.com/library/content/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/FileSystemOverview/FileSystemOverview.html
    """
    grammar = load_grammar()
    module = 'fake parser'

    # Create the file
    path = tmpdir.dirname + '/some_path'
    with open(path, 'w'):
        pass
    io = file_io.FileIO(path)

    try_to_save_module(grammar._hashed, io, module, lines=[])
    assert load_module(grammar._hashed, io) == module

    os.unlink(_get_hashed_path(grammar._hashed, path))
    parser_cache.clear()

    cached2 = load_module(grammar._hashed, io)
    assert cached2 is None


@pytest.mark.skip
def test_cache_limit():
    def cache_size():
        return sum(len(v) for v in parser_cache.values())

    try:
        parser_cache.clear()
        future_node_cache_item = _NodeCacheItem('bla', [], change_time=time.time() + 10e6)
        old_node_cache_item = _NodeCacheItem('bla', [], change_time=time.time() - 10e4)
        parser_cache['some_hash_old'] = {
            '/path/%s' % i: old_node_cache_item for i in range(300)
        }
        parser_cache['some_hash_new'] = {
            '/path/%s' % i: future_node_cache_item for i in range(300)
        }
        assert cache_size() == 600
        parse('somecode', cache=True, path='/path/somepath')
        assert cache_size() == 301
    finally:
        parser_cache.clear()


class _FixedTimeFileIO(file_io.KnownContentFileIO):
    def __init__(self, path, content, last_modified):
        super(_FixedTimeFileIO, self).__init__(path, content)
        self._last_modified = last_modified

    def get_last_modified(self):
        return self._last_modified


@pytest.mark.skip
@pytest.mark.parametrize('diff_cache', [False, True])
@pytest.mark.parametrize('use_file_io', [False, True])
def test_cache_last_used_update(diff_cache, use_file_io):
    p = '/path/last-used'
    parser_cache.clear()  # Clear, because then it's easier to find stuff.
    parse('somecode', cache=True, path=p)
    node_cache_item = next(iter(parser_cache.values()))[p]
    now = time.time()
    assert node_cache_item.last_used < now

    if use_file_io:
        f = _FixedTimeFileIO(p, 'code', node_cache_item.last_used - 10)
        parse(file_io=f, cache=True, diff_cache=diff_cache)
    else:
        parse('somecode2', cache=True, path=p, diff_cache=diff_cache)

    node_cache_item = next(iter(parser_cache.values()))[p]
    assert now < node_cache_item.last_used < time.time()


@skip_pypy
def test_inactive_cache(tmpdir, isolated_parso_cache):
    parser_cache.clear()
    test_subjects = "abcdef"
    for path in test_subjects:
        parse('somecode', cache=True, path=os.path.join(str(tmpdir), path))
    raw_cache_path = os.path.join(isolated_parso_cache, _VERSION_TAG)
    assert os.path.exists(raw_cache_path)
    paths = os.listdir(raw_cache_path)
    a_while_ago = time.time() - _CACHED_FILE_MAXIMUM_SURVIVAL
    old_paths = set()
    for path in paths[:len(test_subjects) // 2]:  # make certain number of paths old
        os.utime(os.path.join(raw_cache_path, path), (a_while_ago, a_while_ago))
        old_paths.add(path)
    # nothing should be cleared while the lock is on
    assert os.path.exists(_get_cache_clear_lock().path)
    _remove_cache_and_update_lock()  # it shouldn't clear anything
    assert len(os.listdir(raw_cache_path)) == len(test_subjects)
    assert old_paths.issubset(os.listdir(raw_cache_path))

    os.utime(_get_cache_clear_lock().path, (a_while_ago, a_while_ago))
    _remove_cache_and_update_lock()
    assert len(os.listdir(raw_cache_path)) == len(test_subjects) // 2
    assert not old_paths.intersection(os.listdir(raw_cache_path))


@pytest.mark.skip
@skip_pypy
def test_permission_error(monkeypatch):
    def save(*args, **kwargs):
        was_called[0] = True  # Python 2... Use nonlocal instead
        raise PermissionError

    was_called = [False]

    monkeypatch.setattr(cache, '_save_to_file_system', save)
    with pytest.warns(Warning):
        parse(path=__file__, cache=True, diff_cache=True)
    assert was_called[0]
