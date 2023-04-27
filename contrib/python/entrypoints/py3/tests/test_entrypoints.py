# Copyright (c) Thomas Kluyver and contributors
# Distributed under the terms of the MIT license; see LICENSE file.

import os.path as osp
import pytest
import warnings
from zipfile import ZipFile

import entrypoints

import yatest

samples_dir = yatest.common.source_path('contrib/python/entrypoints/py3/tests/samples')


sample_path = [
    osp.join(samples_dir, 'packages1'),
    osp.join(samples_dir, 'packages1', 'baz-0.3.egg'),
    osp.join(samples_dir, 'packages2'),
    osp.join(samples_dir, 'packages2', 'qux-0.4.egg'),
]

def test_iter_files_distros():
    result = entrypoints.iter_files_distros(path=sample_path)
    # the sample_path has 4 unique items so iter_files_distros returns 4 tuples
    assert len(list(result)) == 4

    # testing a development, egg aka installed with pip install -e .
    # these don't have version info in the .egg-info directory name
    # (eg dev-0.0.1.egg-info)
    path_with_dev = [osp.join(samples_dir, 'packages4')]
    result = entrypoints.iter_files_distros(path=path_with_dev)
    assert len(list(result)) == 1

    # duplicate dev versions should still return one result
    path_with_dev_duplicates = path_with_dev * 2
    result = entrypoints.iter_files_distros(path=path_with_dev_duplicates)
    assert len(list(result)) == 1

def test_get_group_all():
    group = entrypoints.get_group_all('entrypoints.test1', sample_path)
    print(group)
    assert len(group) == 5
    assert {ep.name for ep in group} == {'abc', 'rew', 'opo', 'njn'}

def test_get_group_named():
    group = entrypoints.get_group_named('entrypoints.test1', sample_path)
    print(group)
    assert len(group) == 4
    assert group['abc'].module_name == 'foo'
    assert group['abc'].object_name == 'abc'

def test_get_single():
    ep = entrypoints.get_single('entrypoints.test1', 'abc', sample_path)
    assert ep.module_name == 'foo'
    assert ep.object_name == 'abc'

    ep2 = entrypoints.get_single('entrypoints.test1', 'njn', sample_path)
    assert ep2.module_name == 'qux.extn'
    assert ep2.object_name == 'Njn.load'

def test_dot_prefix():
    ep = entrypoints.get_single('blogtool.parsers', '.rst', sample_path)
    assert ep.object_name == 'SomeClass.some_classmethod'
    assert ep.extras == ['reST']

    group = entrypoints.get_group_named('blogtool.parsers', sample_path)
    assert set(group.keys()) == {'.rst'}

def test_case_sensitive():
    group = entrypoints.get_group_named('test.case_sensitive', sample_path)
    assert set(group.keys()) == {'Ptangle', 'ptangle'}

def test_load_zip(tmpdir):
    whl_file = str(tmpdir / 'parmesan-1.2.whl')
    with ZipFile(whl_file, 'w') as whl:
        whl.writestr('parmesan-1.2.dist-info/entry_points.txt',
                     b'[entrypoints.test.inzip]\na = edam:gouda')
        whl.writestr('gruyere-2!1b4.dev0.egg-info/entry_points.txt',
                     b'[entrypoints.test.inzip]\nb = wensleydale:gouda')

    ep = entrypoints.get_single('entrypoints.test.inzip', 'a', [str(whl_file)])
    assert ep.module_name == 'edam'
    assert ep.object_name == 'gouda'
    assert ep.distro.name == 'parmesan'
    assert ep.distro.version == '1.2'

    ep2 = entrypoints.get_single('entrypoints.test.inzip', 'b', [str(whl_file)])
    assert ep2.module_name == 'wensleydale'
    assert ep2.object_name == 'gouda'
    assert ep2.distro.name == 'gruyere'
    assert ep2.distro.version == '2!1b4.dev0'

def test_load():
    ep = entrypoints.EntryPoint('get_ep', 'entrypoints', 'get_single', None)
    obj = ep.load()
    assert obj is entrypoints.get_single

    # The object part is optional (e.g. pytest plugins use just a module ref)
    ep = entrypoints.EntryPoint('ep_mod', 'entrypoints', None)
    obj = ep.load()
    assert obj is entrypoints

def test_bad():
    bad_path = [osp.join(samples_dir, 'packages3')]

    with warnings.catch_warnings(record=True) as w:
        group = entrypoints.get_group_named('entrypoints.test1', bad_path)

    assert 'bad' not in group
    assert len(w) == 1

    with warnings.catch_warnings(record=True) as w2, \
            pytest.raises(entrypoints.NoSuchEntryPoint):
        ep = entrypoints.get_single('entrypoints.test1', 'bad')

    assert len(w) == 1

def test_missing():
    with pytest.raises(entrypoints.NoSuchEntryPoint) as ec:
        entrypoints.get_single('no.such.group', 'no_such_name', sample_path)

    assert ec.value.group == 'no.such.group'
    assert ec.value.name == 'no_such_name'

def test_parse():
    ep = entrypoints.EntryPoint.from_string(
        'some.module:some.attr [extra1,extra2]', 'foo'
    )
    assert ep.module_name == 'some.module'
    assert ep.object_name == 'some.attr'
    assert ep.extras == ['extra1', 'extra2']

def test_parse_bad():
    with pytest.raises(entrypoints.BadEntryPoint):
        entrypoints.EntryPoint.from_string("this won't work", 'foo')
