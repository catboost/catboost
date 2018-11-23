import pytest

import build.plugins._common as pc


def test_sort_by_keywords():
    keywords = {'KEY1': 2, 'KEY2': 0, 'KEY3': 1}
    args = 'aaaa bbbb KEY2 KEY1 kkk10 kkk11 ccc ddd KEY3 kkk3 eee'.split()
    flat, spec = pc.sort_by_keywords(keywords, args)
    assert flat == ['aaaa', 'bbbb', 'ccc', 'ddd', 'eee']
    assert spec == {'KEY1': ['kkk10', 'kkk11'], 'KEY2': True, 'KEY3': ['kkk3']}

    keywords = {'KEY1': 0, 'KEY2': 4}
    args = 'aaaa KEY2 eee'.split()
    flat, spec = pc.sort_by_keywords(keywords, args)
    assert flat == ['aaaa']
    assert spec == {'KEY2': ['eee']}

    keywords = {'KEY1': 2, 'KEY2': 2}
    args = 'KEY1 k10 KEY2 k20 KEY1 k11 KEY2 k21 KEY1 k13'.split()
    flat, spec = pc.sort_by_keywords(keywords, args)
    assert flat == []
    assert spec == {'KEY1': ['k10', 'k11', 'k13'], 'KEY2': ['k20', 'k21']}


def test_filter_out_by_keyword():
    assert pc.filter_out_by_keyword([], 'A') == []
    assert pc.filter_out_by_keyword(['x'], 'A') == ['x']
    assert pc.filter_out_by_keyword(['x', 'A'], 'A') == ['x']
    assert pc.filter_out_by_keyword(['x', 'A', 'B'], 'A') == ['x']
    assert pc.filter_out_by_keyword(['x', 'A', 'B', 'y'], 'A') == ['x', 'y']
    assert pc.filter_out_by_keyword(['x', 'A', 'A', 'y'], 'A') == ['x', 'y']
    assert pc.filter_out_by_keyword(['x', 'A', 'A', 'A'], 'A') == ['x']
    assert pc.filter_out_by_keyword(['x', 'A', 'A', 'A', 'B', 'y'], 'A') == ['x', 'y']
    assert pc.filter_out_by_keyword(['x', 'A', 'A', 'A', 'B', 'y', 'A'], 'A') == ['x', 'y']
    assert pc.filter_out_by_keyword(['x', 'A', 'A', 'A', 'B', 'y', 'A', 'F', 'z'], 'A') == ['x', 'y', 'z']


test_data = [
    [[1, 2, 3], 1, [[1], [2], [3]]],
    [[1, 2, 3], 2, [[1, 2], [3]]],
    [[1, 2, 3, 4], 2, [[1, 2], [3, 4]]],
    [[1], 5, [[1]]],
]


@pytest.mark.parametrize('lst, chunk_size, expected', test_data, ids=[str(num + 1) for num in range(len(test_data))])
def test_generate_chunks(lst, chunk_size, expected):
    assert list(pc.generate_chunks(lst, chunk_size)) == expected
