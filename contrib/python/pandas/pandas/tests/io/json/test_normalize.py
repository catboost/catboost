import json

import numpy as np
import pytest

from pandas import DataFrame, Index, compat
import pandas.util.testing as tm

from pandas.io.json import json_normalize
from pandas.io.json.normalize import nested_to_record


@pytest.fixture
def deep_nested():
    # deeply nested data
    return [{'country': 'USA',
             'states': [{'name': 'California',
                         'cities': [{'name': 'San Francisco',
                                     'pop': 12345},
                                    {'name': 'Los Angeles',
                                     'pop': 12346}]
                         },
                        {'name': 'Ohio',
                         'cities': [{'name': 'Columbus',
                                     'pop': 1234},
                                    {'name': 'Cleveland',
                                     'pop': 1236}]}
                        ]
             },
            {'country': 'Germany',
             'states': [{'name': 'Bayern',
                         'cities': [{'name': 'Munich', 'pop': 12347}]
                         },
                        {'name': 'Nordrhein-Westfalen',
                         'cities': [{'name': 'Duesseldorf', 'pop': 1238},
                                    {'name': 'Koeln', 'pop': 1239}]}
                        ]
             }
            ]


@pytest.fixture
def state_data():
    return [
        {'counties': [{'name': 'Dade', 'population': 12345},
                      {'name': 'Broward', 'population': 40000},
                      {'name': 'Palm Beach', 'population': 60000}],
         'info': {'governor': 'Rick Scott'},
         'shortname': 'FL',
         'state': 'Florida'},
        {'counties': [{'name': 'Summit', 'population': 1234},
                      {'name': 'Cuyahoga', 'population': 1337}],
         'info': {'governor': 'John Kasich'},
         'shortname': 'OH',
         'state': 'Ohio'}]


@pytest.fixture
def author_missing_data():
    return [
        {'info': None},
        {'info':
            {'created_at': '11/08/1993', 'last_updated': '26/05/2012'},
            'author_name':
         {'first': 'Jane', 'last_name': 'Doe'}
         }]


class TestJSONNormalize(object):

    def test_simple_records(self):
        recs = [{'a': 1, 'b': 2, 'c': 3},
                {'a': 4, 'b': 5, 'c': 6},
                {'a': 7, 'b': 8, 'c': 9},
                {'a': 10, 'b': 11, 'c': 12}]

        result = json_normalize(recs)
        expected = DataFrame(recs)

        tm.assert_frame_equal(result, expected)

    def test_simple_normalize(self, state_data):
        result = json_normalize(state_data[0], 'counties')
        expected = DataFrame(state_data[0]['counties'])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, 'counties')

        expected = []
        for rec in state_data:
            expected.extend(rec['counties'])
        expected = DataFrame(expected)

        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, 'counties', meta='state')
        expected['state'] = np.array(['Florida', 'Ohio']).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_empty_array(self):
        result = json_normalize([])
        expected = DataFrame()
        tm.assert_frame_equal(result, expected)

    def test_simple_normalize_with_separator(self, deep_nested):
        # GH 14883
        result = json_normalize({'A': {'A': 1, 'B': 2}})
        expected = DataFrame([[1, 2]], columns=['A.A', 'A.B'])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({'A': {'A': 1, 'B': 2}}, sep='_')
        expected = DataFrame([[1, 2]], columns=['A_A', 'A_B'])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize({'A': {'A': 1, 'B': 2}}, sep=u'\u03c3')
        expected = DataFrame([[1, 2]], columns=[u'A\u03c3A', u'A\u03c3B'])
        tm.assert_frame_equal(result.reindex_like(expected), expected)

        result = json_normalize(deep_nested, ['states', 'cities'],
                                meta=['country', ['states', 'name']],
                                sep='_')
        expected = Index(['name', 'pop',
                          'country', 'states_name']).sort_values()
        assert result.columns.sort_values().equals(expected)

    def test_value_array_record_prefix(self):
        # GH 21536
        result = json_normalize({'A': [1, 2]}, 'A', record_prefix='Prefix.')
        expected = DataFrame([[1], [2]], columns=['Prefix.0'])
        tm.assert_frame_equal(result, expected)

    def test_nested_object_record_path(self):
        # GH 22706
        data = {'state': 'Florida',
                'info': {
                    'governor': 'Rick Scott',
                    'counties': [{'name': 'Dade', 'population': 12345},
                                 {'name': 'Broward', 'population': 40000},
                                 {'name': 'Palm Beach', 'population': 60000}]}}
        result = json_normalize(data, record_path=["info", "counties"])
        expected = DataFrame([['Dade', 12345],
                              ['Broward', 40000],
                              ['Palm Beach', 60000]],
                             columns=['name', 'population'])
        tm.assert_frame_equal(result, expected)

    def test_more_deeply_nested(self, deep_nested):

        result = json_normalize(deep_nested, ['states', 'cities'],
                                meta=['country', ['states', 'name']])
        # meta_prefix={'states': 'state_'})

        ex_data = {'country': ['USA'] * 4 + ['Germany'] * 3,
                   'states.name': ['California', 'California', 'Ohio', 'Ohio',
                                   'Bayern', 'Nordrhein-Westfalen',
                                   'Nordrhein-Westfalen'],
                   'name': ['San Francisco', 'Los Angeles', 'Columbus',
                            'Cleveland', 'Munich', 'Duesseldorf', 'Koeln'],
                   'pop': [12345, 12346, 1234, 1236, 12347, 1238, 1239]}

        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_shallow_nested(self):
        data = [{'state': 'Florida',
                 'shortname': 'FL',
                 'info': {
                     'governor': 'Rick Scott'
                 },
                 'counties': [{'name': 'Dade', 'population': 12345},
                              {'name': 'Broward', 'population': 40000},
                              {'name': 'Palm Beach', 'population': 60000}]},
                {'state': 'Ohio',
                 'shortname': 'OH',
                 'info': {
                     'governor': 'John Kasich'
                 },
                 'counties': [{'name': 'Summit', 'population': 1234},
                              {'name': 'Cuyahoga', 'population': 1337}]}]

        result = json_normalize(data, 'counties',
                                ['state', 'shortname',
                                 ['info', 'governor']])
        ex_data = {'name': ['Dade', 'Broward', 'Palm Beach', 'Summit',
                            'Cuyahoga'],
                   'state': ['Florida'] * 3 + ['Ohio'] * 2,
                   'shortname': ['FL', 'FL', 'FL', 'OH', 'OH'],
                   'info.governor': ['Rick Scott'] * 3 + ['John Kasich'] * 2,
                   'population': [12345, 40000, 60000, 1234, 1337]}
        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_meta_name_conflict(self):
        data = [{'foo': 'hello',
                 'bar': 'there',
                 'data': [{'foo': 'something', 'bar': 'else'},
                          {'foo': 'something2', 'bar': 'else2'}]}]

        msg = (r"Conflicting metadata name (foo|bar),"
               " need distinguishing prefix")
        with pytest.raises(ValueError, match=msg):
            json_normalize(data, 'data', meta=['foo', 'bar'])

        result = json_normalize(data, 'data', meta=['foo', 'bar'],
                                meta_prefix='meta')

        for val in ['metafoo', 'metabar', 'foo', 'bar']:
            assert val in result

    def test_meta_parameter_not_modified(self):
        # GH 18610
        data = [{'foo': 'hello',
                 'bar': 'there',
                 'data': [{'foo': 'something', 'bar': 'else'},
                          {'foo': 'something2', 'bar': 'else2'}]}]

        COLUMNS = ['foo', 'bar']
        result = json_normalize(data, 'data', meta=COLUMNS,
                                meta_prefix='meta')

        assert COLUMNS == ['foo', 'bar']
        for val in ['metafoo', 'metabar', 'foo', 'bar']:
            assert val in result

    def test_record_prefix(self, state_data):
        result = json_normalize(state_data[0], 'counties')
        expected = DataFrame(state_data[0]['counties'])
        tm.assert_frame_equal(result, expected)

        result = json_normalize(state_data, 'counties',
                                meta='state',
                                record_prefix='county_')

        expected = []
        for rec in state_data:
            expected.extend(rec['counties'])
        expected = DataFrame(expected)
        expected = expected.rename(columns=lambda x: 'county_' + x)
        expected['state'] = np.array(['Florida', 'Ohio']).repeat([3, 2])

        tm.assert_frame_equal(result, expected)

    def test_non_ascii_key(self):
        if compat.PY3:
            testjson = (
                b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},' +
                b'{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'
            ).decode('utf8')
        else:
            testjson = ('[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},'
                        '{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]')

        testdata = {
            u'sub.A': [1, 3],
            u'sub.B': [2, 4],
            b"\xc3\x9cnic\xc3\xb8de".decode('utf8'): [0, 1]
        }
        expected = DataFrame(testdata)

        result = json_normalize(json.loads(testjson))
        tm.assert_frame_equal(result, expected)

    def test_missing_field(self, author_missing_data):
        # GH20030:
        result = json_normalize(author_missing_data)
        ex_data = [
            {'info': np.nan,
             'author_name.first': np.nan,
             'author_name.last_name': np.nan,
             'info.created_at': np.nan,
             'info.last_updated': np.nan},
            {'info': None,
             'author_name.first': 'Jane',
             'author_name.last_name': 'Doe',
             'info.created_at': '11/08/1993',
             'info.last_updated': '26/05/2012'}
        ]
        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)


class TestNestedToRecord(object):

    def test_flat_stays_flat(self):
        recs = [dict(flat1=1, flat2=2),
                dict(flat1=3, flat2=4),
                ]

        result = nested_to_record(recs)
        expected = recs
        assert result == expected

    def test_one_level_deep_flattens(self):
        data = dict(flat1=1,
                    dict1=dict(c=1, d=2))

        result = nested_to_record(data)
        expected = {'dict1.c': 1,
                    'dict1.d': 2,
                    'flat1': 1}

        assert result == expected

    def test_nested_flattens(self):
        data = dict(flat1=1,
                    dict1=dict(c=1, d=2),
                    nested=dict(e=dict(c=1, d=2),
                                d=2))

        result = nested_to_record(data)
        expected = {'dict1.c': 1,
                    'dict1.d': 2,
                    'flat1': 1,
                    'nested.d': 2,
                    'nested.e.c': 1,
                    'nested.e.d': 2}

        assert result == expected

    def test_json_normalize_errors(self):
        # GH14583: If meta keys are not always present
        # a new option to set errors='ignore' has been implemented
        i = {
            "Trades": [{
                "general": {
                    "tradeid": 100,
                    "trade_version": 1,
                    "stocks": [{

                        "symbol": "AAPL",
                        "name": "Apple",
                        "price": "0"
                    }, {
                        "symbol": "GOOG",
                        "name": "Google",
                        "price": "0"
                    }
                    ]
                }
            }, {
                "general": {
                    "tradeid": 100,
                    "stocks": [{
                        "symbol": "AAPL",
                        "name": "Apple",
                        "price": "0"
                    }, {
                        "symbol": "GOOG",
                        "name": "Google",
                        "price": "0"
                    }
                    ]
                }
            }
            ]
        }
        j = json_normalize(data=i['Trades'],
                           record_path=[['general', 'stocks']],
                           meta=[['general', 'tradeid'],
                                 ['general', 'trade_version']],
                           errors='ignore')
        expected = {'general.trade_version': {0: 1.0, 1: 1.0, 2: '', 3: ''},
                    'general.tradeid': {0: 100, 1: 100, 2: 100, 3: 100},
                    'name': {0: 'Apple', 1: 'Google', 2: 'Apple', 3: 'Google'},
                    'price': {0: '0', 1: '0', 2: '0', 3: '0'},
                    'symbol': {0: 'AAPL', 1: 'GOOG', 2: 'AAPL', 3: 'GOOG'}}

        assert j.fillna('').to_dict() == expected

        msg = ("Try running with errors='ignore' as key 'trade_version'"
               " is not always present")
        with pytest.raises(KeyError, match=msg):
            json_normalize(
                data=i['Trades'],
                record_path=[['general', 'stocks']],
                meta=[['general', 'tradeid'],
                      ['general', 'trade_version']],
                errors='raise')

    def test_donot_drop_nonevalues(self):
        # GH21356
        data = [
            {'info': None,
             'author_name':
             {'first': 'Smith', 'last_name': 'Appleseed'}
             },
            {'info':
                {'created_at': '11/08/1993', 'last_updated': '26/05/2012'},
             'author_name':
                {'first': 'Jane', 'last_name': 'Doe'}
             }
        ]
        result = nested_to_record(data)
        expected = [
            {'info': None,
             'author_name.first': 'Smith',
             'author_name.last_name': 'Appleseed'},
            {'author_name.first': 'Jane',
             'author_name.last_name': 'Doe',
             'info.created_at': '11/08/1993',
             'info.last_updated': '26/05/2012'}]

        assert result == expected

    def test_nonetype_top_level_bottom_level(self):
        # GH21158: If inner level json has a key with a null value
        # make sure it doesnt do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "country": {
                    "state": {
                        "id": None,
                        "town.info": {
                            "id": None,
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656}}}
            }
        }
        result = nested_to_record(data)
        expected = {
            'id': None,
            'location.country.state.id': None,
            'location.country.state.town.info.id': None,
            'location.country.state.town.info.region': None,
            'location.country.state.town.info.x': 49.151580810546875,
            'location.country.state.town.info.y': -33.148521423339844,
            'location.country.state.town.info.z': 27.572303771972656}
        assert result == expected

    def test_nonetype_multiple_levels(self):
        # GH21158: If inner level json has a key with a null value
        # make sure it doesnt do a new_d.pop twice and except
        data = {
            "id": None,
            "location": {
                "id": None,
                "country": {
                    "id": None,
                    "state": {
                        "id": None,
                        "town.info": {
                            "region": None,
                            "x": 49.151580810546875,
                            "y": -33.148521423339844,
                            "z": 27.572303771972656}}}
            }
        }
        result = nested_to_record(data)
        expected = {
            'id': None,
            'location.id': None,
            'location.country.id': None,
            'location.country.state.id': None,
            'location.country.state.town.info.region': None,
            'location.country.state.town.info.x': 49.151580810546875,
            'location.country.state.town.info.y': -33.148521423339844,
            'location.country.state.town.info.z': 27.572303771972656}
        assert result == expected
