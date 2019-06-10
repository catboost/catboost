import copy
import re
import textwrap

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import DataFrame
import pandas.util.testing as tm

jinja2 = pytest.importorskip('jinja2')
from pandas.io.formats.style import Styler, _get_level_lengths  # noqa  # isort:skip


class TestStyler(object):

    def setup_method(self, method):
        np.random.seed(24)
        self.s = DataFrame({'A': np.random.permutation(range(6))})
        self.df = DataFrame({'A': [0, 1], 'B': np.random.randn(2)})
        self.f = lambda x: x
        self.g = lambda x: x

        def h(x, foo='bar'):
            return pd.Series(
                'color: {foo}'.format(foo=foo), index=x.index, name=x.name)

        self.h = h
        self.styler = Styler(self.df)
        self.attrs = pd.DataFrame({'A': ['color: red', 'color: blue']})
        self.dataframes = [
            self.df,
            pd.DataFrame({'f': [1., 2.], 'o': ['a', 'b'],
                          'c': pd.Categorical(['a', 'b'])})
        ]

    def test_init_non_pandas(self):
        with pytest.raises(TypeError):
            Styler([1, 2, 3])

    def test_init_series(self):
        result = Styler(pd.Series([1, 2]))
        assert result.data.ndim == 2

    def test_repr_html_ok(self):
        self.styler._repr_html_()

    def test_repr_html_mathjax(self):
        # gh-19824
        assert 'tex2jax_ignore' not in self.styler._repr_html_()

        with pd.option_context('display.html.use_mathjax', False):
            assert 'tex2jax_ignore' in self.styler._repr_html_()

    def test_update_ctx(self):
        self.styler._update_ctx(self.attrs)
        expected = {(0, 0): ['color: red'],
                    (1, 0): ['color: blue']}
        assert self.styler.ctx == expected

    def test_update_ctx_flatten_multi(self):
        attrs = DataFrame({"A": ['color: red; foo: bar',
                                 'color: blue; foo: baz']})
        self.styler._update_ctx(attrs)
        expected = {(0, 0): ['color: red', ' foo: bar'],
                    (1, 0): ['color: blue', ' foo: baz']}
        assert self.styler.ctx == expected

    def test_update_ctx_flatten_multi_traliing_semi(self):
        attrs = DataFrame({"A": ['color: red; foo: bar;',
                                 'color: blue; foo: baz;']})
        self.styler._update_ctx(attrs)
        expected = {(0, 0): ['color: red', ' foo: bar'],
                    (1, 0): ['color: blue', ' foo: baz']}
        assert self.styler.ctx == expected

    def test_copy(self):
        s2 = copy.copy(self.styler)
        assert self.styler is not s2
        assert self.styler.ctx is s2.ctx  # shallow
        assert self.styler._todo is s2._todo

        self.styler._update_ctx(self.attrs)
        self.styler.highlight_max()
        assert self.styler.ctx == s2.ctx
        assert self.styler._todo == s2._todo

    def test_deepcopy(self):
        s2 = copy.deepcopy(self.styler)
        assert self.styler is not s2
        assert self.styler.ctx is not s2.ctx
        assert self.styler._todo is not s2._todo

        self.styler._update_ctx(self.attrs)
        self.styler.highlight_max()
        assert self.styler.ctx != s2.ctx
        assert s2._todo == []
        assert self.styler._todo != s2._todo

    def test_clear(self):
        s = self.df.style.highlight_max()._compute()
        assert len(s.ctx) > 0
        assert len(s._todo) > 0
        s.clear()
        assert len(s.ctx) == 0
        assert len(s._todo) == 0

    def test_render(self):
        df = pd.DataFrame({"A": [0, 1]})
        style = lambda x: pd.Series(["color: red", "color: blue"], name=x.name)
        s = Styler(df, uuid='AB').apply(style)
        s.render()
        # it worked?

    def test_render_empty_dfs(self):
        empty_df = DataFrame()
        es = Styler(empty_df)
        es.render()
        # An index but no columns
        DataFrame(columns=['a']).style.render()
        # A column but no index
        DataFrame(index=['a']).style.render()
        # No IndexError raised?

    def test_render_double(self):
        df = pd.DataFrame({"A": [0, 1]})
        style = lambda x: pd.Series(["color: red; border: 1px",
                                     "color: blue; border: 2px"], name=x.name)
        s = Styler(df, uuid='AB').apply(style)
        s.render()
        # it worked?

    def test_set_properties(self):
        df = pd.DataFrame({"A": [0, 1]})
        result = df.style.set_properties(color='white',
                                         size='10px')._compute().ctx
        # order is deterministic
        v = ["color: white", "size: 10px"]
        expected = {(0, 0): v, (1, 0): v}
        assert result.keys() == expected.keys()
        for v1, v2 in zip(result.values(), expected.values()):
            assert sorted(v1) == sorted(v2)

    def test_set_properties_subset(self):
        df = pd.DataFrame({'A': [0, 1]})
        result = df.style.set_properties(subset=pd.IndexSlice[0, 'A'],
                                         color='white')._compute().ctx
        expected = {(0, 0): ['color: white']}
        assert result == expected

    def test_empty_index_name_doesnt_display(self):
        # https://github.com/pandas-dev/pandas/pull/12090#issuecomment-180695902
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = df.style._translate()

        expected = [[{'class': 'blank level0', 'type': 'th', 'value': '',
                      'is_visible': True, 'display_value': ''},
                     {'class': 'col_heading level0 col0',
                      'display_value': 'A',
                      'type': 'th',
                      'value': 'A',
                      'is_visible': True,
                      },
                     {'class': 'col_heading level0 col1',
                      'display_value': 'B',
                      'type': 'th',
                      'value': 'B',
                      'is_visible': True,
                      },
                     {'class': 'col_heading level0 col2',
                      'display_value': 'C',
                      'type': 'th',
                      'value': 'C',
                      'is_visible': True,
                      }]]

        assert result['head'] == expected

    def test_index_name(self):
        # https://github.com/pandas-dev/pandas/issues/11655
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = df.set_index('A').style._translate()

        expected = [[{'class': 'blank level0', 'type': 'th', 'value': '',
                      'display_value': '', 'is_visible': True},
                     {'class': 'col_heading level0 col0', 'type': 'th',
                      'value': 'B', 'display_value': 'B', 'is_visible': True},
                     {'class': 'col_heading level0 col1', 'type': 'th',
                      'value': 'C', 'display_value': 'C', 'is_visible': True}],
                    [{'class': 'index_name level0', 'type': 'th',
                      'value': 'A'},
                     {'class': 'blank', 'type': 'th', 'value': ''},
                     {'class': 'blank', 'type': 'th', 'value': ''}]]

        assert result['head'] == expected

    def test_multiindex_name(self):
        # https://github.com/pandas-dev/pandas/issues/11655
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = df.set_index(['A', 'B']).style._translate()

        expected = [[
            {'class': 'blank', 'type': 'th', 'value': '',
             'display_value': '', 'is_visible': True},
            {'class': 'blank level0', 'type': 'th', 'value': '',
             'display_value': '', 'is_visible': True},
            {'class': 'col_heading level0 col0', 'type': 'th',
             'value': 'C', 'display_value': 'C', 'is_visible': True}],
            [{'class': 'index_name level0', 'type': 'th',
              'value': 'A'},
             {'class': 'index_name level1', 'type': 'th',
              'value': 'B'},
             {'class': 'blank', 'type': 'th', 'value': ''}]]

        assert result['head'] == expected

    def test_numeric_columns(self):
        # https://github.com/pandas-dev/pandas/issues/12125
        # smoke test for _translate
        df = pd.DataFrame({0: [1, 2, 3]})
        df.style._translate()

    def test_apply_axis(self):
        df = pd.DataFrame({'A': [0, 0], 'B': [1, 1]})
        f = lambda x: ['val: {max}'.format(max=x.max()) for v in x]
        result = df.style.apply(f, axis=1)
        assert len(result._todo) == 1
        assert len(result.ctx) == 0
        result._compute()
        expected = {(0, 0): ['val: 1'], (0, 1): ['val: 1'],
                    (1, 0): ['val: 1'], (1, 1): ['val: 1']}
        assert result.ctx == expected

        result = df.style.apply(f, axis=0)
        expected = {(0, 0): ['val: 0'], (0, 1): ['val: 1'],
                    (1, 0): ['val: 0'], (1, 1): ['val: 1']}
        result._compute()
        assert result.ctx == expected
        result = df.style.apply(f)  # default
        result._compute()
        assert result.ctx == expected

    def test_apply_subset(self):
        axes = [0, 1]
        slices = [pd.IndexSlice[:], pd.IndexSlice[:, ['A']],
                  pd.IndexSlice[[1], :], pd.IndexSlice[[1], ['A']],
                  pd.IndexSlice[:2, ['A', 'B']]]
        for ax in axes:
            for slice_ in slices:
                result = self.df.style.apply(self.h, axis=ax, subset=slice_,
                                             foo='baz')._compute().ctx
                expected = {(r, c): ['color: baz']
                            for r, row in enumerate(self.df.index)
                            for c, col in enumerate(self.df.columns)
                            if row in self.df.loc[slice_].index and
                            col in self.df.loc[slice_].columns}
                assert result == expected

    def test_applymap_subset(self):
        def f(x):
            return 'foo: bar'

        slices = [pd.IndexSlice[:], pd.IndexSlice[:, ['A']],
                  pd.IndexSlice[[1], :], pd.IndexSlice[[1], ['A']],
                  pd.IndexSlice[:2, ['A', 'B']]]

        for slice_ in slices:
            result = self.df.style.applymap(f, subset=slice_)._compute().ctx
            expected = {(r, c): ['foo: bar']
                        for r, row in enumerate(self.df.index)
                        for c, col in enumerate(self.df.columns)
                        if row in self.df.loc[slice_].index and
                        col in self.df.loc[slice_].columns}
            assert result == expected

    def test_applymap_subset_multiindex(self):
        # GH 19861
        # Smoke test for applymap
        def color_negative_red(val):
            """
            Takes a scalar and returns a string with
            the css property `'color: red'` for negative
            strings, black otherwise.
            """
            color = 'red' if val < 0 else 'black'
            return 'color: %s' % color

        dic = {
            ('a', 'd'): [-1.12, 2.11],
            ('a', 'c'): [2.78, -2.88],
            ('b', 'c'): [-3.99, 3.77],
            ('b', 'd'): [4.21, -1.22],
        }

        idx = pd.IndexSlice
        df = pd.DataFrame(dic, index=[0, 1])

        (df.style
         .applymap(color_negative_red, subset=idx[:, idx['b', 'd']])
         .render())

    def test_where_with_one_style(self):
        # GH 17474
        def f(x):
            return x > 0.5

        style1 = 'foo: bar'

        result = self.df.style.where(f, style1)._compute().ctx
        expected = {(r, c): [style1 if f(self.df.loc[row, col]) else '']
                    for r, row in enumerate(self.df.index)
                    for c, col in enumerate(self.df.columns)}
        assert result == expected

    def test_where_subset(self):
        # GH 17474
        def f(x):
            return x > 0.5

        style1 = 'foo: bar'
        style2 = 'baz: foo'

        slices = [pd.IndexSlice[:], pd.IndexSlice[:, ['A']],
                  pd.IndexSlice[[1], :], pd.IndexSlice[[1], ['A']],
                  pd.IndexSlice[:2, ['A', 'B']]]

        for slice_ in slices:
            result = self.df.style.where(f, style1, style2,
                                         subset=slice_)._compute().ctx
            expected = {(r, c):
                        [style1 if f(self.df.loc[row, col]) else style2]
                        for r, row in enumerate(self.df.index)
                        for c, col in enumerate(self.df.columns)
                        if row in self.df.loc[slice_].index and
                        col in self.df.loc[slice_].columns}
            assert result == expected

    def test_where_subset_compare_with_applymap(self):
        # GH 17474
        def f(x):
            return x > 0.5

        style1 = 'foo: bar'
        style2 = 'baz: foo'

        def g(x):
            return style1 if f(x) else style2

        slices = [pd.IndexSlice[:], pd.IndexSlice[:, ['A']],
                  pd.IndexSlice[[1], :], pd.IndexSlice[[1], ['A']],
                  pd.IndexSlice[:2, ['A', 'B']]]

        for slice_ in slices:
            result = self.df.style.where(f, style1, style2,
                                         subset=slice_)._compute().ctx
            expected = self.df.style.applymap(g, subset=slice_)._compute().ctx
            assert result == expected

    def test_empty(self):
        df = pd.DataFrame({'A': [1, 0]})
        s = df.style
        s.ctx = {(0, 0): ['color: red'],
                 (1, 0): ['']}

        result = s._translate()['cellstyle']
        expected = [{'props': [['color', ' red']], 'selector': 'row0_col0'},
                    {'props': [['', '']], 'selector': 'row1_col0'}]
        assert result == expected

    def test_bar_align_left(self):
        df = pd.DataFrame({'A': [0, 1, 2]})
        result = df.style.bar()._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient('
                     '90deg,#d65f5f 50.0%, transparent 50.0%)'],
            (2, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient('
                     '90deg,#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

        result = df.style.bar(color='red', width=50)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient('
                     '90deg,red 25.0%, transparent 25.0%)'],
            (2, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient('
                     '90deg,red 50.0%, transparent 50.0%)']
        }
        assert result == expected

        df['C'] = ['a'] * len(df)
        result = df.style.bar(color='red', width=50)._compute().ctx
        assert result == expected
        df['C'] = df['C'].astype('category')
        result = df.style.bar(color='red', width=50)._compute().ctx
        assert result == expected

    def test_bar_align_left_0points(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = df.style.bar()._compute().ctx
        expected = {(0, 0): ['width: 10em', ' height: 80%'],
                    (0, 1): ['width: 10em', ' height: 80%'],
                    (0, 2): ['width: 10em', ' height: 80%'],
                    (1, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 50.0%,'
                             ' transparent 50.0%)'],
                    (1, 1): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 50.0%,'
                             ' transparent 50.0%)'],
                    (1, 2): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 50.0%,'
                             ' transparent 50.0%)'],
                    (2, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 100.0%'
                             ', transparent 100.0%)'],
                    (2, 1): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 100.0%'
                             ', transparent 100.0%)'],
                    (2, 2): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 100.0%'
                             ', transparent 100.0%)']}
        assert result == expected

        result = df.style.bar(axis=1)._compute().ctx
        expected = {(0, 0): ['width: 10em', ' height: 80%'],
                    (0, 1): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 50.0%,'
                             ' transparent 50.0%)'],
                    (0, 2): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 100.0%'
                             ', transparent 100.0%)'],
                    (1, 0): ['width: 10em', ' height: 80%'],
                    (1, 1): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 50.0%'
                             ', transparent 50.0%)'],
                    (1, 2): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 100.0%'
                             ', transparent 100.0%)'],
                    (2, 0): ['width: 10em', ' height: 80%'],
                    (2, 1): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 50.0%'
                             ', transparent 50.0%)'],
                    (2, 2): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,#d65f5f 100.0%'
                             ', transparent 100.0%)']}
        assert result == expected

    def test_bar_align_mid_pos_and_neg(self):
        df = pd.DataFrame({'A': [-10, 0, 20, 90]})

        result = df.style.bar(align='mid', color=[
                              '#d65f5f', '#5fba7d'])._compute().ctx

        expected = {(0, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,'
                             '#d65f5f 10.0%, transparent 10.0%)'],
                    (1, 0): ['width: 10em', ' height: 80%', ],
                    (2, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 10.0%, #5fba7d 10.0%'
                             ', #5fba7d 30.0%, transparent 30.0%)'],
                    (3, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 10.0%, '
                             '#5fba7d 10.0%, #5fba7d 100.0%, '
                             'transparent 100.0%)']}

        assert result == expected

    def test_bar_align_mid_all_pos(self):
        df = pd.DataFrame({'A': [10, 20, 50, 100]})

        result = df.style.bar(align='mid', color=[
                              '#d65f5f', '#5fba7d'])._compute().ctx

        expected = {(0, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,'
                             '#5fba7d 10.0%, transparent 10.0%)'],
                    (1, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,'
                             '#5fba7d 20.0%, transparent 20.0%)'],
                    (2, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,'
                             '#5fba7d 50.0%, transparent 50.0%)'],
                    (3, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,'
                             '#5fba7d 100.0%, transparent 100.0%)']}

        assert result == expected

    def test_bar_align_mid_all_neg(self):
        df = pd.DataFrame({'A': [-100, -60, -30, -20]})

        result = df.style.bar(align='mid', color=[
                              '#d65f5f', '#5fba7d'])._compute().ctx

        expected = {(0, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg,'
                             '#d65f5f 100.0%, transparent 100.0%)'],
                    (1, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 40.0%, '
                             '#d65f5f 40.0%, #d65f5f 100.0%, '
                             'transparent 100.0%)'],
                    (2, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 70.0%, '
                             '#d65f5f 70.0%, #d65f5f 100.0%, '
                             'transparent 100.0%)'],
                    (3, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 80.0%, '
                             '#d65f5f 80.0%, #d65f5f 100.0%, '
                             'transparent 100.0%)']}
        assert result == expected

    def test_bar_align_zero_pos_and_neg(self):
        # See https://github.com/pandas-dev/pandas/pull/14757
        df = pd.DataFrame({'A': [-10, 0, 20, 90]})

        result = df.style.bar(align='zero', color=[
                              '#d65f5f', '#5fba7d'], width=90)._compute().ctx
        expected = {(0, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 40.0%, #d65f5f 40.0%, '
                             '#d65f5f 45.0%, transparent 45.0%)'],
                    (1, 0): ['width: 10em', ' height: 80%'],
                    (2, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 45.0%, #5fba7d 45.0%, '
                             '#5fba7d 55.0%, transparent 55.0%)'],
                    (3, 0): ['width: 10em', ' height: 80%',
                             'background: linear-gradient(90deg, '
                             'transparent 45.0%, #5fba7d 45.0%, '
                             '#5fba7d 90.0%, transparent 90.0%)']}
        assert result == expected

    def test_bar_align_left_axis_none(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [2, 4]})
        result = df.style.bar(axis=None)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 25.0%, transparent 25.0%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 50.0%, transparent 50.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_align_zero_axis_none(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [-2, 4]})
        result = df.style.bar(align='zero', axis=None)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 50.0%, #d65f5f 50.0%, '
                     '#d65f5f 62.5%, transparent 62.5%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 25.0%, #d65f5f 25.0%, '
                     '#d65f5f 50.0%, transparent 50.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 50.0%, #d65f5f 50.0%, '
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_align_mid_axis_none(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [-2, 4]})
        result = df.style.bar(align='mid', axis=None)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 33.3%, #d65f5f 33.3%, '
                     '#d65f5f 50.0%, transparent 50.0%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 33.3%, transparent 33.3%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 33.3%, #d65f5f 33.3%, '
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_align_mid_vmin(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [-2, 4]})
        result = df.style.bar(align='mid', axis=None, vmin=-6)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 60.0%, #d65f5f 60.0%, '
                     '#d65f5f 70.0%, transparent 70.0%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 40.0%, #d65f5f 40.0%, '
                     '#d65f5f 60.0%, transparent 60.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 60.0%, #d65f5f 60.0%, '
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_align_mid_vmax(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [-2, 4]})
        result = df.style.bar(align='mid', axis=None, vmax=8)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 20.0%, #d65f5f 20.0%, '
                     '#d65f5f 30.0%, transparent 30.0%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 20.0%, transparent 20.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 20.0%, #d65f5f 20.0%, '
                     '#d65f5f 60.0%, transparent 60.0%)']
        }
        assert result == expected

    def test_bar_align_mid_vmin_vmax_wide(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [-2, 4]})
        result = df.style.bar(align='mid', axis=None,
                              vmin=-3, vmax=7)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 30.0%, #d65f5f 30.0%, '
                     '#d65f5f 40.0%, transparent 40.0%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 10.0%, #d65f5f 10.0%, '
                     '#d65f5f 30.0%, transparent 30.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 30.0%, #d65f5f 30.0%, '
                     '#d65f5f 70.0%, transparent 70.0%)']
        }
        assert result == expected

    def test_bar_align_mid_vmin_vmax_clipping(self):
        df = pd.DataFrame({'A': [0, 1], 'B': [-2, 4]})
        result = df.style.bar(align='mid', axis=None,
                              vmin=-1, vmax=3)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%'],
            (1, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 25.0%, #d65f5f 25.0%, '
                     '#d65f5f 50.0%, transparent 50.0%)'],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 25.0%, transparent 25.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 25.0%, #d65f5f 25.0%, '
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_align_mid_nans(self):
        df = pd.DataFrame({'A': [1, None], 'B': [-1, 3]})
        result = df.style.bar(align='mid', axis=None)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 25.0%, #d65f5f 25.0%, '
                     '#d65f5f 50.0%, transparent 50.0%)'],
            (1, 0): [''],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg,'
                     '#d65f5f 25.0%, transparent 25.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 25.0%, #d65f5f 25.0%, '
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_align_zero_nans(self):
        df = pd.DataFrame({'A': [1, None], 'B': [-1, 2]})
        result = df.style.bar(align='zero', axis=None)._compute().ctx
        expected = {
            (0, 0): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 50.0%, #d65f5f 50.0%, '
                     '#d65f5f 75.0%, transparent 75.0%)'],
            (1, 0): [''],
            (0, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 25.0%, #d65f5f 25.0%, '
                     '#d65f5f 50.0%, transparent 50.0%)'],
            (1, 1): ['width: 10em', ' height: 80%',
                     'background: linear-gradient(90deg, '
                     'transparent 50.0%, #d65f5f 50.0%, '
                     '#d65f5f 100.0%, transparent 100.0%)']
        }
        assert result == expected

    def test_bar_bad_align_raises(self):
        df = pd.DataFrame({'A': [-100, -60, -30, -20]})
        with pytest.raises(ValueError):
            df.style.bar(align='poorly', color=['#d65f5f', '#5fba7d'])

    def test_highlight_null(self, null_color='red'):
        df = pd.DataFrame({'A': [0, np.nan]})
        result = df.style.highlight_null()._compute().ctx
        expected = {(0, 0): [''],
                    (1, 0): ['background-color: red']}
        assert result == expected

    def test_nonunique_raises(self):
        df = pd.DataFrame([[1, 2]], columns=['A', 'A'])
        with pytest.raises(ValueError):
            df.style

        with pytest.raises(ValueError):
            Styler(df)

    def test_caption(self):
        styler = Styler(self.df, caption='foo')
        result = styler.render()
        assert all(['caption' in result, 'foo' in result])

        styler = self.df.style
        result = styler.set_caption('baz')
        assert styler is result
        assert styler.caption == 'baz'

    def test_uuid(self):
        styler = Styler(self.df, uuid='abc123')
        result = styler.render()
        assert 'abc123' in result

        styler = self.df.style
        result = styler.set_uuid('aaa')
        assert result is styler
        assert result.uuid == 'aaa'

    def test_unique_id(self):
        # See https://github.com/pandas-dev/pandas/issues/16780
        df = pd.DataFrame({'a': [1, 3, 5, 6], 'b': [2, 4, 12, 21]})
        result = df.style.render(uuid='test')
        assert 'test' in result
        ids = re.findall('id="(.*?)"', result)
        assert np.unique(ids).size == len(ids)

    def test_table_styles(self):
        style = [{'selector': 'th', 'props': [('foo', 'bar')]}]
        styler = Styler(self.df, table_styles=style)
        result = ' '.join(styler.render().split())
        assert 'th { foo: bar; }' in result

        styler = self.df.style
        result = styler.set_table_styles(style)
        assert styler is result
        assert styler.table_styles == style

    def test_table_attributes(self):
        attributes = 'class="foo" data-bar'
        styler = Styler(self.df, table_attributes=attributes)
        result = styler.render()
        assert 'class="foo" data-bar' in result

        result = self.df.style.set_table_attributes(attributes).render()
        assert 'class="foo" data-bar' in result

    def test_precision(self):
        with pd.option_context('display.precision', 10):
            s = Styler(self.df)
        assert s.precision == 10
        s = Styler(self.df, precision=2)
        assert s.precision == 2

        s2 = s.set_precision(4)
        assert s is s2
        assert s.precision == 4

    def test_apply_none(self):
        def f(x):
            return pd.DataFrame(np.where(x == x.max(), 'color: red', ''),
                                index=x.index, columns=x.columns)
        result = (pd.DataFrame([[1, 2], [3, 4]])
                  .style.apply(f, axis=None)._compute().ctx)
        assert result[(1, 1)] == ['color: red']

    def test_trim(self):
        result = self.df.style.render()  # trim=True
        assert result.count('#') == 0

        result = self.df.style.highlight_max().render()
        assert result.count('#') == len(self.df.columns)

    def test_highlight_max(self):
        df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        # max(df) = min(-df)
        for max_ in [True, False]:
            if max_:
                attr = 'highlight_max'
            else:
                df = -df
                attr = 'highlight_min'
            result = getattr(df.style, attr)()._compute().ctx
            assert result[(1, 1)] == ['background-color: yellow']

            result = getattr(df.style, attr)(color='green')._compute().ctx
            assert result[(1, 1)] == ['background-color: green']

            result = getattr(df.style, attr)(subset='A')._compute().ctx
            assert result[(1, 0)] == ['background-color: yellow']

            result = getattr(df.style, attr)(axis=0)._compute().ctx
            expected = {(1, 0): ['background-color: yellow'],
                        (1, 1): ['background-color: yellow'],
                        (0, 1): [''], (0, 0): ['']}
            assert result == expected

            result = getattr(df.style, attr)(axis=1)._compute().ctx
            expected = {(0, 1): ['background-color: yellow'],
                        (1, 1): ['background-color: yellow'],
                        (0, 0): [''], (1, 0): ['']}
            assert result == expected

        # separate since we can't negate the strs
        df['C'] = ['a', 'b']
        result = df.style.highlight_max()._compute().ctx
        expected = {(1, 1): ['background-color: yellow']}

        result = df.style.highlight_min()._compute().ctx
        expected = {(0, 0): ['background-color: yellow']}

    def test_export(self):
        f = lambda x: 'color: red' if x > 0 else 'color: blue'
        g = lambda x, y, z: 'color: {z}'.format(z=z) \
            if x > 0 else 'color: {z}'.format(z=z)
        style1 = self.styler
        style1.applymap(f)\
            .applymap(g, y='a', z='b')\
            .highlight_max()
        result = style1.export()
        style2 = self.df.style
        style2.use(result)
        assert style1._todo == style2._todo
        style2.render()

    def test_display_format(self):
        df = pd.DataFrame(np.random.random(size=(2, 2)))
        ctx = df.style.format("{:0.1f}")._translate()

        assert all(['display_value' in c for c in row]
                   for row in ctx['body'])
        assert all([len(c['display_value']) <= 3 for c in row[1:]]
                   for row in ctx['body'])
        assert len(ctx['body'][0][1]['display_value'].lstrip('-')) <= 3

    def test_display_format_raises(self):
        df = pd.DataFrame(np.random.randn(2, 2))
        with pytest.raises(TypeError):
            df.style.format(5)
        with pytest.raises(TypeError):
            df.style.format(True)

    def test_display_subset(self):
        df = pd.DataFrame([[.1234, .1234], [1.1234, 1.1234]],
                          columns=['a', 'b'])
        ctx = df.style.format({"a": "{:0.1f}", "b": "{0:.2%}"},
                              subset=pd.IndexSlice[0, :])._translate()
        expected = '0.1'
        assert ctx['body'][0][1]['display_value'] == expected
        assert ctx['body'][1][1]['display_value'] == '1.1234'
        assert ctx['body'][0][2]['display_value'] == '12.34%'

        raw_11 = '1.1234'
        ctx = df.style.format("{:0.1f}",
                              subset=pd.IndexSlice[0, :])._translate()
        assert ctx['body'][0][1]['display_value'] == expected
        assert ctx['body'][1][1]['display_value'] == raw_11

        ctx = df.style.format("{:0.1f}",
                              subset=pd.IndexSlice[0, :])._translate()
        assert ctx['body'][0][1]['display_value'] == expected
        assert ctx['body'][1][1]['display_value'] == raw_11

        ctx = df.style.format("{:0.1f}",
                              subset=pd.IndexSlice['a'])._translate()
        assert ctx['body'][0][1]['display_value'] == expected
        assert ctx['body'][0][2]['display_value'] == '0.1234'

        ctx = df.style.format("{:0.1f}",
                              subset=pd.IndexSlice[0, 'a'])._translate()
        assert ctx['body'][0][1]['display_value'] == expected
        assert ctx['body'][1][1]['display_value'] == raw_11

        ctx = df.style.format("{:0.1f}",
                              subset=pd.IndexSlice[[0, 1], ['a']])._translate()
        assert ctx['body'][0][1]['display_value'] == expected
        assert ctx['body'][1][1]['display_value'] == '1.1'
        assert ctx['body'][0][2]['display_value'] == '0.1234'
        assert ctx['body'][1][2]['display_value'] == '1.1234'

    def test_display_dict(self):
        df = pd.DataFrame([[.1234, .1234], [1.1234, 1.1234]],
                          columns=['a', 'b'])
        ctx = df.style.format({"a": "{:0.1f}", "b": "{0:.2%}"})._translate()
        assert ctx['body'][0][1]['display_value'] == '0.1'
        assert ctx['body'][0][2]['display_value'] == '12.34%'
        df['c'] = ['aaa', 'bbb']
        ctx = df.style.format({"a": "{:0.1f}", "c": str.upper})._translate()
        assert ctx['body'][0][1]['display_value'] == '0.1'
        assert ctx['body'][0][3]['display_value'] == 'AAA'

    def test_bad_apply_shape(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            df.style._apply(lambda x: 'x', subset=pd.IndexSlice[[0, 1], :])

        with pytest.raises(ValueError):
            df.style._apply(lambda x: [''], subset=pd.IndexSlice[[0, 1], :])

        with pytest.raises(ValueError):
            df.style._apply(lambda x: ['', '', '', ''])

        with pytest.raises(ValueError):
            df.style._apply(lambda x: ['', '', ''], subset=1)

        with pytest.raises(ValueError):
            df.style._apply(lambda x: ['', '', ''], axis=1)

    def test_apply_bad_return(self):
        def f(x):
            return ''
        df = pd.DataFrame([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            df.style._apply(f, axis=None)

    def test_apply_bad_labels(self):
        def f(x):
            return pd.DataFrame(index=[1, 2], columns=['a', 'b'])
        df = pd.DataFrame([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            df.style._apply(f, axis=None)

    def test_get_level_lengths(self):
        index = pd.MultiIndex.from_product([['a', 'b'], [0, 1, 2]])
        expected = {(0, 0): 3, (0, 3): 3, (1, 0): 1, (1, 1): 1, (1, 2): 1,
                    (1, 3): 1, (1, 4): 1, (1, 5): 1}
        result = _get_level_lengths(index)
        tm.assert_dict_equal(result, expected)

    def test_get_level_lengths_un_sorted(self):
        index = pd.MultiIndex.from_arrays([
            [1, 1, 2, 1],
            ['a', 'b', 'b', 'd']
        ])
        expected = {(0, 0): 2, (0, 2): 1, (0, 3): 1,
                    (1, 0): 1, (1, 1): 1, (1, 2): 1, (1, 3): 1}
        result = _get_level_lengths(index)
        tm.assert_dict_equal(result, expected)

    def test_mi_sparse(self):
        df = pd.DataFrame({'A': [1, 2]},
                          index=pd.MultiIndex.from_arrays([['a', 'a'],
                                                           [0, 1]]))

        result = df.style._translate()
        body_0 = result['body'][0][0]
        expected_0 = {
            "value": "a", "display_value": "a", "is_visible": True,
            "type": "th", "attributes": ["rowspan=2"],
            "class": "row_heading level0 row0", "id": "level0_row0"
        }
        tm.assert_dict_equal(body_0, expected_0)

        body_1 = result['body'][0][1]
        expected_1 = {
            "value": 0, "display_value": 0, "is_visible": True,
            "type": "th", "class": "row_heading level1 row0",
            "id": "level1_row0"
        }
        tm.assert_dict_equal(body_1, expected_1)

        body_10 = result['body'][1][0]
        expected_10 = {
            "value": 'a', "display_value": 'a', "is_visible": False,
            "type": "th", "class": "row_heading level0 row1",
            "id": "level0_row1"
        }
        tm.assert_dict_equal(body_10, expected_10)

        head = result['head'][0]
        expected = [
            {'type': 'th', 'class': 'blank', 'value': '',
             'is_visible': True, "display_value": ''},
            {'type': 'th', 'class': 'blank level0', 'value': '',
             'is_visible': True, 'display_value': ''},
            {'type': 'th', 'class': 'col_heading level0 col0', 'value': 'A',
             'is_visible': True, 'display_value': 'A'}]
        assert head == expected

    def test_mi_sparse_disabled(self):
        with pd.option_context('display.multi_sparse', False):
            df = pd.DataFrame({'A': [1, 2]},
                              index=pd.MultiIndex.from_arrays([['a', 'a'],
                                                               [0, 1]]))
            result = df.style._translate()
        body = result['body']
        for row in body:
            assert 'attributes' not in row[0]

    def test_mi_sparse_index_names(self):
        df = pd.DataFrame({'A': [1, 2]}, index=pd.MultiIndex.from_arrays(
            [['a', 'a'], [0, 1]],
            names=['idx_level_0', 'idx_level_1'])
        )
        result = df.style._translate()
        head = result['head'][1]
        expected = [{
            'class': 'index_name level0', 'value': 'idx_level_0',
            'type': 'th'},
            {'class': 'index_name level1', 'value': 'idx_level_1',
             'type': 'th'},
            {'class': 'blank', 'value': '', 'type': 'th'}]

        assert head == expected

    def test_mi_sparse_column_names(self):
        df = pd.DataFrame(
            np.arange(16).reshape(4, 4),
            index=pd.MultiIndex.from_arrays(
                [['a', 'a', 'b', 'a'], [0, 1, 1, 2]],
                names=['idx_level_0', 'idx_level_1']),
            columns=pd.MultiIndex.from_arrays(
                [['C1', 'C1', 'C2', 'C2'], [1, 0, 1, 0]],
                names=['col_0', 'col_1']
            )
        )
        result = df.style._translate()
        head = result['head'][1]
        expected = [
            {'class': 'blank', 'value': '', 'display_value': '',
             'type': 'th', 'is_visible': True},
            {'class': 'index_name level1', 'value': 'col_1',
             'display_value': 'col_1', 'is_visible': True, 'type': 'th'},
            {'class': 'col_heading level1 col0',
             'display_value': 1,
             'is_visible': True,
             'type': 'th',
             'value': 1},
            {'class': 'col_heading level1 col1',
             'display_value': 0,
             'is_visible': True,
             'type': 'th',
             'value': 0},

            {'class': 'col_heading level1 col2',
             'display_value': 1,
             'is_visible': True,
             'type': 'th',
             'value': 1},

            {'class': 'col_heading level1 col3',
             'display_value': 0,
             'is_visible': True,
             'type': 'th',
             'value': 0},
        ]
        assert head == expected

    def test_hide_single_index(self):
        # GH 14194
        # single unnamed index
        ctx = self.df.style._translate()
        assert ctx['body'][0][0]['is_visible']
        assert ctx['head'][0][0]['is_visible']
        ctx2 = self.df.style.hide_index()._translate()
        assert not ctx2['body'][0][0]['is_visible']
        assert not ctx2['head'][0][0]['is_visible']

        # single named index
        ctx3 = self.df.set_index('A').style._translate()
        assert ctx3['body'][0][0]['is_visible']
        assert len(ctx3['head']) == 2  # 2 header levels
        assert ctx3['head'][0][0]['is_visible']

        ctx4 = self.df.set_index('A').style.hide_index()._translate()
        assert not ctx4['body'][0][0]['is_visible']
        assert len(ctx4['head']) == 1  # only 1 header levels
        assert not ctx4['head'][0][0]['is_visible']

    def test_hide_multiindex(self):
        # GH 14194
        df = pd.DataFrame({'A': [1, 2]}, index=pd.MultiIndex.from_arrays(
            [['a', 'a'], [0, 1]],
            names=['idx_level_0', 'idx_level_1'])
        )
        ctx1 = df.style._translate()
        # tests for 'a' and '0'
        assert ctx1['body'][0][0]['is_visible']
        assert ctx1['body'][0][1]['is_visible']
        # check for blank header rows
        assert ctx1['head'][0][0]['is_visible']
        assert ctx1['head'][0][1]['is_visible']

        ctx2 = df.style.hide_index()._translate()
        # tests for 'a' and '0'
        assert not ctx2['body'][0][0]['is_visible']
        assert not ctx2['body'][0][1]['is_visible']
        # check for blank header rows
        assert not ctx2['head'][0][0]['is_visible']
        assert not ctx2['head'][0][1]['is_visible']

    def test_hide_columns_single_level(self):
        # GH 14194
        # test hiding single column
        ctx = self.df.style._translate()
        assert ctx['head'][0][1]['is_visible']
        assert ctx['head'][0][1]['display_value'] == 'A'
        assert ctx['head'][0][2]['is_visible']
        assert ctx['head'][0][2]['display_value'] == 'B'
        assert ctx['body'][0][1]['is_visible']  # col A, row 1
        assert ctx['body'][1][2]['is_visible']  # col B, row 1

        ctx = self.df.style.hide_columns('A')._translate()
        assert not ctx['head'][0][1]['is_visible']
        assert not ctx['body'][0][1]['is_visible']  # col A, row 1
        assert ctx['body'][1][2]['is_visible']  # col B, row 1

        # test hiding mulitiple columns
        ctx = self.df.style.hide_columns(['A', 'B'])._translate()
        assert not ctx['head'][0][1]['is_visible']
        assert not ctx['head'][0][2]['is_visible']
        assert not ctx['body'][0][1]['is_visible']  # col A, row 1
        assert not ctx['body'][1][2]['is_visible']  # col B, row 1

    def test_hide_columns_mult_levels(self):
        # GH 14194
        # setup dataframe with multiple column levels and indices
        i1 = pd.MultiIndex.from_arrays([['a', 'a'], [0, 1]],
                                       names=['idx_level_0',
                                              'idx_level_1'])
        i2 = pd.MultiIndex.from_arrays([['b', 'b'], [0, 1]],
                                       names=['col_level_0',
                                              'col_level_1'])
        df = pd.DataFrame([[1, 2], [3, 4]], index=i1, columns=i2)
        ctx = df.style._translate()
        # column headers
        assert ctx['head'][0][2]['is_visible']
        assert ctx['head'][1][2]['is_visible']
        assert ctx['head'][1][3]['display_value'] == 1
        # indices
        assert ctx['body'][0][0]['is_visible']
        # data
        assert ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][2]['display_value'] == 3
        assert ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][3]['display_value'] == 4

        # hide top column level, which hides both columns
        ctx = df.style.hide_columns('b')._translate()
        assert not ctx['head'][0][2]['is_visible']  # b
        assert not ctx['head'][1][2]['is_visible']  # 0
        assert not ctx['body'][1][2]['is_visible']  # 3
        assert ctx['body'][0][0]['is_visible']  # index

        # hide first column only
        ctx = df.style.hide_columns([('b', 0)])._translate()
        assert ctx['head'][0][2]['is_visible']  # b
        assert not ctx['head'][1][2]['is_visible']  # 0
        assert not ctx['body'][1][2]['is_visible']  # 3
        assert ctx['body'][1][3]['is_visible']
        assert ctx['body'][1][3]['display_value'] == 4

        # hide second column and index
        ctx = df.style.hide_columns([('b', 1)]).hide_index()._translate()
        assert not ctx['body'][0][0]['is_visible']  # index
        assert ctx['head'][0][2]['is_visible']  # b
        assert ctx['head'][1][2]['is_visible']  # 0
        assert not ctx['head'][1][3]['is_visible']  # 1
        assert not ctx['body'][1][3]['is_visible']  # 4
        assert ctx['body'][1][2]['is_visible']
        assert ctx['body'][1][2]['display_value'] == 3

    def test_pipe(self):
        def set_caption_from_template(styler, a, b):
            return styler.set_caption(
                'Dataframe with a = {a} and b = {b}'.format(a=a, b=b))

        styler = self.df.style.pipe(set_caption_from_template, 'A', b='B')
        assert 'Dataframe with a = A and b = B' in styler.render()

        # Test with an argument that is a (callable, keyword_name) pair.
        def f(a, b, styler):
            return (a, b, styler)

        styler = self.df.style
        result = styler.pipe((f, 'styler'), a=1, b=2)
        assert result == (1, 2, styler)


@td.skip_if_no_mpl
class TestStylerMatplotlibDep(object):

    def test_background_gradient(self):
        df = pd.DataFrame([[1, 2], [2, 4]], columns=['A', 'B'])

        for c_map in [None, 'YlOrRd']:
            result = df.style.background_gradient(cmap=c_map)._compute().ctx
            assert all("#" in x[0] for x in result.values())
            assert result[(0, 0)] == result[(0, 1)]
            assert result[(1, 0)] == result[(1, 1)]

        result = df.style.background_gradient(
            subset=pd.IndexSlice[1, 'A'])._compute().ctx

        assert result[(1, 0)] == ['background-color: #fff7fb',
                                  'color: #000000']

    @pytest.mark.parametrize(
        'c_map,expected', [
            (None, {
                (0, 0): ['background-color: #440154', 'color: #f1f1f1'],
                (1, 0): ['background-color: #fde725', 'color: #000000']}),
            ('YlOrRd', {
                (0, 0): ['background-color: #ffffcc', 'color: #000000'],
                (1, 0): ['background-color: #800026', 'color: #f1f1f1']})])
    def test_text_color_threshold(self, c_map, expected):
        df = pd.DataFrame([1, 2], columns=['A'])
        result = df.style.background_gradient(cmap=c_map)._compute().ctx
        assert result == expected

    @pytest.mark.parametrize("text_color_threshold", [1.1, '1', -1, [2, 2]])
    def test_text_color_threshold_raises(self, text_color_threshold):
        df = pd.DataFrame([[1, 2], [2, 4]], columns=['A', 'B'])
        msg = "`text_color_threshold` must be a value from 0 to 1."
        with pytest.raises(ValueError, match=msg):
            df.style.background_gradient(
                text_color_threshold=text_color_threshold)._compute()

    @td.skip_if_no_mpl
    def test_background_gradient_axis(self):
        df = pd.DataFrame([[1, 2], [2, 4]], columns=['A', 'B'])

        low = ['background-color: #f7fbff', 'color: #000000']
        high = ['background-color: #08306b', 'color: #f1f1f1']
        mid = ['background-color: #abd0e6', 'color: #000000']
        result = df.style.background_gradient(cmap='Blues',
                                              axis=0)._compute().ctx
        assert result[(0, 0)] == low
        assert result[(0, 1)] == low
        assert result[(1, 0)] == high
        assert result[(1, 1)] == high

        result = df.style.background_gradient(cmap='Blues',
                                              axis=1)._compute().ctx
        assert result[(0, 0)] == low
        assert result[(0, 1)] == high
        assert result[(1, 0)] == low
        assert result[(1, 1)] == high

        result = df.style.background_gradient(cmap='Blues',
                                              axis=None)._compute().ctx
        assert result[(0, 0)] == low
        assert result[(0, 1)] == mid
        assert result[(1, 0)] == mid
        assert result[(1, 1)] == high


def test_block_names():
    # catch accidental removal of a block
    expected = {
        'before_style', 'style', 'table_styles', 'before_cellstyle',
        'cellstyle', 'before_table', 'table', 'caption', 'thead', 'tbody',
        'after_table', 'before_head_rows', 'head_tr', 'after_head_rows',
        'before_rows', 'tr', 'after_rows',
    }
    result = set(Styler.template.blocks)
    assert result == expected


def test_from_custom_template(tmpdir):
    p = tmpdir.mkdir("templates").join("myhtml.tpl")
    p.write(textwrap.dedent("""\
        {% extends "html.tpl" %}
        {% block table %}
        <h1>{{ table_title|default("My Table") }}</h1>
        {{ super() }}
        {% endblock table %}"""))
    result = Styler.from_custom_template(str(tmpdir.join('templates')),
                                         'myhtml.tpl')
    assert issubclass(result, Styler)
    assert result.env is not Styler.env
    assert result.template is not Styler.template
    styler = result(pd.DataFrame({"A": [1, 2]}))
    assert styler.render()
