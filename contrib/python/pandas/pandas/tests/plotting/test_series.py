# coding: utf-8

""" Test cases for Series.plot """


from datetime import datetime
from itertools import chain

import numpy as np
from numpy.random import randn
import pytest

from pandas.compat import lrange, range
import pandas.util._test_decorators as td

import pandas as pd
from pandas import DataFrame, Series, date_range
from pandas.tests.plotting.common import (
    TestPlotBase, _check_plot_works, _ok_for_gaussian_kde,
    _skip_if_no_scipy_gaussian_kde)
import pandas.util.testing as tm

import pandas.plotting as plotting


@td.skip_if_no_mpl
class TestSeriesPlots(TestPlotBase):

    def setup_method(self, method):
        TestPlotBase.setup_method(self, method)
        import matplotlib as mpl
        mpl.rcdefaults()

        self.ts = tm.makeTimeSeries()
        self.ts.name = 'ts'

        self.series = tm.makeStringSeries()
        self.series.name = 'series'

        self.iseries = tm.makePeriodSeries()
        self.iseries.name = 'iseries'

    @pytest.mark.slow
    def test_plot(self):
        _check_plot_works(self.ts.plot, label='foo')
        _check_plot_works(self.ts.plot, use_index=False)
        axes = _check_plot_works(self.ts.plot, rot=0)
        self._check_ticks_props(axes, xrot=0)

        ax = _check_plot_works(self.ts.plot, style='.', logy=True)
        self._check_ax_scales(ax, yaxis='log')

        ax = _check_plot_works(self.ts.plot, style='.', logx=True)
        self._check_ax_scales(ax, xaxis='log')

        ax = _check_plot_works(self.ts.plot, style='.', loglog=True)
        self._check_ax_scales(ax, xaxis='log', yaxis='log')

        _check_plot_works(self.ts[:10].plot.bar)
        _check_plot_works(self.ts.plot.area, stacked=False)
        _check_plot_works(self.iseries.plot)

        for kind in ['line', 'bar', 'barh', 'kde', 'hist', 'box']:
            if not _ok_for_gaussian_kde(kind):
                continue
            _check_plot_works(self.series[:5].plot, kind=kind)

        _check_plot_works(self.series[:10].plot.barh)
        ax = _check_plot_works(Series(randn(10)).plot.bar, color='black')
        self._check_colors([ax.patches[0]], facecolors=['black'])

        # GH 6951
        ax = _check_plot_works(self.ts.plot, subplots=True)
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1))

        ax = _check_plot_works(self.ts.plot, subplots=True, layout=(-1, 1))
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1))
        ax = _check_plot_works(self.ts.plot, subplots=True, layout=(1, -1))
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_plot_figsize_and_title(self):
        # figsize and title
        _, ax = self.plt.subplots()
        ax = self.series.plot(title='Test', figsize=(16, 8), ax=ax)
        self._check_text_labels(ax.title, 'Test')
        self._check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self):
        # GH 8242
        key = 'axes.prop_cycle'
        colors = self.plt.rcParams[key]
        _, ax = self.plt.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        assert colors == self.plt.rcParams[key]

    def test_ts_line_lim(self):
        fig, ax = self.plt.subplots()
        ax = self.ts.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]
        tm.close()

        ax = self.ts.plot(secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.area(stacked=False, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        tm.close()

        # GH 7471
        _, ax = self.plt.subplots()
        ax = self.ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        tm.close()

        tz_ts = self.ts.copy()
        tz_ts.index = tz_ts.tz_localize('GMT').tz_convert('CET')
        _, ax = self.plt.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]
        tm.close()

        _, ax = self.plt.subplots()
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        xmin, xmax = ax.get_xlim()
        line = ax.get_lines()[0].get_data(orig=False)[0]
        assert xmin <= line[0]
        assert xmax >= line[-1]

    def test_label(self):
        s = Series([1, 2])
        _, ax = self.plt.subplots()
        ax = s.plot(label='LABEL', legend=True, ax=ax)
        self._check_legend_labels(ax, labels=['LABEL'])
        self.plt.close()
        _, ax = self.plt.subplots()
        ax = s.plot(legend=True, ax=ax)
        self._check_legend_labels(ax, labels=['None'])
        self.plt.close()
        # get name from index
        s.name = 'NAME'
        _, ax = self.plt.subplots()
        ax = s.plot(legend=True, ax=ax)
        self._check_legend_labels(ax, labels=['NAME'])
        self.plt.close()
        # override the default
        _, ax = self.plt.subplots()
        ax = s.plot(legend=True, label='LABEL', ax=ax)
        self._check_legend_labels(ax, labels=['LABEL'])
        self.plt.close()
        # Add lebel info, but don't draw
        _, ax = self.plt.subplots()
        ax = s.plot(legend=False, label='LABEL', ax=ax)
        assert ax.get_legend() is None  # Hasn't been drawn
        ax.legend()  # draw it
        self._check_legend_labels(ax, labels=['LABEL'])

    def test_line_area_nan_series(self):
        values = [1, 2, np.nan, 3]
        s = Series(values)
        ts = Series(values, index=tm.makeDateIndex(k=4))

        for d in [s, ts]:
            ax = _check_plot_works(d.plot)
            masked = ax.lines[0].get_ydata()
            # remove nan for comparison purpose
            exp = np.array([1, 2, 3], dtype=np.float64)
            tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
            tm.assert_numpy_array_equal(
                masked.mask, np.array([False, False, True, False]))

            expected = np.array([1, 2, 0, 3], dtype=np.float64)
            ax = _check_plot_works(d.plot, stacked=True)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
            ax = _check_plot_works(d.plot.area)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
            ax = _check_plot_works(d.plot.area, stacked=False)
            tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    def test_line_use_index_false(self):
        s = Series([1, 2, 3], index=['a', 'b', 'c'])
        s.index.name = 'The Index'
        _, ax = self.plt.subplots()
        ax = s.plot(use_index=False, ax=ax)
        label = ax.get_xlabel()
        assert label == ''
        _, ax = self.plt.subplots()
        ax2 = s.plot.bar(use_index=False, ax=ax)
        label2 = ax2.get_xlabel()
        assert label2 == ''

    @pytest.mark.slow
    def test_bar_log(self):
        expected = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

        _, ax = self.plt.subplots()
        ax = Series([200, 500]).plot.bar(log=True, ax=ax)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)
        tm.close()

        _, ax = self.plt.subplots()
        ax = Series([200, 500]).plot.barh(log=True, ax=ax)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), expected)
        tm.close()

        # GH 9905
        expected = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

        _, ax = self.plt.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind='bar', ax=ax)
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = ax.get_ylim()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)
        tm.close()

        _, ax = self.plt.subplots()
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind='barh', ax=ax)
        res = ax.get_xlim()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), expected)

    @pytest.mark.slow
    def test_bar_ignore_index(self):
        df = Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        _, ax = self.plt.subplots()
        ax = df.plot.bar(use_index=False, ax=ax)
        self._check_text_labels(ax.get_xticklabels(), ['0', '1', '2', '3'])

    def test_bar_user_colors(self):
        s = Series([1, 2, 3, 4])
        ax = s.plot.bar(color=['red', 'blue', 'blue', 'red'])
        result = [p.get_facecolor() for p in ax.patches]
        expected = [(1., 0., 0., 1.),
                    (0., 0., 1., 1.),
                    (0., 0., 1., 1.),
                    (1., 0., 0., 1.)]
        assert result == expected

    def test_rotation(self):
        df = DataFrame(randn(5, 5))
        # Default rot 0
        _, ax = self.plt.subplots()
        axes = df.plot(ax=ax)
        self._check_ticks_props(axes, xrot=0)

        _, ax = self.plt.subplots()
        axes = df.plot(rot=30, ax=ax)
        self._check_ticks_props(axes, xrot=30)

    def test_irregular_datetime(self):
        rng = date_range('1/1/2000', '3/1/2000')
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        ser = Series(randn(len(rng)), rng)
        _, ax = self.plt.subplots()
        ax = ser.plot(ax=ax)
        xp = datetime(1999, 1, 1).toordinal()
        ax.set_xlim('1/1/1999', '1/1/2001')
        assert xp == ax.get_xlim()[0]

    def test_unsorted_index_xlim(self):
        ser = Series([0., 1., np.nan, 3., 4., 5., 6.],
                     index=[1., 0., 3., 2., np.nan, 3., 2.])
        _, ax = self.plt.subplots()
        ax = ser.plot(ax=ax)
        xmin, xmax = ax.get_xlim()
        lines = ax.get_lines()
        assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
        assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])

    @pytest.mark.slow
    def test_pie_series(self):
        # if sum of values is less than 1.0, pie handle them as rate and draw
        # semicircle.
        series = Series(np.random.randint(1, 5),
                        index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
        ax = _check_plot_works(series.plot.pie)
        self._check_text_labels(ax.texts, series.index)
        assert ax.get_ylabel() == 'YLABEL'

        # without wedge labels
        ax = _check_plot_works(series.plot.pie, labels=None)
        self._check_text_labels(ax.texts, [''] * 5)

        # with less colors than elements
        color_args = ['r', 'g', 'b']
        ax = _check_plot_works(series.plot.pie, colors=color_args)

        color_expected = ['r', 'g', 'b', 'r', 'g']
        self._check_colors(ax.patches, facecolors=color_expected)

        # with labels and colors
        labels = ['A', 'B', 'C', 'D', 'E']
        color_args = ['r', 'g', 'b', 'c', 'm']
        ax = _check_plot_works(series.plot.pie, labels=labels,
                               colors=color_args)
        self._check_text_labels(ax.texts, labels)
        self._check_colors(ax.patches, facecolors=color_args)

        # with autopct and fontsize
        ax = _check_plot_works(series.plot.pie, colors=color_args,
                               autopct='%.2f', fontsize=7)
        pcts = ['{0:.2f}'.format(s * 100)
                for s in series.values / float(series.sum())]
        expected_texts = list(chain.from_iterable(zip(series.index, pcts)))
        self._check_text_labels(ax.texts, expected_texts)
        for t in ax.texts:
            assert t.get_fontsize() == 7

        # includes negative value
        with pytest.raises(ValueError):
            series = Series([1, 2, 0, 4, -1], index=['a', 'b', 'c', 'd', 'e'])
            series.plot.pie()

        # includes nan
        series = Series([1, 2, np.nan, 4], index=['a', 'b', 'c', 'd'],
                        name='YLABEL')
        ax = _check_plot_works(series.plot.pie)
        self._check_text_labels(ax.texts, ['a', 'b', '', 'd'])

    def test_pie_nan(self):
        s = Series([1, np.nan, 1, 1])
        _, ax = self.plt.subplots()
        ax = s.plot.pie(legend=True, ax=ax)
        expected = ['0', '', '2', '3']
        result = [x.get_text() for x in ax.texts]
        assert result == expected

    @pytest.mark.slow
    def test_hist_df_kwargs(self):
        df = DataFrame(np.random.randn(10, 2))
        _, ax = self.plt.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 10

    @pytest.mark.slow
    def test_hist_df_with_nonnumerics(self):
        # GH 9853
        with tm.RNGContext(1):
            df = DataFrame(
                np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
        df['E'] = ['x', 'y'] * 5
        _, ax = self.plt.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 20

        _, ax = self.plt.subplots()
        ax = df.plot.hist(ax=ax)  # bins=10
        assert len(ax.patches) == 40

    @pytest.mark.slow
    def test_hist_legacy(self):
        _check_plot_works(self.ts.hist)
        _check_plot_works(self.ts.hist, grid=False)
        _check_plot_works(self.ts.hist, figsize=(8, 10))
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(self.ts.hist,
                              by=self.ts.index.month)
        with tm.assert_produces_warning(UserWarning):
            _check_plot_works(self.ts.hist,
                              by=self.ts.index.month, bins=5)

        fig, ax = self.plt.subplots(1, 1)
        _check_plot_works(self.ts.hist, ax=ax)
        _check_plot_works(self.ts.hist, ax=ax, figure=fig)
        _check_plot_works(self.ts.hist, figure=fig)
        tm.close()

        fig, (ax1, ax2) = self.plt.subplots(1, 2)
        _check_plot_works(self.ts.hist, figure=fig, ax=ax1)
        _check_plot_works(self.ts.hist, figure=fig, ax=ax2)

        with pytest.raises(ValueError):
            self.ts.hist(by=self.ts.index, figure=fig)

    @pytest.mark.slow
    def test_hist_bins_legacy(self):
        df = DataFrame(np.random.randn(10, 2))
        ax = df.hist(bins=2)[0][0]
        assert len(ax.patches) == 2

    @pytest.mark.slow
    def test_hist_layout(self):
        df = self.hist_df
        with pytest.raises(ValueError):
            df.height.hist(layout=(1, 1))

        with pytest.raises(ValueError):
            df.height.hist(layout=[1, 1])

    @pytest.mark.slow
    def test_hist_layout_with_by(self):
        df = self.hist_df

        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.gender, layout=(2, 1))
        self._check_axes_shape(axes, axes_num=2, layout=(2, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.gender, layout=(3, -1))
        self._check_axes_shape(axes, axes_num=2, layout=(3, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.category, layout=(4, 1))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 1))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.category, layout=(2, -1))
        self._check_axes_shape(axes, axes_num=4, layout=(2, 2))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.category, layout=(3, -1))
        self._check_axes_shape(axes, axes_num=4, layout=(3, 2))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.category, layout=(-1, 4))
        self._check_axes_shape(axes, axes_num=4, layout=(1, 4))

        with tm.assert_produces_warning(UserWarning):
            axes = _check_plot_works(df.height.hist,
                                     by=df.classroom, layout=(2, 2))
        self._check_axes_shape(axes, axes_num=3, layout=(2, 2))

        axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))
        self._check_axes_shape(axes, axes_num=4, layout=(4, 2),
                               figsize=(12, 7))

    @pytest.mark.slow
    def test_hist_no_overlap(self):
        from matplotlib.pyplot import subplot, gcf
        x = Series(randn(2))
        y = Series(randn(2))
        subplot(121)
        x.hist()
        subplot(122)
        y.hist()
        fig = gcf()
        axes = fig.axes
        assert len(axes) == 2

    @pytest.mark.slow
    def test_hist_secondary_legend(self):
        # GH 9610
        df = DataFrame(np.random.randn(30, 4), columns=list('abcd'))

        # primary -> secondary
        _, ax = self.plt.subplots()
        ax = df['a'].plot.hist(legend=True, ax=ax)
        df['b'].plot.hist(ax=ax, legend=True, secondary_y=True)
        # both legends are dran on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax, labels=['a', 'b (right)'])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary
        _, ax = self.plt.subplots()
        ax = df['a'].plot.hist(legend=True, secondary_y=True, ax=ax)
        df['b'].plot.hist(ax=ax, legend=True, secondary_y=True)
        # both legends are draw on left ax
        # left axis must be invisible, right axis must be visible
        self._check_legend_labels(ax.left_ax,
                                  labels=['a (right)', 'b (right)'])
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> primary
        _, ax = self.plt.subplots()
        ax = df['a'].plot.hist(legend=True, secondary_y=True, ax=ax)
        # right axes is returned
        df['b'].plot.hist(ax=ax, legend=True)
        # both legends are draw on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax.left_ax, labels=['a (right)', 'b'])
        assert ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

    @pytest.mark.slow
    def test_df_series_secondary_legend(self):
        # GH 9779
        df = DataFrame(np.random.randn(30, 3), columns=list('abc'))
        s = Series(np.random.randn(30), name='x')

        # primary -> secondary (without passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are dran on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax, labels=['a', 'b', 'c', 'x (right)'])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()
        tm.close()

        # primary -> secondary (with passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are dran on left ax
        # left and right axis must be visible
        self._check_legend_labels(ax, labels=['a', 'b', 'c', 'x (right)'])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()
        tm.close()

        # seconcary -> secondary (without passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(legend=True, secondary_y=True, ax=ax)
        # both legends are dran on left ax
        # left axis must be invisible and right axis must be visible
        expected = ['a (right)', 'b (right)', 'c (right)', 'x (right)']
        self._check_legend_labels(ax.left_ax, labels=expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary (with passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(secondary_y=True, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are dran on left ax
        # left axis must be invisible and right axis must be visible
        expected = ['a (right)', 'b (right)', 'c (right)', 'x (right)']
        self._check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

        # secondary -> secondary (with passing ax)
        _, ax = self.plt.subplots()
        ax = df.plot(secondary_y=True, mark_right=False, ax=ax)
        s.plot(ax=ax, legend=True, secondary_y=True)
        # both legends are dran on left ax
        # left axis must be invisible and right axis must be visible
        expected = ['a', 'b', 'c', 'x (right)']
        self._check_legend_labels(ax.left_ax, expected)
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
        tm.close()

    @pytest.mark.slow
    def test_secondary_logy(self):
        # GH 25545
        s1 = Series(np.random.randn(30))
        s2 = Series(np.random.randn(30))

        ax1 = s1.plot(logy=True)
        ax2 = s2.plot(secondary_y=True, logy=True)

        assert ax1.get_yscale() == 'log'
        assert ax2.get_yscale() == 'log'

    @pytest.mark.slow
    def test_plot_fails_with_dupe_color_and_style(self):
        x = Series(randn(2))
        with pytest.raises(ValueError):
            _, ax = self.plt.subplots()
            x.plot(style='k--', color='k', ax=ax)

    @pytest.mark.slow
    @td.skip_if_no_scipy
    def test_hist_kde(self):

        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(logy=True, ax=ax)
        self._check_ax_scales(ax, yaxis='log')
        xlabels = ax.get_xticklabels()
        # ticks are values, thus ticklabels are blank
        self._check_text_labels(xlabels, [''] * len(xlabels))
        ylabels = ax.get_yticklabels()
        self._check_text_labels(ylabels, [''] * len(ylabels))

        _skip_if_no_scipy_gaussian_kde()
        _check_plot_works(self.ts.plot.kde)
        _check_plot_works(self.ts.plot.density)
        _, ax = self.plt.subplots()
        ax = self.ts.plot.kde(logy=True, ax=ax)
        self._check_ax_scales(ax, yaxis='log')
        xlabels = ax.get_xticklabels()
        self._check_text_labels(xlabels, [''] * len(xlabels))
        ylabels = ax.get_yticklabels()
        self._check_text_labels(ylabels, [''] * len(ylabels))

    @pytest.mark.slow
    @td.skip_if_no_scipy
    def test_kde_kwargs(self):
        _skip_if_no_scipy_gaussian_kde()

        sample_points = np.linspace(-100, 100, 20)
        _check_plot_works(self.ts.plot.kde, bw_method='scott', ind=20)
        _check_plot_works(self.ts.plot.kde, bw_method=None, ind=20)
        _check_plot_works(self.ts.plot.kde, bw_method=None, ind=np.int(20))
        _check_plot_works(self.ts.plot.kde, bw_method=.5, ind=sample_points)
        _check_plot_works(self.ts.plot.density, bw_method=.5,
                          ind=sample_points)
        _, ax = self.plt.subplots()
        ax = self.ts.plot.kde(logy=True, bw_method=.5, ind=sample_points,
                              ax=ax)
        self._check_ax_scales(ax, yaxis='log')
        self._check_text_labels(ax.yaxis.get_label(), 'Density')

    @pytest.mark.slow
    @td.skip_if_no_scipy
    def test_kde_missing_vals(self):
        _skip_if_no_scipy_gaussian_kde()

        s = Series(np.random.uniform(size=50))
        s[0] = np.nan
        axes = _check_plot_works(s.plot.kde)

        # gh-14821: check if the values have any missing values
        assert any(~np.isnan(axes.lines[0].get_xdata()))

    @pytest.mark.slow
    def test_hist_kwargs(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 5
        self._check_text_labels(ax.yaxis.get_label(), 'Frequency')
        tm.close()

        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(orientation='horizontal', ax=ax)
        self._check_text_labels(ax.xaxis.get_label(), 'Frequency')
        tm.close()

        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(align='left', stacked=True, ax=ax)
        tm.close()

    @pytest.mark.slow
    @td.skip_if_no_scipy
    def test_hist_kde_color(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.hist(logy=True, bins=10, color='b', ax=ax)
        self._check_ax_scales(ax, yaxis='log')
        assert len(ax.patches) == 10
        self._check_colors(ax.patches, facecolors=['b'] * 10)

        _skip_if_no_scipy_gaussian_kde()
        _, ax = self.plt.subplots()
        ax = self.ts.plot.kde(logy=True, color='r', ax=ax)
        self._check_ax_scales(ax, yaxis='log')
        lines = ax.get_lines()
        assert len(lines) == 1
        self._check_colors(lines, ['r'])

    @pytest.mark.slow
    def test_boxplot_series(self):
        _, ax = self.plt.subplots()
        ax = self.ts.plot.box(logy=True, ax=ax)
        self._check_ax_scales(ax, yaxis='log')
        xlabels = ax.get_xticklabels()
        self._check_text_labels(xlabels, [self.ts.name])
        ylabels = ax.get_yticklabels()
        self._check_text_labels(ylabels, [''] * len(ylabels))

    @pytest.mark.slow
    def test_kind_both_ways(self):
        s = Series(range(3))
        kinds = (plotting._core._common_kinds +
                 plotting._core._series_kinds)
        _, ax = self.plt.subplots()
        for kind in kinds:
            if not _ok_for_gaussian_kde(kind):
                continue
            s.plot(kind=kind, ax=ax)
            getattr(s.plot, kind)()

    @pytest.mark.slow
    def test_invalid_plot_data(self):
        s = Series(list('abcd'))
        _, ax = self.plt.subplots()
        for kind in plotting._core._common_kinds:
            if not _ok_for_gaussian_kde(kind):
                continue
            with pytest.raises(TypeError):
                s.plot(kind=kind, ax=ax)

    @pytest.mark.slow
    def test_valid_object_plot(self):
        s = Series(lrange(10), dtype=object)
        for kind in plotting._core._common_kinds:
            if not _ok_for_gaussian_kde(kind):
                continue
            _check_plot_works(s.plot, kind=kind)

    def test_partially_invalid_plot_data(self):
        s = Series(['a', 'b', 1.0, 2])
        _, ax = self.plt.subplots()
        for kind in plotting._core._common_kinds:
            if not _ok_for_gaussian_kde(kind):
                continue
            with pytest.raises(TypeError):
                s.plot(kind=kind, ax=ax)

    def test_invalid_kind(self):
        s = Series([1, 2])
        with pytest.raises(ValueError):
            s.plot(kind='aasdf')

    @pytest.mark.slow
    def test_dup_datetime_index_plot(self):
        dr1 = date_range('1/1/2009', periods=4)
        dr2 = date_range('1/2/2009', periods=4)
        index = dr1.append(dr2)
        values = randn(index.size)
        s = Series(values, index=index)
        _check_plot_works(s.plot)

    @pytest.mark.slow
    def test_errorbar_plot(self):

        s = Series(np.arange(10), name='x')
        s_err = np.random.randn(10)
        d_err = DataFrame(randn(10, 2), index=s.index, columns=['x', 'y'])
        # test line and bar plots
        kinds = ['line', 'bar']
        for kind in kinds:
            ax = _check_plot_works(s.plot, yerr=Series(s_err), kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, yerr=s_err, kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, yerr=s_err.tolist(), kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, yerr=d_err, kind=kind)
            self._check_has_errorbars(ax, xerr=0, yerr=1)
            ax = _check_plot_works(s.plot, xerr=0.2, yerr=0.2, kind=kind)
            self._check_has_errorbars(ax, xerr=1, yerr=1)

        ax = _check_plot_works(s.plot, xerr=s_err)
        self._check_has_errorbars(ax, xerr=1, yerr=0)

        # test time series plotting
        ix = date_range('1/1/2000', '1/1/2001', freq='M')
        ts = Series(np.arange(12), index=ix, name='x')
        ts_err = Series(np.random.randn(12), index=ix)
        td_err = DataFrame(randn(12, 2), index=ix, columns=['x', 'y'])

        ax = _check_plot_works(ts.plot, yerr=ts_err)
        self._check_has_errorbars(ax, xerr=0, yerr=1)
        ax = _check_plot_works(ts.plot, yerr=td_err)
        self._check_has_errorbars(ax, xerr=0, yerr=1)

        # check incorrect lengths and types
        with pytest.raises(ValueError):
            s.plot(yerr=np.arange(11))

        s_err = ['zzz'] * 10
        # MPL > 2.0.0 will most likely use TypeError here
        with pytest.raises((TypeError, ValueError)):
            s.plot(yerr=s_err)

    # This XPASSES when tested with mpl == 3.0.1
    @td.xfail_if_mpl_2_2
    def test_table(self):
        _check_plot_works(self.series.plot, table=True)
        _check_plot_works(self.series.plot, table=self.series)

    @pytest.mark.slow
    def test_series_grid_settings(self):
        # Make sure plot defaults to rcParams['axes.grid'] setting, GH 9792
        self._check_grid_settings(Series([1, 2, 3]),
                                  plotting._core._series_kinds +
                                  plotting._core._common_kinds)

    @pytest.mark.slow
    def test_standard_colors(self):
        from pandas.plotting._style import _get_standard_colors

        for c in ['r', 'red', 'green', '#FF0000']:
            result = _get_standard_colors(1, color=c)
            assert result == [c]

            result = _get_standard_colors(1, color=[c])
            assert result == [c]

            result = _get_standard_colors(3, color=c)
            assert result == [c] * 3

            result = _get_standard_colors(3, color=[c])
            assert result == [c] * 3

    @pytest.mark.slow
    def test_standard_colors_all(self):
        import matplotlib.colors as colors
        from pandas.plotting._style import _get_standard_colors

        # multiple colors like mediumaquamarine
        for c in colors.cnames:
            result = _get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = _get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = _get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = _get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

        # single letter colors like k
        for c in colors.ColorConverter.colors:
            result = _get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = _get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = _get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = _get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

    def test_series_plot_color_kwargs(self):
        # GH1890
        _, ax = self.plt.subplots()
        ax = Series(np.arange(12) + 1).plot(color='green', ax=ax)
        self._check_colors(ax.get_lines(), linecolors=['green'])

    def test_time_series_plot_color_kwargs(self):
        # #1890
        _, ax = self.plt.subplots()
        ax = Series(np.arange(12) + 1, index=date_range(
            '1/1/2000', periods=12)).plot(color='green', ax=ax)
        self._check_colors(ax.get_lines(), linecolors=['green'])

    def test_time_series_plot_color_with_empty_kwargs(self):
        import matplotlib as mpl

        def_colors = self._unpack_cycler(mpl.rcParams)
        index = date_range('1/1/2000', periods=12)
        s = Series(np.arange(1, 13), index=index)

        ncolors = 3

        _, ax = self.plt.subplots()
        for i in range(ncolors):
            ax = s.plot(ax=ax)
        self._check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])

    def test_xticklabels(self):
        # GH11529
        s = Series(np.arange(10), index=['P%02d' % i for i in range(10)])
        _, ax = self.plt.subplots()
        ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
        exp = ['P%02d' % i for i in [0, 3, 5, 9]]
        self._check_text_labels(ax.get_xticklabels(), exp)

    def test_custom_business_day_freq(self):
        # GH7222
        from pandas.tseries.offsets import CustomBusinessDay
        s = Series(range(100, 121), index=pd.bdate_range(
            start='2014-05-01', end='2014-06-01',
            freq=CustomBusinessDay(holidays=['2014-05-26'])))

        _check_plot_works(s.plot)

    @pytest.mark.xfail
    def test_plot_accessor_updates_on_inplace(self):
        s = Series([1, 2, 3, 4])
        _, ax = self.plt.subplots()
        ax = s.plot(ax=ax)
        before = ax.xaxis.get_ticklocs()

        s.drop([0, 1], inplace=True)
        _, ax = self.plt.subplots()
        after = ax.xaxis.get_ticklocs()
        tm.assert_numpy_array_equal(before, after)
