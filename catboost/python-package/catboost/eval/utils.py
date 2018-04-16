"""
Module that contains different utils functions.
"""
import os


def make_dirs_if_not_exists(name):
    try:
        os.makedirs(name)
    except OSError:
        pass


def series_to_line(row, sep):
    return sep.join(map(str, row.tolist()))


def save_plot(fig, file_name=None):
    try:
        from plotly.offline import iplot, init_notebook_mode
        init_notebook_mode(connected=True)
        iplot(fig, filename=file_name)
    except:
        from plotly.offline import plot
        plot(fig, auto_open=False, filename=file_name)
