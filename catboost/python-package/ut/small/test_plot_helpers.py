import io
from unittest import mock

import pytest


def test_save_plot_file_plotly_7_compat():
    # plotly 7.x removed the `show_link` option from `plotly.offline.plot`,
    # so `save_plot_file` must not pass it (see GH issue #3160).
    pytest.importorskip("plotly.offline")

    from catboost.plot_helpers import save_plot_file

    buf = io.StringIO()
    fig = {"data": [], "layout": {}}
    with mock.patch("plotly.offline.plot", return_value="<div></div>") as mocked_plot:
        save_plot_file(buf, "test-plot", [fig])

    kwargs = mocked_plot.call_args.kwargs
    assert "show_link" not in kwargs
    assert kwargs["output_type"] == "div"
    assert kwargs["include_plotlyjs"] is False
