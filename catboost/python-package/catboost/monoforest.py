import math

from . import _catboost
from .core import CatBoost, CatBoostError

from .utils import _import_matplotlib

FeatureExplanation = _catboost.FeatureExplanation


def _check_model(model):
    if not isinstance(model, CatBoost):
        raise CatBoostError("Model should be CatBoost")


def to_polynom(model):
    _check_model(model)
    return _catboost.to_polynom(model._object)


def to_polynom_string(model):
    _check_model(model)
    return _catboost.to_polynom_string(model._object)


def explain_features(model):
    _check_model(model)
    return _catboost.explain_features(model._object)


def calc_features_strength(model):
    explanations = explain_features(model)
    features_strength = [expl.calc_strength() for expl in explanations]
    return features_strength


def plot_pdp(arg, size_per_plot=(5, 5), plots_per_row=None):
    with _import_matplotlib() as _plt:
        plt = _plt
    if isinstance(arg, CatBoost):
        arg = explain_features(arg)
    if isinstance(arg, _catboost.FeatureExplanation):
        arg = [arg]
    assert len(arg) > 0
    assert isinstance(arg, list)
    for element in arg:
        assert isinstance(element, _catboost.FeatureExplanation)

    figs = []
    for feature_explanation in arg:
        dimension = feature_explanation.dimension()
        if not plots_per_row:
            plots_per_row = min(5, dimension)
        rows = int(math.ceil(dimension / plots_per_row))
        fig, axes = plt.subplots(rows, plots_per_row)
        fig.suptitle("Feature #{}".format(feature_explanation.feature))
        if rows == 1:
            axes = [axes]
        if plots_per_row == 1:
            axes = [[row_axes] for row_axes in axes]
        fig.set_size_inches(size_per_plot[0] * plots_per_row, size_per_plot[1] * rows)

        for dim in range(dimension):
            ax = axes[dim // plots_per_row][dim % plots_per_row]
            ax.set_title("Dimension={}".format(dim))
            ax.set_xlabel("feature value")
            ax.set_ylabel("model value")

            borders, values = feature_explanation.calc_pdp(dim)
            xs = []
            ys = []
            if feature_explanation.type == "Float":
                if len(borders) == 0:
                    xs.append(-0.1)
                    xs.append(0.1)
                    ys.append(feature_explanation.expected_bias[dim])
                    ys.append(feature_explanation.expected_bias[dim])
                    ax.plot(xs, ys)
                else:
                    offset = max(0.1, (borders[0] + borders[-1]) / 2)
                    xs.append(borders[0] - offset)
                    ys.append(feature_explanation.expected_bias[dim])
                    for border, value in zip(borders, values):
                        xs.append(border)
                        ys.append(ys[-1])
                        xs.append(border)
                        ys.append(value)
                    xs.append(borders[-1] + offset)
                    ys.append(ys[-1])
                    ax.plot(xs, ys)
            else:
                xs = ['bias'] + list(map(str, borders))
                ys = feature_explanation.expected_bias[dim] + values
                ax.bar(xs, ys)
        figs.append(fig)

    return figs


def plot_features_strength(model, height_per_feature=0.5, width_per_plot=5, plots_per_row=None):
    with _import_matplotlib() as _plt:
        plt = _plt
    strengths = calc_features_strength(model)
    dimension = len(strengths[0])
    features = len(strengths)
    if not plots_per_row:
        plots_per_row = min(5, dimension)
    rows = int(math.ceil(dimension / plots_per_row))
    fig, axes = plt.subplots(rows, plots_per_row)
    if rows == 1:
        axes = [axes]
    if plots_per_row == 1:
        axes = [[row_axes] for row_axes in axes]
    fig.suptitle("Features Strength")
    fig.set_size_inches(width_per_plot * plots_per_row, height_per_feature * features * rows)

    for dim in range(dimension):
        strengths = [(s[dim], i) for i, s in enumerate(strengths)]
        # strengths = list(reversed(sorted(strengths)))
        strengths = list(sorted(strengths))
        labels = ["Feature #{}".format(f) for _, f in strengths]
        strengths = [s for s, _ in strengths]

        ax = axes[dim // plots_per_row][dim % plots_per_row]
        colors = [(1, 0, 0) if s > 0 else (0, 0, 1) for s in strengths]
        ax.set_title("Dimension={}".format(dim))
        ax.barh(range(len(strengths)), strengths, align='center', color=colors)
        ax.set_yticks(range(len(strengths)))
        ax.set_yticklabels(labels)
        # ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Prediction value change')

    return fig
