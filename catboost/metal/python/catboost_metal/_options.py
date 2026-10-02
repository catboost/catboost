"""Validated CatBoost loss descriptions for the Metal estimators."""

import math


def parse_loss(description, *, classifier=False):
    if description is None:
        description = "Logloss" if classifier else "RMSE"
    if not isinstance(description, str) or not description:
        raise ValueError("loss_function must be a CatBoost loss description string.")
    base, separator, arguments = description.partition(":")
    supported = {"Logloss", "CrossEntropy", "MultiClass", "MultiClassOneVsAll",
                 "MultiLogloss", "MultiCrossEntropy"} if classifier else {
        "RMSE", "Poisson", "Huber", "Expectile", "Lq", "Tweedie", "LogLinQuantile",
        "Quantile", "MAE", "MAPE", "MultiRMSE", "RMSEWithUncertainty"}
    if base not in supported:
        raise ValueError(f"This estimator does not yet support loss_function={base!r}.")
    parameters = {}
    if separator:
        for item in arguments.split(";"):
            key, equals, value = item.partition("=")
            if not equals or not key or key in parameters:
                raise ValueError("Invalid or duplicate loss parameter.")
            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError("Loss parameters must be finite numbers.") from exc
            if not math.isfinite(number):
                raise ValueError("Loss parameters must be finite numbers.")
            parameters[key] = number
    expected = {"Huber": "delta", "Expectile": "alpha", "Lq": "q",
                "Tweedie": "variance_power"}.get(base)
    optional = {"LogLinQuantile": {"alpha"}, "Quantile": {"alpha", "delta"}}.get(base, set())
    required = {expected} if expected else set()
    if not required.issubset(parameters) or set(parameters) - required - optional:
        raise ValueError(f"{base} requires exactly the {expected!r} parameter." if expected else
                         f"Loss parameters for {base} are not supported by this adapter.")
    parameter = parameters[expected] if expected else parameters.get(
        "alpha", 0.5 if base in ("LogLinQuantile", "Quantile", "MAE") else 1.0)
    if base == "Huber" and parameter < 0:
        raise ValueError("Huber delta must be nonnegative.")
    if base == "Expectile" and not 0 <= parameter <= 1:
        raise ValueError("Expectile alpha must be in [0, 1].")
    if base == "Lq" and parameter < 1:
        raise ValueError("Lq q must be at least 1.")
    if base == "Tweedie" and not 1 < parameter < 2:
        raise ValueError("Tweedie variance_power must be in (1, 2).")
    if base in ("LogLinQuantile", "Quantile", "MAE") and not 0 <= parameter <= 1:
        raise ValueError("Quantile alpha must be in [0, 1].")
    if base == "MAE" and parameter != 0.5:
        raise ValueError("MAE alpha must equal 0.5.")
    if "delta" in parameters and base in ("Quantile", "MAE") and not 0 <= parameters["delta"] <= 0.01:
        raise ValueError("Quantile delta must be in [0, 0.01].")
    # Return the original description for standard model metadata, alongside
    # the native objective selector and its scalar tuning parameter.
    return description, base, parameters, parameter
