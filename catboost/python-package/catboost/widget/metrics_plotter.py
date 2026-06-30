import time
import logging

import typing as tp

from IPython.display import display
from copy import deepcopy
from typing import List, Optional, Any, Union

from .ipythonwidget import MetricWidget

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MetricsWidget(MetricWidget):
    def __init__(self):
        super(self.__class__, self).__init__()

    def update_data(self, data: tp.Dict) -> None:
        # deepcopy is crucial here
        self.data = deepcopy(data)


class MetricsPlotter:
    """
    Context manager that enables widget with learning curves in
    JupyterLab / Jupyter Notebook
    """

    def __init__(self, train_metrics: List[Union[str, tp.Dict[str, str]]],
                 test_metrics: Optional[List[Union[str, tp.Dict[str, str]]]] = None,
                 total_iterations: Optional[int] = None) -> None:
        """
        Constructor that defines metrics to be plotted and total iterations count.

        Parameters
        ----------
        train_metrics : list of str or list of dict
            List of train metrics to be tracked.
            Each item in list can be either string with metric name or dict
            with the following format:
            {
                "name": "{metric_name}",
                "best_value": "Max|Min|Undefined",
            }

        test_metrics : list of str or list of dict, optional (default=None)
            List of test metrics to be tracked.
            Has the same format as train_metrics. Equals to train_metrics, if not defined

        total_iterations: int, optional (default=None)
            Total number of iterations, allows for remaining time estimation.
        """

        self._widget = MetricsWidget()

        self._values = {
            "iterations": [],
            "meta": {
                "launch_mode": "Train",
                "parameters": "",
                "name": "experiment",
                "iteration_count": None,  # set later
                "learn_sets": ["learn"],
                "learn_metrics": None,  # set later
                "test_sets": ["test"],
                "test_metrics": None,  # set later
            }
        }

        self._content = {
            "passed_iterations": 0,
            "total_iterations": None,  # set later
            "data": self._values,
        }

        # data propagated to widget class
        self._data = {
            "test_path": {
                "path": "test_path",
                "name": "experiment",
                "content": self._content,
            }
        }

        test_metrics = test_metrics or train_metrics
        train_metrics_meta: List[tp.Dict[str, str]] = self.construct_metrics_meta(train_metrics)
        test_metrics_meta: List[tp.Dict[str, str]] = self.construct_metrics_meta(test_metrics)

        self._train_metrics_positions = {
            meta["name"]: pos for pos, meta in enumerate(train_metrics_meta)}
        self._test_metrics_positions = {
            meta["name"]: pos for pos, meta in enumerate(test_metrics_meta)}

        self._values["meta"].update({
            "learn_metrics": train_metrics_meta,
            "test_metrics": test_metrics_meta,
        })

        self.passed_iterations = 0
        self.total_iterations = total_iterations

        if total_iterations is not None:
            self._values["meta"]["iteration_count"] = total_iterations

        self._content["total_iterations"] = total_iterations or 0

        self._start_time = time.time()

    def __enter__(self) -> 'MetricsPlotter':
        display(self._widget)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Any:
        if exc_type == KeyboardInterrupt:
            logger.info(
                f"Learning was stopped manually after {self.passed_iterations} epochs")
            return True

    @staticmethod
    def construct_metrics_meta(metrics: List[Union[str, tp.Dict[str, str]]]) -> List[tp.Dict[str, str]]:
        meta: List[tp.Dict[str, str]] = []
        for item in metrics:
            if isinstance(item, str):
                name, best_value = item, "Undefined"
            elif isinstance(item, dict):
                assert "name" in item and "best_value" in item, \
                    "Wrong metrics definition format: should have " \
                    "`name` and `best_value` fields"
                name, best_value = item["name"], item["best_value"]
            else:
                assert False, "Each metric should be defined as str or as" \
                    "dict with `name` and `best_value` fields"
            meta.append({"best_value": best_value, "name": name})
        return meta

    @staticmethod
    def construct_metrics_array(metrics_positions: tp.Dict[str, int],
                                metrics: tp.Dict[str, float]) -> List[float]:
        array: List[float] = [0.] * len(metrics_positions)

        # data validation
        assert set(metrics.keys()) == set(metrics_positions.keys()), \
            f"Not all metrics were passed while logging, expected " \
            f"following: {', '.join(list(metrics_positions.keys()))}"

        for metric, value in metrics.items():
            assert isinstance(value, float), "Type of metric {metric} should be float"
            array[metrics_positions[metric]] = value
        return array

    def estimate_remaining_time(self, time_from_start: float) -> Optional[float]:
        if self.total_iterations is None:
            return None
        remaining_iterations: int = self.total_iterations - self.passed_iterations
        return time_from_start / self.passed_iterations * remaining_iterations

    def log(self, epoch: int, train: bool, metrics: tp.Dict[str, float]) -> None:
        """
        Save metrics at specific training epoch.

        Parameters
        ----------
        epoch : int
            Current epoch

        train : bool
            Flag that indicates whether metrics are calculated on train or test data

        metrics: dict
            Values for each of metrics defined in `__init__` method of this class
        """

        self.passed_iterations = epoch + 1
        self._content["passed_iterations"] = self.passed_iterations
        total_iterations = max(self._content["total_iterations"], self.passed_iterations)

        self._content["total_iterations"] = total_iterations
        self._values["meta"]["iteration_count"] = total_iterations

        assert len(self._values["iterations"]) in (epoch, epoch + 1), \
            "Data for epochs should be passed successively (wrong epoch number)"

        time_from_start: float = time.time() - self._start_time

        should_redraw: bool = len(self._values["iterations"]) == epoch + 1

        if len(self._values["iterations"]) == epoch:
            self._values["iterations"].append({
                "learn": [],
                "test": [],
                "iteration": epoch,
                "passed_time": time_from_start,
            })

            remaining_time = self.estimate_remaining_time(time_from_start)
            if remaining_time is not None:
                self._values["iterations"][-1]["remaining_time"] = remaining_time

        key: str = "learn" if train else "test"
        value: List[float] = self.construct_metrics_array(
            self._train_metrics_positions if train else self._test_metrics_positions, metrics)

        self._values["iterations"][-1].update({key: value})

        if should_redraw:
            self._widget.update_data(self._data)
