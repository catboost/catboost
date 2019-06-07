import os

import pytest

from pandas import read_csv, read_table


class BaseParser(object):
    engine = None
    low_memory = True
    float_precision_choices = []

    def update_kwargs(self, kwargs):
        kwargs = kwargs.copy()
        kwargs.update(dict(engine=self.engine,
                           low_memory=self.low_memory))

        return kwargs

    def read_csv(self, *args, **kwargs):
        kwargs = self.update_kwargs(kwargs)
        return read_csv(*args, **kwargs)

    def read_table(self, *args, **kwargs):
        kwargs = self.update_kwargs(kwargs)
        return read_table(*args, **kwargs)


class CParser(BaseParser):
    engine = "c"
    float_precision_choices = [None, "high", "round_trip"]


class CParserHighMemory(CParser):
    low_memory = False


class CParserLowMemory(CParser):
    low_memory = True


class PythonParser(BaseParser):
    engine = "python"
    float_precision_choices = [None]


@pytest.fixture
def csv_dir_path(datapath):
    return datapath("io", "parser", "data")


@pytest.fixture
def csv1(csv_dir_path):
    return os.path.join(csv_dir_path, "test1.csv")


_cParserHighMemory = CParserHighMemory()
_cParserLowMemory = CParserLowMemory()
_pythonParser = PythonParser()

_py_parsers_only = [_pythonParser]
_c_parsers_only = [_cParserHighMemory, _cParserLowMemory]
_all_parsers = _c_parsers_only + _py_parsers_only

_py_parser_ids = ["python"]
_c_parser_ids = ["c_high", "c_low"]
_all_parser_ids = _c_parser_ids + _py_parser_ids


@pytest.fixture(params=_all_parsers,
                ids=_all_parser_ids)
def all_parsers(request):
    return request.param


@pytest.fixture(params=_c_parsers_only,
                ids=_c_parser_ids)
def c_parser_only(request):
    return request.param


@pytest.fixture(params=_py_parsers_only,
                ids=_py_parser_ids)
def python_parser_only(request):
    return request.param
