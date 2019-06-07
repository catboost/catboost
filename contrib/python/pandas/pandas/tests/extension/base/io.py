import numpy as np
import pytest

from pandas.compat import StringIO

import pandas as pd

from .base import BaseExtensionTests


class BaseParsingTests(BaseExtensionTests):

    @pytest.mark.parametrize('engine', ['c', 'python'])
    def test_EA_types(self, engine, data):
        df = pd.DataFrame({
            'with_dtype': pd.Series(data, dtype=str(data.dtype))
        })
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        result = pd.read_csv(StringIO(csv_output), dtype={
            'with_dtype': str(data.dtype)
        }, engine=engine)
        expected = df
        self.assert_frame_equal(result, expected)
