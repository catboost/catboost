Model model_gbm.onnx and predictions predict were obtained as follows.

````
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
model = LGBMRegressor()
number_features = 50

# (train, test) were loaded from files train and test in current directory

label = 2 * train[:, 0] + 3 * train[:, 1]

model.fit(train, label)
lightgbm_predict = model.predict(test)


import onnxmltools
from onnxconverter_common import *
from onnxmltools.convert.common.data_types import FloatTensorType
model_onnx = onnxmltools.convert_lightgbm(model, name='LightGBM',initial_types=[('input', FloatTensorType([0, number_features]))])
onnxmltools.utils.save_model(model_onnx, 'model_gbm.onnx')
````

Versions: 

+ LightGbm 2.2.3
+ onnxconverter-common 1.5.2
+ onnxmltools 1.5.0