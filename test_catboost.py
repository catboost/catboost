from catboost import CatBoostClassifier, Pool
import numpy as np

X = np.random.rand(100, 10)
y = np.random.randint(0, 3, 100)

model = CatBoostClassifier(iterations=10, loss_function='MultiClass', allow_writing_files=False)
model.fit(X, y, verbose=False)

try:
    print("Trying get_object_importance...")
    model.get_object_importance(Pool(X, y), Pool(X, y), type='Average', update_method='SinglePoint')
    print("Success")
except Exception as e:
    print("Failed!")
    print(e)
