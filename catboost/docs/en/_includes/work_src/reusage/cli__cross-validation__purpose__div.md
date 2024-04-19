
Training can be launched in cross-validation mode. In this case, only the training dataset is required. This dataset is split, and the resulting folds are used as the learning and evaluation datasets. If the input dataset contains the {{ cd-file__col-type__GroupId }} column, all objects from one group are added to the same fold.

Each cross-validation run from the command-line interface launches one training out of N trainings in N-fold cross-validation.

Use one of the following methods to get aggregated N-fold cross-validation results:
- Run the training in cross-validation mode from the command-line interface N times with different validation folds and aggregate results by hand.
- Use the [cv](../../../concepts/python-reference_cv.md) function of the [Python package](../../../concepts/python-quickstart.md) instead of the command-line version. It returns aggregated results out-of-the-box.
