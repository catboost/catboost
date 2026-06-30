
Must be in the form of a one- or two- dimensional array. The type of data in the array depends on the machine learning task being solved:
- Binary classification
    One-dimensional array containing one of:

    * Booleans, integers or strings that represent the labels of the classes (only two unique values).
    * Numeric values.
        The interpretation of numeric values depends on the selected loss function:

        - {{ error-function--Logit }} — The value is considered a positive class if it is strictly greater than the value of the `target_border` training parameter. Otherwise, it is considered a negative class.
        - {{ error-function--CrossEntropy }} — The value is interpreted as the probability that the dataset object belongs to the positive class. Possible values are in the range `[0; 1]`.

- Multiclassification — One-dimensional array of integers or strings that represent the labels of the classes.
- Multi label classification
    Two-dimensional array. The first index is for a label/class, the second index is for an object.

    Possible values depend on the selected loss function:

    * MultiLogloss — Only {0, 1} or {False, True} values are allowed that specify whether an object belongs to the class corresponding to the first index.
    * MultiCrossEntropy — Numerical values in the range `[0; 1]` that are interpreted as the probability that the dataset object belongs to the class corresponding to the first index.
