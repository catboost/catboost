
Must be in the form of a one-dimensional array. The type of data in the array depends on the machine learning task being solved:
- Regression , multiregression and ranking  — Numeric values.
- Binary classification — Numeric values.
    
    The interpretation of numeric values depends on the selected loss function:
    
    - {{ error-function--Logit }} — The value is considered a positive class if it is strictly greater than the value of the `` parameter of the loss function. Otherwise, it is considered a negative class.
    - {{ error-function--CrossEntropy }} — The value is interpreted as the probability that the dataset object belongs to the positive class. Possible values are in the range `[0; 1]`.
    
- Multiclassification — Integers or strings that represents the labels of the classes.
