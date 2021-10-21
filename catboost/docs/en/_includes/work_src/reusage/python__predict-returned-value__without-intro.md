
- Single object — The returned value depends on the specified value of the `prediction_type` parameter:
    - {{ prediction-type--RawFormulaVal }} — Raw formula value.
    
    - {{ prediction-type--Class }} — Class label.
    
    - {{ prediction-type--Probability }} — One-dimensional {{ python-type__np_ndarray }} with the probability for every class.
    
- Multiple objects — The returned value depends on the specified value of the `prediction_type` parameter:
    - {{ prediction-type--RawFormulaVal }} — One-dimensional {{ python-type__np_ndarray }} of raw formula values (one for each object).
    
    - {{ prediction-type--Class }} — One-dimensional {{ python-type__np_ndarray }} of class label (one for each object).
    
    - {{ prediction-type--Probability }} — Two-dimensional {{ python-type__np_ndarray }} of shape `(number_of_objects, number_of_classes)` with the probability for every class for each object.
    
