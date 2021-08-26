
Return the values of all training parameters (including the ones that are not explicitly specified by users).

If the value of a parameter is not explicitly specified, it is set to the default value. In some cases, these default values change dynamically depending on dataset properties and values of user-defined parameters. For example, in classification mode the default learning rate changes depending on the number of iterations and the dataset size. This method returns the values of all parameters, including the ones that are calculated during the training.
