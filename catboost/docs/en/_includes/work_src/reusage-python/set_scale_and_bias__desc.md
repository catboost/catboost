
Set the scale and bias of a trained model. Calling `fit` without `init_model`
trains a new model with its own scale and bias.

With array-like training data, pass `baseline` to `fit` to supply initial
predictions. With a `Pool`, call `set_baseline` on the `Pool` before fitting.
For training data stored in a file, first load it into a `Pool`.
