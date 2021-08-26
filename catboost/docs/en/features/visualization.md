# Data visualization

{{ product }} provides tools for the [{{ python-package }}](../concepts/python-installation.md) that allow plotting charts with different training statistics. This information can be accessed both during and after the training procedure. [Additional packages](../installation/python-installation-additional-data-visualization-packages.md) must be installed to support the visualization tools.

The following information is reflected on the charts:
- [metric](../concepts/loss-functions.md) values
- best metric values on the validation dataset
- elapsed time of training
- remaining time of training
- current metric value
- metric value on the best iteration

The table below lists the [Python training parameters](../references/training-parameters/index.md) that affect visualization.

**Parameter** | **Usage tips**
:-------- | :---------
`plot` | Set to <q>true</q>
`--name` | The given value is used for signing the charts of the corresponding experiment. This parameter is useful when viewing results of different experiments on one chart.
`custom_metric`, `--loss-function`, `--eval-metric` | All the metrics specified in these parameters are output.


The following applications can be used for viewing the charts:
- [Jupyter Notebook](visualization_jupyter-notebook.md)
- [TensorBoard](visualization_tensorboard.md)

