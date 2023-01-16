
This parameter defines the step to iterate over the range `[--ntree-start; --ntree-end)`. For example, let's assume that the following parameter values are set:

- `--ntree-start` is set 0
- `--ntree-end` is set to N (the total tree count)
- `--eval-period` is set to 2

In this case, the results are returned for the following tree ranges: `[0, 2)`, `[0, 4)`, ... , `[0, N)`.
