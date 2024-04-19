# Profiler information

#### {{ output--contains }}

Detailed information about the duration of different operations within each iteration of training.

#### {{ output--format }}

- The resulting file consists of blocks of information regarding a certain iteration and the average times of execution of different operations.
- The block of information regarding a certain iteration starts from a number representing a zero-based index of this iteration.
- Each of the following lines in the iteration block with the exception of the last line contains information regarding a single operation in the format:
    ```
    <verbal description operation>: <time> <sec|ms>
    ```

- The last line in the iteration block starts with a tab and contains tab-separated values of the time that has passed since training started and the remaining time until the end of training.
- The last block in the file contains information regarding the average times of execution of different operations and starts with the `Average times:` string on a new line.
- Each of the following lines in the block with average times of execution contains information regarding a single operation in the format:
    ```
    <verbal description operation>: <time> <sec|ms>
    ```


#### {{ output--example }}

```
0
Profile:
Bootstrap, depth 0: 7.3e-05 sec
Bootstrap, depth 1: 3.59e-05 sec
Bootstrap, depth 2: 3.5e-05 sec
Bootstrap, depth 3: 5.78e-05 sec
Bootstrap, depth 4: 5.11e-05 sec
Bootstrap, depth 5: 4.68e-05 sec
Calc derivatives: 0.000106 sec
Calc errors: 0.000239 sec
Calc scores 0: 0.000641 sec
Calc scores 1: 0.000923 sec
Calc scores 2: 0.00157 sec
Calc scores 3: 0.00282 sec
Calc scores 4: 0.0048 sec
Calc scores 5: 0.00827 sec
CalcApprox result leafs: 7.88e-05 sec
CalcApprox tree struct and update tree structure approx: 0.000424 sec
ComputeOnlineCTRs for tree struct (train folds and test fold): 4.73e-06 sec
Select best split 0: 1.77e-05 sec
Select best split 1: 1.54e-05 sec
Select best split 2: 3.35e-05 sec
Select best split 3: 2.93e-05 sec
Select best split 4: 3.4e-05 sec
Select best split 5: 2.68e-05 sec
Update final approxes: 2.83e-06 sec
Passed: 0.0203 sec
	total: 20.3ms	remaining: 20.3s
1
Profile:
Bootstrap, depth 0: 4.83e-05 sec
Bootstrap, depth 1: 3.41e-05 sec
Bootstrap, depth 2: 4.06e-05 sec
Bootstrap, depth 3: 6.94e-05 sec
Bootstrap, depth 4: 6.44e-05 sec
Bootstrap, depth 5: 5.76e-05 sec
Calc derivatives: 0.00012 sec
Calc errors: 0.000195 sec
Calc scores 0: 0.000607 sec
Calc scores 1: 0.00194 sec
Calc scores 2: 0.00147 sec
Calc scores 3: 0.00293 sec
Calc scores 4: 0.00452 sec
Calc scores 5: 0.00899 sec
CalcApprox result leafs: 7.13e-05 sec
CalcApprox tree struct and update tree structure approx: 0.000525 sec
ComputeOnlineCTRs for tree struct (train folds and test fold): 5.67e-06 sec
Select best split 0: 1.63e-05 sec
Select best split 1: 2.46e-05 sec
Select best split 2: 2.62e-05 sec
Select best split 3: 4.57e-05 sec
Select best split 4: 3.46e-05 sec
Select best split 5: 2.93e-05 sec
Update final approxes: 2.71e-06 sec
Passed: 0.0219 sec
	total: 42.2ms	remaining: 21.1s
...

Average times:
Iteration time: 0.0223 sec
Bootstrap, depth 0: 7.13e-05 sec
Bootstrap, depth 1: 4.95e-05 sec
Bootstrap, depth 2: 4.91e-05 sec
Bootstrap, depth 3: 5.1e-05 sec
Bootstrap, depth 4: 5.28e-05 sec
Bootstrap, depth 5: 5.68e-05 sec
Calc derivatives: 0.000103 sec
Calc errors: 0.000223 sec
Calc scores 0: 0.000769 sec
Calc scores 1: 0.0011 sec
Calc scores 2: 0.00169 sec
Calc scores 3: 0.00278 sec
Calc scores 4: 0.00503 sec
Calc scores 5: 0.00915 sec
CalcApprox result leafs: 8.78e-05 sec
CalcApprox tree struct and update tree structure approx: 0.000598 sec
ComputeOnlineCTRs for tree struct (train folds and test fold): 7.34e-06 sec
Select best split 0: 7.16e-05 sec
Select best split 1: 6.31e-05 sec
Select best split 2: 6.34e-05 sec
Select best split 3: 5.98e-05 sec
Select best split 4: 5.86e-05 sec
Select best split 5: 4.66e-05 sec
Update final approxes: 3.72e-06 sec
```

