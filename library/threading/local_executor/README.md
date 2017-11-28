# Library for parallel task execution in thread pool

This library allows easy parallelization of existing code and cycles.
It provides `NPar::TLocalExecutor` class and `NPar::LocalExecutor()` singleton accessor.
At start, `TLocalExecutor` has no threads in thread pool and all async tasks will be queued for later execution when extra threads appear.
All tasks should be `NPar::ILocallyExecutable` child class or function equal to `std::function<void(int)>`

## TLocalExecutor methods

`TLocalExecutor::Run(int threadcount)` - add threads to thread pool (**WARNING!** `Run(threadcount)` will *add* `threadcount` threads to pool)

`void TLocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags)` - run one task and pass id as task function input, flags - bitmask composition of:

- `TLocalExecutor::HIGH_PRIORITY = 0` - put task in high priority queue
- `TLocalExecutor::MED_PRIORITY = 1` - put task in medium priority queue
- `TLocalExecutor::LOW_PRIORITY = 2` - put task in low priority queue
- `TLocalExecutor::WAIT_COMPLETE = 4` - wait for task completion

`void TLocalExecutor::ExecRange(TLocallyExecutableFunction exec, TExecRangeParams blockParams, int flags);` - run range of tasks `[TExecRangeParams::FirstId, TExecRangeParams::LastId).`

`flags` is the same as for `TLocalExecutor::Exec`.

`TExecRangeParams` is a structure that describes the range.
By default each task is executed separately. Threads from thread pool are taking
the tasks in the manner first come first serve.

It is also possible to partition range of tasks in consequtive blocks and execute each block as a bigger task.
`TExecRangeParams::SetBlockCountToThreadCount()` will result in thread count tasks,
    where thread count is the count of threads in thread pool.
    each thread will execute approximately equal count of tasks from range.

`TExecRangeParams::SetBlockSize()` and `TExecRangeParams::SetBlockCount()` will partition
the range of tasks into consequtive blocks of approximately given size, or of size calculated
     by partitioning the range into approximately equal size blocks of given count.

## Examples

### Simple task async exec with medium priority

```cpp
using namespace NPar;

LocalExecutor().Run(4);
TEvent event;
LocalExecutor().Exec([](int) {
    SomeFunc();
    event.Signal();
}, 0, TLocalExecutor::MED_PRIORITY);

SomeOtherCode();
event.WaitI();
```

### Execute task range and wait completion

```cpp
using namespace NPar;

LocalExecutor().Run(4);
LocalExecutor().ExecRange([](int id) {
    SomeFunc(id);
}, TExecRangeParams(0, 10), TLocalExecutor::WAIT_COMPLETE | TLocalExecutor::MED_PRIORITY);
```

### Exception handling

By default if a not caught exception arise in a task which runs through the Local Executor, then std::terminate() will be called immediately. The exception will be printed to stderr before the termination. Best practice is to handle exception within a task, or avoid throwing exceptions at all for performance reasons.

However, if you'd like to handle and/or rethrow exceptions outside of a range, you can use ExecRangeWithFuture().
It returns vector [0 .. LastId-FirstId] elements, where i-th element is a TFuture corresponding to task with id = (FirstId + i).
Use method .HasValue() of the element to check in Async mode if the corresponding task is complete.
Use .GetValue() or .GetValueSync() to wait for completion of the corresponding task. GetValue() and GetValueSync() will also rethrow an exception if it appears during execution of the task.

You may also use ExecRangeWithThrow() to just receive an exception from a range if it appears. It rethrows an exception from a task with minimal id if such an exception exists, and guarantees normal flow if no exception arise.
