# Library for parallel task execution in a thread pool

This library allows easy parallelization of the existing code, particularly loops.
It provides `NPar::TLocalExecutor` class and `NPar::LocalExecutor()` singleton accessor.
At the start, `TLocalExecutor` has no threads in the thread pool and all async tasks will be queued for later execution when extra threads appear.
All tasks should be either derived from `NPar::ILocallyExecutable` or be of type `std::function<void(int)>`.

## TLocalExecutor methods

`TLocalExecutor::Run(int threadcount)` - add threads to the thread pool (**WARNING!** `Run(threadcount)` will *add* `threadcount` threads to pool)

`void TLocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags)` - run a single task and pass `id` as a task function argument, flags - bitmask that can contain:

- `TLocalExecutor::HIGH_PRIORITY = 0` - put the task in the high priority queue
- `TLocalExecutor::MED_PRIORITY = 1` - put the task in the medium priority queue
- `TLocalExecutor::LOW_PRIORITY = 2` - put the task in the low priority queue
- `TLocalExecutor::WAIT_COMPLETE = 4` - wait for the task completion

`void TLocalExecutor::ExecRange(TLocallyExecutableFunction exec, TExecRangeParams blockParams, int flags);` - run a range of tasks with ids `[TExecRangeParams::FirstId, TExecRangeParams::LastId).`

`flags` is the same as for `TLocalExecutor::Exec`.

By default each task for each `id` is executed separately. Threads from the thread pool are taking the tasks in the FIFO manner.

It is also possible to partition a range of tasks to consecutive blocks and execute each block as a bigger task.

`TExecRangeParams` is a structure that is used for that.

`TExecRangeParams::SetBlockCountToThreadCount()`  will partition
the range of tasks into consecutive blocks with the number of tasks equivalent to the number of threads in the execution pool. The intent is that each thread will take an exactly single block from this partition, although it is not guaranteed, especially if the thread pool is already busy.

`TExecRangeParams::SetBlockSize(TBlockSize blockSize)` will partition
the range of tasks into consecutive blocks of the size approximately equal to `blockSize`.

`TExecRangeParams::SetBlockCount(TBlockCount blockCount)` will partition
the range of tasks into consecutive `blockCount` blocks with the approximately equal size.

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

### Execute a task range and wait for completion

```cpp
using namespace NPar;

LocalExecutor().Run(4);
LocalExecutor().ExecRange([](int id) {
    SomeFunc(id);
}, TExecRangeParams(0, 10), TLocalExecutor::WAIT_COMPLETE | TLocalExecutor::MED_PRIORITY);
```

### Exception handling

By default if an uncaught exception is thrown in a task that runs through the Local Executor, then `std::terminate()` will be called immediately. Best practice is to handle exception within a task, or avoid throwing exceptions at all for performance reasons.

However, if you'd like to get exceptions that might have occured during the tasks execution instead, you can use `ExecRangeWithFutures()`.
It returns a vector of [0 .. LastId-FirstId] elements, where i-th element is a `TFuture` corresponding to the task with `id = (FirstId + i)`.
Use a method `.HasValue()` of the element to check in Async mode if the corresponding task is complete.
Use `.GetValue()` or `.GetValueSync()` to wait for completion of the corresponding task. `GetValue()` and `GetValueSync()` will also rethrow an exception if it has been thrown during the execution of the task.

You may also use `ExecRangeWithThrow()` to just receive an exception from a range if it has been thrown from at least one task. It rethrows an exception from a task with the minimal `id` from all the tasks where exceptions have been thrown or just continues as normal of there were no exceptions.
