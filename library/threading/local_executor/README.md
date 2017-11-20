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

`void TLocalExecutor::ExecRange(TLocallyExecutableFunction exec, TBlockParams blockParams, int flags);` - run range of tasks `[TBlockParams::FirstId, TBlockParams::LastId).`

`flags` is the same as for `TLocalExecutor::Exec`.

`TBlockParams` is a structure that describes the range.
By default each task is executed separately. Threads from thread pool are taking
the tasks in the manner first come first serve.

It is also possible to partition range of tasks in consequtive blocks and execute each block as a bigger task.
`TBlockParams::SetBlockCountToThreadCount()` will result in thread count tasks,
    where thread count is the count of threads in thread pool.
    each thread will execute approximately equal count of tasks from range.

`TBlockParams::SetBlockSize()` and `TBlockParams::SetBlockCount()` will partition
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
}, TBlockParams(0, 10), TLocalExecutor::WAIT_COMPLETE | TLocalExecutor::MED_PRIORITY);
```
