# Library for parallel task execution in thread pool

This library allows easy parallelization of existing code and cycles.
It provides `NPar::TLocalExecutor` class and `NPar::LocalExecutor()` singleton accessor.
At start, `TLocalExecutor` has no threads in thread pool and will try to execute all tasks synchronously.
All tasks should be `NPar::ILocallyExecutable` child class or function equal to `std::function<void(int)>`

## TLocalExecutor methods

`TLocalExecutor::Run(int threadcount)` - add threads to thread pool (**WARNING!** `Run(threadcount)` will *add* `threadcount` threads to pool)

`void TLocalExecutor::Exec(TLocallyExecutableFunction exec, int id, int flags)` - run one task and pass id as task function input, flags - bitmask composition of:

- `TLocalExecutor::HIGH_PRIORITY = 0` - put task in high priority queue
- `TLocalExecutor::MED_PRIORITY = 1` - put task in medium priority queue
- `TLocalExecutor::LOW_PRIORITY = 2` - put task in low priority queue
- `TLocalExecutor::WAIT_COMPLETE = 4` - wait for task completion

`void TLocalExecutor::ExecRange(TLocallyExecutableFunction exec, int firstId, int lastId, int flags);` - run tasks on range `[firstId; int lastId)`, `flags` - the same as for `TLocalExecutor::Exec`

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
LocalExecutor().Exec([](int id) {
    SomeFunc(id);
}, 0, 10, TLocalExecutor::WAIT_COMPLETE | TLocalExecutor::MED_PRIORITY);
```