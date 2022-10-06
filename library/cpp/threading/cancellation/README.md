The Cancellation library
========================

Intro
-----

This small library provides primitives for implementation of a cooperative cancellation of long running or asynchronous operations.
The design has been copied from the well-known CancellationTokenSource/CancellationToken classes of the .NET Framework

To use the library include `cancellation_token.h`.

Examples
--------

1. Simple check for cancellation

    ```c++
    void LongRunningOperation(TCancellationToken token) {
        ...
        if (token.IsCancellationRequested()) {
            return;
        }
        ...
    }

    TCancellationTokenSource source;
    TThread thread([token = source.Token()]() { LongRunningOperation(std::move(token)); });
    thread.Start();
    ...
    source.Cancel();
    thread.Join();
    ```

2. Exit via an exception

    ```c++
    void LongRunningOperation(TCancellationToken token) {
        try {
            for (;;) {
                ...
                token.ThrowIfCancellationRequested();
                ...
            }
        } catch (TOperationCancelledException const&) {
            return;
        } catch (...) {
            Y_FAIL("Never should be there")
        }
    }

    TCancellationTokenSource source;
    TThread thread([token = source.Token()]() { LongRunningOperation(std::move(token)); });
    thread.Start();
    ...
    source.Cancel();
    thread.Join();
    ```

3. Periodic poll with cancellation

    ```c++
    void LongRunningOperation(TCancellationToken token) {
        while (!token.Wait(PollInterval)) {
            ...
        }
    }

    TCancellationTokenSource source;
    TThread thread([token = source.Token()]() { LongRunningOperation(std::move(token)); });
    thread.Start();
    ...
    source.Cancel();
    thread.Join();
    ```

4. Waiting on the future

    ```c++
    TFuture<void> InnerOperation();
    TFuture<void> OuterOperation(TCancellationToken token) {
        return WaitAny(FirstOperation(), token.Future())
                    .Apply([token = std::move(token)](auto&&) {
                        token.ThrowIfCancellationRequested();
                    });
    }

    TCancellationTokenSource source;
    auto future = OuterOperation();
    ...
    source.Cancel()
    ...
    try {
        auto value = future.ExtractValueSync();
    } catch (TOperationCancelledException const&) {
        // cancelled
    }
    ```

5. Using default token when no cancellation needed

    ```c++
    void LongRunningOperation(TCancellationToken token) {
        ...
        if (token.IsCancellationRequested()) {
            return;
        }
        ...
    }

    // We do not want to cancel the operation. So, there is no need to create a cancellation token source
    LongRunningOperation(TCancellationToken::Default);
    ```
