Subscriptions manager and wait primitives library
=================================================

Wait primitives
---------------

All wait primitives are futures those being signaled when some or all of theirs dependencies are signaled.
Wait privimitives could be constructed either from an initializer_list or from a standard container of futures.

1. WaitAll is signaled when all its dependencies are signaled:

    ```C++
    #include <library/cpp/threading/subscriptions/wait_all.h>

    auto w = NWait::WaitAll({ future1, future2, ..., futureN });
    ...
    w.Wait(); // wait for all futures
    ```

2. WaitAny is signaled when any of its dependencies is signaled:

    ```C++
    #include <library/cpp/threading/subscriptions/wait_any.h>

    auto w = NWait::WaitAny(TVector<TFuture<T>>{ future1, future2, ..., futureN });
    ...
    w.Wait(); // wait for any future
    ```

3. WaitAllOrException is signaled when all its dependencies are signaled with values or any dependency is signaled with an exception:

    ```C++
    #include <library/cpp/threading/subscriptions/wait_all_or_exception.h>

    auto w = NWait::WaitAllOrException(TVector<TFuture<T>>{ future1, future2, ..., futureN });
    ...
    w.Wait(); // wait for all values or for an exception
    ```

Subscriptions manager
---------------------

The subscription manager can manage multiple links beetween futures and callbacks. Multiple managed subscriptions to a single future shares just a single underlying subscription to the future. That allows dynamic creation and deletion of subscriptions and efficient implementation of different wait primitives.
The subscription manager could be used in the following way:

1. Subscribe to a single future:

    ```C++
    #include <library/cpp/threading/subscriptions/subscription.h>

    TFuture<int> LongOperation();

    ...
    auto future = LongRunnigOperation();
    auto m = MakeSubsriptionManager<int>();
    auto id = m->Subscribe(future, [](TFuture<int> const& f) {
        try {
            auto value = f.GetValue();
            ...
        } catch (...) {
            ... // handle exception
        }
    });
    if (id.has_value()) {
        ... // Callback will run asynchronously
    } else {
        ... // Future has been signaled already. The callback has been invoked synchronously
    }
    ```

    Note that a callback could be invoked synchronously during a Subscribe call. In this case the returned optional will have no value.

2. Unsubscribe from a single future:

    ```C++
    // id holds the subscription id from a previous Subscribe call
    m->Unsubscribe(id.value());
    ```

    There is no need to call Unsubscribe if the callback has been called. In this case Unsubscribe will do nothing. And it is safe to call Unsubscribe with the same id multiple times.

3. Subscribe a single callback to multiple futures:

    ```C++
    auto ids = m->Subscribe({ future1, future2, ..., futureN }, [](auto&& f) { ... });
    ...
    ```

    Futures could be passed to Subscribe method either via an initializer_list or via a standard container like vector or list. Subscribe method accept an optional boolean parameter revertOnSignaled. If the parameter is false (default) then all suscriptions will be performed regardless of the futures states and the returned vector will have a subscription id for each future (even if callback has been executed synchronously for some futures). Otherwise the method will stop on the first signaled future (the callback will be synchronously called for it), no suscriptions will be created and an empty vector will be returned.

4. Unsubscribe multiple subscriptions:

    ```C++
    // ids is the vector or subscription ids
    m->Unsubscribe(ids);
    ```

    The vector of IDs could be a result of a previous Subscribe call or an arbitrary set of IDs of previously created subscriptions.

5. If you do not want to instantiate a new instance of the subscription manager it is possible to use the default instance:

    ```C++
    auto m = TSubscriptionManager<T>::Default();
    ```
