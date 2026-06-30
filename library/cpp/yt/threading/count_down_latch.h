#pragma once

#include "public.h"

#ifndef _linux_
    #include <util/system/condvar.h>
    #include <util/system/mutex.h>
#endif

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

//! A synchronization aid that allows one or more threads to wait until
//! a set of operations being performed in other threads completes.
/*!
 *  See https://docs.oracle.com/javase/7/docs/api/java/util/concurrent/CountDownLatch.html
 */
class TCountDownLatch final
{
public:
    explicit TCountDownLatch(int count);

    void CountDown();

    void Wait() const;
    bool TryWait() const;

    int GetCount() const;

private:
    std::atomic<int> Count_;

#ifndef _linux_
    mutable TCondVar ConditionVariable_;
    mutable TMutex Mutex_;
#endif
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
