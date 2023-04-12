#pragma once

#include "rw_spin_lock.h"

#include <functional>

namespace NYT::NThreading {

////////////////////////////////////////////////////////////////////////////////

using TAtForkHandler = std::function<void()>;

//! Registers handlers to be invoked at fork time.
//! See |pthread_atfork| for more details.
/*!
 *  Once all prepare handlers are invoked, fork lock is acquired
 *  in writer mode. This lock is subsequently released in both child
 *  and parent processes once fork is complete.
 */
void RegisterAtForkHandlers(
    TAtForkHandler prepare,
    TAtForkHandler parent,
    TAtForkHandler child);

//! Returns the fork lock.
TReaderWriterSpinLock* GetForkLock();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NThreading
