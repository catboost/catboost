use std::{
    cell::Cell,
    io,
    marker::PhantomData,
    os::windows::io::{AsRawHandle, RawHandle},
    process::{Child, ChildStderr},
    thread::sleep,
    time::Duration,
};

use ::find_msvc_tools::windows_sys::{
    WaitForMultipleObjects, INFINITE, MAXIMUM_WAIT_OBJECTS, WAIT_FAILED, WAIT_OBJECT_0,
};

use crate::{cell_modify, cell_push};

#[derive(Default)]
pub(crate) struct Reactor {
    /// Handles to wait on: process handles, signaled once the process exits,
    /// after the semaphore if any, so that it is always waited on.
    handles: Cell<Vec<RawHandle>>,
    /// Called with a count taken from the semaphore at `handles[0]`, if any.
    on_semaphore: Cell<Option<Box<dyn FnOnce()>>>,
}

/// Anonymous pipes can't be waited on, so stderr is only read on wake-ups
/// for other reasons, including its process exiting, and timeouts.
pub(crate) struct Registration<'a>(PhantomData<&'a Reactor>);

/// Keeps a child registered with the [`Reactor`] until dropped, which must
/// happen before the [`Child`] is dropped, as it uses its handle.
pub(crate) struct ChildRegistration<'a> {
    reactor: &'a Reactor,
    handle: RawHandle,
}

/// Keeps a semaphore registered with the [`Reactor`] until dropped, which
/// must happen before the semaphore is closed.
pub(crate) struct SemaphoreRegistration<'a> {
    reactor: &'a Reactor,
}

impl Reactor {
    pub(crate) fn register_stderr(&self, _stderr: &ChildStderr) -> io::Result<Registration<'_>> {
        Ok(Registration(PhantomData))
    }

    /// Wake [`Reactor::wait_with_timeout`] when `child` exits.
    pub(crate) fn register_child(&self, child: &Child) -> io::Result<ChildRegistration<'_>> {
        let handle = child.as_raw_handle();
        cell_push(&self.handles, handle);
        Ok(ChildRegistration {
            reactor: self,
            handle,
        })
    }

    /// Wake [`Reactor::wait_with_timeout`] when `semaphore` has a count to
    /// take. The wake-up takes it, and hands it to `on_acquired` right away,
    /// after which the semaphore is no longer waited on.
    ///
    /// At most one semaphore can be registered at a time.
    pub(crate) fn register_semaphore(
        &self,
        semaphore: RawHandle,
        on_acquired: Box<dyn FnOnce()>,
    ) -> SemaphoreRegistration<'_> {
        let previous = self.on_semaphore.replace(Some(on_acquired));
        assert!(previous.is_none(), "a semaphore is already registered");
        cell_modify(&self.handles, |handles| {
            handles.push(semaphore);
            let last = handles.len() - 1;
            handles.swap(0, last);
        });
        SemaphoreRegistration { reactor: self }
    }

    pub(crate) fn wait_with_timeout(&self, timeout: Duration) -> io::Result<()> {
        // Never `INFINITE`, which would wait forever.
        let timeout = u32::try_from(timeout.as_millis())
            .unwrap_or(u32::MAX)
            .min(INFINITE - 1);
        cell_modify(&self.handles, |handles| {
            if handles.is_empty() {
                // Waiting on no handles is an error.
                sleep(Duration::from_millis(timeout.into()));
                return Ok(());
            }
            // Any beyond these are only noticed through other wake-ups, as
            // their order changes, or the timeout.
            let count = handles.len().min(MAXIMUM_WAIT_OBJECTS);
            let ret = unsafe { WaitForMultipleObjects(count as u32, handles.as_ptr(), 0, timeout) };
            if ret == WAIT_FAILED {
                return Err(io::Error::last_os_error());
            }
            // Only the handle returned for has been waited on successfully,
            // and the semaphore is at index 0.
            if ret == WAIT_OBJECT_0 {
                if let Some(on_acquired) = self.on_semaphore.take() {
                    // That has taken a count from the semaphore, so stop
                    // waiting on it, which would take more.
                    handles.swap_remove(0);
                    on_acquired();
                }
            }
            Ok(())
        })
    }
}

impl Drop for ChildRegistration<'_> {
    fn drop(&mut self) {
        let handle = self.handle;
        cell_modify(&self.reactor.handles, |handles| {
            // Never index 0 while the semaphore is there, so it stays there.
            if let Some(i) = handles.iter().position(|h| *h == handle) {
                handles.swap_remove(i);
            }
        });
    }
}

impl Drop for SemaphoreRegistration<'_> {
    fn drop(&mut self) {
        if self.reactor.on_semaphore.take().is_some() {
            cell_modify(&self.reactor.handles, |handles| handles.swap_remove(0));
        }
    }
}
