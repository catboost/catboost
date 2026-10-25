use std::{
    cell::Cell,
    io,
    marker::PhantomData,
    os::unix::io::{AsRawFd, RawFd},
    process::{Child, ChildStderr},
    time::Duration,
};

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd"
))]
use std::os::unix::io::{FromRawFd, OwnedFd};

#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "netbsd",
    target_os = "openbsd"
))]
use std::{
    mem::{self, MaybeUninit},
    ptr,
};

use crate::{cell_modify, cell_push};

#[derive(Default)]
pub(crate) struct Reactor {
    poll_fds: Cell<Vec<libc::pollfd>>,
    /// Set once `pidfd_open` turns out not to be available, such as on Linux
    /// before 5.3 or when blocked by seccomp, so it isn't tried again.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    no_pidfd: Cell<bool>,
    /// Created by the first [`Reactor::register_child`], to be told about
    /// child exits. From then on it is always polled, as it is only readable
    /// while it has events, which every wait consumes.
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    kqueue: Cell<Option<OwnedFd>>,
}

pub(crate) struct Registration<'a> {
    reactor: &'a Reactor,
    fd: RawFd,
}

/// Keeps a child registered with the [`Reactor`] until dropped.
pub(crate) struct ChildRegistration<'a> {
    /// Its pidfd, if it could be opened, and the reactor polling it.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pidfd: Option<(&'a Reactor, OwnedFd)>,
    _reactor: PhantomData<&'a Reactor>,
}

impl Reactor {
    pub(crate) fn register_stderr(&self, stderr: &ChildStderr) -> io::Result<Registration<'_>> {
        Ok(self.register_fd(stderr.as_raw_fd()))
    }

    pub(crate) fn register_fd(&self, fd: RawFd) -> Registration<'_> {
        self.add_fd(fd);
        Registration { reactor: self, fd }
    }

    fn add_fd(&self, fd: RawFd) {
        cell_push(
            &self.poll_fds,
            libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            },
        );
    }

    /// Must be called before `fd` is closed, while its number can't have
    /// been reused.
    fn remove_fd(&self, fd: RawFd) {
        cell_modify(&self.poll_fds, |poll_fds| {
            if let Some(i) = poll_fds.iter().position(|pollfd| pollfd.fd == fd) {
                poll_fds.swap_remove(i);
            }
        });
    }

    /// Wake [`Reactor::wait_with_timeout`] when `child` exits, where that can
    /// be waited for. Otherwise its exit is only noticed through its stderr
    /// being closed, which a process it started may keep open, or the
    /// timeout.
    pub(crate) fn register_child(&self, _child: &Child) -> io::Result<ChildRegistration<'_>> {
        // Nothing to undo when it is dropped: the event is deleted once
        // returned, or with the kqueue.
        #[cfg(any(
            target_vendor = "apple",
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        self.register_proc_kevent(_child)?;

        Ok(ChildRegistration {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            pidfd: self.register_pidfd(_child)?,
            _reactor: PhantomData,
        })
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn register_pidfd(&self, child: &Child) -> io::Result<Option<(&Self, OwnedFd)>> {
        if !self.no_pidfd.get() {
            // Not yet waited on, so the pid is still the child's.
            let ret = unsafe {
                libc::syscall(
                    libc::SYS_pidfd_open,
                    libc::c_long::from(child.id() as libc::pid_t),
                    0 as libc::c_long,
                )
            };
            if ret >= 0 {
                // Always close-on-exec.
                let pidfd = unsafe { OwnedFd::from_raw_fd(ret as RawFd) };
                self.add_fd(pidfd.as_raw_fd());
                return Ok(Some((self, pidfd)));
            }
            let err = io::Error::last_os_error();
            if matches!(
                err.raw_os_error(),
                Some(libc::ENOSYS | libc::EPERM | libc::EACCES)
            ) {
                self.no_pidfd.set(true);
                return Ok(None);
            }

            return Err(err);
        }

        Ok(None)
    }

    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    fn register_proc_kevent(&self, child: &Child) -> io::Result<()> {
        let kqueue = cell_modify(&self.kqueue, |kqueue| {
            if let Some(kqueue) = kqueue {
                return Ok(kqueue.as_raw_fd());
            }
            let fd = unsafe { libc::kqueue() };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            *kqueue = Some(unsafe { OwnedFd::from_raw_fd(fd) });
            self.add_fd(fd);
            Ok(fd)
        })?;

        let change = libc::kevent {
            ident: child.id() as _,
            filter: libc::EVFILT_PROC as _,
            // Deleted once returned by `kevent`, see `drain_kqueue`.
            flags: (libc::EV_ADD | libc::EV_ONESHOT) as _,
            fflags: libc::NOTE_EXIT as _,
            // `data` and `udata` are unused, but their types differ between
            // these, and FreeBSD 12+ adds `ext`, depending on how libc is
            // built. All are plain integers or pointers, zero is valid.
            ..unsafe { mem::zeroed() }
        };

        // Only applies `change`: with no room for events, it doesn't wait for
        // any, and a failure to apply it is returned as an error.
        let ret = unsafe { libc::kevent(kqueue, &change, 1, ptr::null_mut(), 0, ptr::null()) };
        if ret < 0 {
            let err = io::Error::last_os_error();
            // It has already exited, so there is nothing to wait for.
            if err.raw_os_error() == Some(libc::ESRCH) {
                return Ok(());
            }
            return Err(err);
        }

        Ok(())
    }

    /// Consumes the child exits so far, which are what the futures are about
    /// to be polled for, so that the kqueue isn't readable for them anymore.
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    fn drain_kqueue(&self) -> io::Result<()> {
        cell_modify(&self.kqueue, |kqueue| {
            let Some(kqueue) = kqueue else {
                return Ok(());
            };
            const N: usize = 64;
            // Only written to by `kevent`, never read.
            let mut events = MaybeUninit::<[libc::kevent; N]>::uninit();
            let timeout = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            loop {
                // No changes, up to `N` events, without waiting.
                let n = unsafe {
                    libc::kevent(
                        kqueue.as_raw_fd(),
                        ptr::null(),
                        0,
                        events.as_mut_ptr().cast(),
                        N as _,
                        &timeout,
                    )
                };
                if n < 0 {
                    return Err(io::Error::last_os_error());
                }
                // Fewer than asked for: there are no more.
                if n < N as _ {
                    return Ok(());
                }
            }
        })
    }

    pub(crate) fn wait_with_timeout(&self, timeout: Duration) -> io::Result<()> {
        let timeout = timeout.as_millis().try_into().unwrap();
        let result = cell_modify(&self.poll_fds, |poll_fds| {
            let ret = unsafe {
                libc::poll(
                    poll_fds.as_mut_ptr(),
                    poll_fds.len() as libc::nfds_t,
                    timeout,
                )
            };
            if ret < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
        #[cfg(any(
            target_vendor = "apple",
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        let result = result.and_then(|()| self.drain_kqueue());
        match result {
            Err(err) if err.kind() == io::ErrorKind::Interrupted => Ok(()),
            result => result,
        }
    }
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        self.reactor.remove_fd(self.fd);
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
impl Drop for ChildRegistration<'_> {
    fn drop(&mut self) {
        // The pidfd itself is closed right after, as a field.
        if let Some((reactor, pidfd)) = &self.pidfd {
            reactor.remove_fd(pidfd.as_raw_fd());
        }
    }
}
