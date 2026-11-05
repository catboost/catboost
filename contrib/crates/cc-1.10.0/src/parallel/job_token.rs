use core::marker::PhantomData;

use crate::{parallel::reactor::Reactor, utilities::OnceLock, Error};

pub(crate) struct JobToken(PhantomData<()>);

impl JobToken {
    fn new() -> Self {
        Self(PhantomData)
    }
}

impl Drop for JobToken {
    fn drop(&mut self) {
        match JobTokenServer::new() {
            JobTokenServer::Inherited(jobserver) => jobserver.release_token_raw(),
            JobTokenServer::InProcess(jobserver) => jobserver.release_token_raw(),
        }
    }
}

enum JobTokenServer {
    Inherited(inherited_jobserver::JobServer),
    InProcess(inprocess_jobserver::JobServer),
}

impl JobTokenServer {
    /// This function returns a static reference to the jobserver because
    ///  - creating a jobserver from env is a bit fd-unsafe (e.g. the fd might
    ///    be closed by other jobserver users in the process) and better do it
    ///    at the start of the program.
    ///  - in case a jobserver cannot be created from env (e.g. it's not
    ///    present), we will create a global in-process only jobserver
    ///    that has to be static so that it will be shared by all cc
    ///    compilation.
    fn new() -> &'static Self {
        // TODO: Replace with a OnceLock once MSRV is 1.70
        static JOBSERVER: OnceLock<JobTokenServer> = OnceLock::new();

        JOBSERVER.get_or_init(|| {
            unsafe { inherited_jobserver::JobServer::from_env() }
                .map(Self::Inherited)
                .unwrap_or_else(|| Self::InProcess(inprocess_jobserver::JobServer::new()))
        })
    }
}

pub(crate) enum ActiveJobTokenServer {
    Inherited(inherited_jobserver::ActiveJobServer<'static>),
    InProcess(&'static inprocess_jobserver::JobServer),
}

impl ActiveJobTokenServer {
    pub(crate) fn new() -> Self {
        match JobTokenServer::new() {
            JobTokenServer::Inherited(inherited_jobserver) => {
                Self::Inherited(inherited_jobserver.enter_active())
            }
            JobTokenServer::InProcess(inprocess_jobserver) => Self::InProcess(inprocess_jobserver),
        }
    }

    pub(crate) async fn acquire(&mut self, reactor: &Reactor) -> Result<JobToken, Error> {
        match self {
            Self::Inherited(jobserver) => jobserver.acquire(reactor).await,
            Self::InProcess(jobserver) => Ok(jobserver.acquire().await),
        }
    }
}

mod inherited_jobserver {
    use super::JobToken;

    use crate::{
        parallel::{async_executor::YieldOnce, reactor::Reactor},
        Error, ErrorKind,
    };

    #[cfg(windows)]
    use crate::{cell_modify, cell_push};

    #[cfg(unix)]
    use std::{
        fs::File,
        io::{Read, Write},
        os::unix::io::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd},
    };

    #[cfg(windows)]
    use std::{
        cell::Cell,
        os::windows::io::{AsRawHandle, OwnedHandle},
        rc::Rc,
    };

    use std::{
        io, mem,
        sync::{Arc, Mutex, MutexGuard, PoisonError},
    };

    pub(super) struct JobServer {
        /// Implicit token for this process which is obtained and will be
        /// released in parent. Since `JobTokens` only give back what they got,
        /// there should be at most one global implicit token in the wild.
        ///
        /// Since Rust does not execute any `Drop` for global variables,
        /// we can't just put it back to jobserver and then re-acquire it at
        /// the end of the process.
        ///
        /// Use `Mutex` to avoid race between acquire and release.
        /// If an `AtomicBool` is used, then it's possible for:
        ///  - `release_token_raw`: Tries to set `global_implicit_token` to true, but it is already
        ///    set  to `true`, continue to release it to jobserver
        ///  - `acquire` takes the global implicit token, set `global_implicit_token` to false
        ///  - `release_token_raw` now writes the token back into the jobserver, while
        ///    `global_implicit_token` is `false`
        ///
        /// If the program exits here, then cc effectively increases parallelism by one, which is
        /// incorrect, hence we use a `Mutex` here.
        global_implicit_token: Mutex<bool>,
        inner: jobserver::Client,
        #[cfg(unix)]
        fd: Option<OwnedFd>,
        #[cfg(windows)]
        semaphore: Option<OwnedHandle>,
    }

    impl JobServer {
        pub(super) unsafe fn from_env() -> Option<Self> {
            jobserver::Client::from_env().map(|inner| Self {
                #[cfg(unix)]
                fd: Self::jobserver_fd(&inner),
                #[cfg(windows)]
                semaphore: Self::jobserver_semaphore(&inner),
                inner,
                global_implicit_token: Mutex::new(true),
            })
        }

        /// `jobserver::Client` doesn't expose its pipe, fifo or semaphore, so
        /// take it from the `MAKEFLAGS` it passes to child processes, which is
        /// what they use.
        #[cfg(any(unix, windows))]
        fn jobserver_auth(client: &jobserver::Client) -> Option<String> {
            let mut cmd = std::process::Command::new("");
            client.configure(&mut cmd);
            let (_, flags) = cmd.get_envs().find(|(key, _)| *key == "CARGO_MAKEFLAGS")?;
            // Last in the flags, so a fifo path may contain spaces.
            let (_, auth) = flags?.to_str()?.split_once("--jobserver-auth=")?;
            Some(auth.to_owned())
        }

        #[cfg(unix)]
        fn jobserver_fd(client: &jobserver::Client) -> Option<OwnedFd> {
            use std::fs::OpenOptions;

            let auth = Self::jobserver_auth(client)?;

            if let Some(path) = auth.strip_prefix("fifo:") {
                // Read and write, as opening a fifo read-only blocks until it has
                // a writer.
                let fifo = OpenOptions::new().read(true).write(true).open(path);
                Some(fifo.ok()?.into())
            } else {
                let (read, _) = auth.split_once(',')?;
                let read = read.parse::<RawFd>().ok().filter(|fd| *fd >= 0)?;
                // `from_env` has just opened the client from this fd, and left it
                // open.
                unsafe { BorrowedFd::borrow_raw(read) }
                    .try_clone_to_owned()
                    .ok()
            }
        }

        #[cfg(windows)]
        fn jobserver_semaphore(client: &jobserver::Client) -> Option<OwnedHandle> {
            use ::find_msvc_tools::windows_sys::{OpenSemaphoreA, SYNCHRONIZE};
            use std::{ffi::CString, os::windows::io::FromRawHandle};

            let name = CString::new(Self::jobserver_auth(client)?).ok()?;
            // Only to wait on, which takes a token, that is then released
            // through `inner`.
            let handle = unsafe { OpenSemaphoreA(SYNCHRONIZE, 0, name.as_ptr().cast()) };
            if handle.is_null() {
                None
            } else {
                Some(unsafe { OwnedHandle::from_raw_handle(handle) })
            }
        }

        fn get_global_implicit_token(&self) -> MutexGuard<'_, bool> {
            self.global_implicit_token
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
        }

        /// All tokens except for the global implicit token will be put back into the jobserver
        /// immediately and they cannot be cached, since Rust does not call `Drop::drop` on
        /// global variables.
        pub(super) fn release_token_raw(&self) {
            let mut global_implicit_token = self.get_global_implicit_token();

            if *global_implicit_token {
                // There's already a global implicit token, so this token must
                // be released back into jobserver.
                //
                // `release_raw` should not block
                let _ = self.inner.release_raw();
            } else {
                *global_implicit_token = true;
            }
        }

        pub(super) fn enter_active(&self) -> ActiveJobServer<'_> {
            ActiveJobServer {
                jobserver: self,
                helper_thread: None,
                #[cfg(windows)]
                semaphore_tokens: Rc::default(),
            }
        }
    }

    struct HelperThread {
        inner: jobserver::HelperThread,
        /// What the helper thread has sent, which only holds it weakly.
        /// Dropping it puts the tokens left in it back.
        tokens: Arc<Mutex<Vec<io::Result<jobserver::Acquired>>>>,
        /// Whether a token was requested and is yet to be received.
        ///
        /// Kept across `acquire`s: one that ends up with another token leaves
        /// its request pending, and requesting again would make the helper
        /// thread take more tokens than are ever used.
        has_requested_token: bool,
        /// Readable once the helper thread has sent a token, to be polled.
        #[cfg(unix)]
        notified: File,
    }

    impl HelperThread {
        fn new(jobserver: &JobServer) -> Result<Self, Error> {
            let tokens = Arc::default();
            let weak = Arc::downgrade(&tokens);
            #[cfg(unix)]
            let (notified, notifier) = Self::pipe()?;

            Ok(Self {
                tokens,
                inner: jobserver.inner.clone().into_helper_thread(move |res| {
                    // Only gone if dropping `Self` gave up on joining the
                    // helper thread, and dropping `res` puts the token back.
                    let Some(tokens) = weak.upgrade() else {
                        return;
                    };
                    tokens
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push(res);
                    // Only to wake up the reactor: if it fails, the token is
                    // still found on its timeout. If the pipe is full, it is
                    // readable already. If `notified` is gone, it fails with
                    // `EPIPE`, as Rust programs ignore `SIGPIPE`.
                    #[cfg(unix)]
                    let _ = (&notifier).write(&[0]);
                })?,
                has_requested_token: false,
                #[cfg(unix)]
                notified,
            })
        }

        /// A non-blocking pipe, as (read, write).
        #[cfg(unix)]
        fn pipe() -> io::Result<(File, File)> {
            let mut fds = [0; 2];
            if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let pipe = unsafe { (File::from_raw_fd(fds[0]), File::from_raw_fd(fds[1])) };
            for fd in fds {
                // `pipe2` isn't on all of them.
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                if flags == -1
                    || unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } == -1
                    || unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) } == -1
                {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(pipe)
        }

        fn try_recv(&mut self) -> Result<Option<JobToken>, Error> {
            // Before receiving, so that a token sent after it notifies again.
            #[cfg(unix)]
            loop {
                match (&self.notified).read(&mut [0; 64]) {
                    // The helper thread is gone, see below.
                    Ok(0) => break,
                    Ok(_) => (),
                    Err(err) if err.kind() == io::ErrorKind::WouldBlock => break,
                    Err(err) if err.kind() == io::ErrorKind::Interrupted => (),
                    // Left readable, it would wake up every wait.
                    Err(err) => return Err(err.into()),
                }
            }

            let mut tokens = self.tokens.lock().unwrap_or_else(PoisonError::into_inner);
            match tokens.pop() {
                Some(res) => {
                    drop(tokens);
                    self.has_requested_token = false;
                    res?.drop_without_releasing();
                    Ok(Some(JobToken::new()))
                }
                // The helper thread's callback holds the only `Weak`, and
                // pushes before dropping it. Read under the lock, so that a
                // token pushed before is popped above.
                None if Arc::weak_count(&self.tokens) == 0 => Err(Error::new(
                    ErrorKind::JobserverHelpThreadError,
                    "jobserver help thread has returned before ActiveJobServer is dropped",
                )),
                None => Ok(None),
            }
        }

        /// At most one at a time, see `has_requested_token`.
        fn request_token(&mut self) {
            if !self.has_requested_token {
                self.inner.request_token();
                self.has_requested_token = true;
            }
        }
    }

    pub(crate) struct ActiveJobServer<'a> {
        jobserver: &'a JobServer,
        helper_thread: Option<HelperThread>,
        /// Tokens of counts taken from the semaphore by waiting on it.
        #[cfg(windows)]
        semaphore_tokens: Rc<Cell<Vec<JobToken>>>,
    }

    impl ActiveJobServer<'_> {
        /// Take a token without blocking.
        fn try_acquire(&self) -> io::Result<Option<JobToken>> {
            // Already taken by waiting on the semaphore.
            #[cfg(windows)]
            if let Some(token) = cell_modify(&self.semaphore_tokens, Vec::pop) {
                return Ok(Some(token));
            }

            Ok(self.jobserver.inner.try_acquire()?.map(|acquired| {
                acquired.drop_without_releasing();
                JobToken::new()
            }))
        }

        pub(super) async fn acquire(&mut self, _reactor: &Reactor) -> Result<JobToken, Error> {
            // Kept while waiting, dropped once a token is acquired, or the
            // future is: a jobserver with tokens to spare would wake every
            // wait at once, and waiting on the semaphore takes tokens.
            #[cfg(unix)]
            let mut _registration = None;
            #[cfg(windows)]
            let mut _semaphore_registration = None;

            loop {
                // Fast path
                if mem::replace(&mut *self.jobserver.get_global_implicit_token(), false) {
                    break Ok(JobToken::new());
                }

                match self.try_acquire() {
                    Ok(Some(token)) => break Ok(token),
                    Ok(None) => {
                        // Other processes, such as rustc, other build
                        // scripts or make, put the tokens they release back
                        // into the jobserver.
                        #[cfg(unix)]
                        if _registration.is_none() {
                            _registration = self
                                .jobserver
                                .fd
                                .as_ref()
                                .map(|fd| _reactor.register_fd(fd.as_raw_fd()));
                        }
                        #[cfg(windows)]
                        if _semaphore_registration.is_none() {
                            if let Some(semaphore) = &self.jobserver.semaphore {
                                let tokens = Rc::clone(&self.semaphore_tokens);
                                _semaphore_registration = Some(_reactor.register_semaphore(
                                    semaphore.as_raw_handle(),
                                    Box::new(move || cell_push(&tokens, JobToken::new())),
                                ));
                            }
                        }
                        YieldOnce::default().await
                    }
                    Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                        // Fallback to creating a help thread with blocking acquire
                        let helper_thread = if let Some(thread) = self.helper_thread.as_mut() {
                            thread
                        } else {
                            self.helper_thread
                                .insert(HelperThread::new(self.jobserver)?)
                        };

                        if let Some(token) = helper_thread.try_recv()? {
                            break Ok(token);
                        }

                        helper_thread.request_token();
                        // The helper thread notifies once it has sent a token.
                        #[cfg(unix)]
                        if _registration.is_none() {
                            _registration =
                                Some(_reactor.register_fd(helper_thread.notified.as_raw_fd()));
                        }
                        YieldOnce::default().await
                    }
                    Err(err) => break Err(err.into()),
                }
            }
        }
    }
}

mod inprocess_jobserver {
    use super::JobToken;

    use crate::{parallel::async_executor::YieldOnce, utilities::cargo_env_var_os};

    use core::sync::atomic::{
        AtomicU32,
        Ordering::{AcqRel, Acquire},
    };

    pub(crate) struct JobServer(AtomicU32);

    impl JobServer {
        pub(super) fn new() -> Self {
            // Use `NUM_JOBS` if set (it's configured by Cargo) and otherwise
            // just fall back to the number of cores on the local machine, or a reasonable
            // default if that cannot be determined.

            let parallelism = cargo_env_var_os("NUM_JOBS")
                .and_then(|j| j.to_str()?.parse::<u32>().ok())
                .or_else(|| Some(std::thread::available_parallelism().ok()?.get() as u32))
                .unwrap_or(4);

            Self(AtomicU32::new(parallelism))
        }

        pub(super) async fn acquire(&self) -> JobToken {
            loop {
                // TODO: use try_fetch once msrv bump to 1.95
                #[allow(deprecated)]
                let res = self
                    .0
                    .fetch_update(AcqRel, Acquire, |tokens| tokens.checked_sub(1));

                if res.is_ok() {
                    break JobToken::new();
                }

                YieldOnce::default().await
            }
        }

        pub(super) fn release_token_raw(&self) {
            self.0.fetch_add(1, AcqRel);
        }
    }
}
