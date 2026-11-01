use std::any::Any;
use std::cell::UnsafeCell;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use super::unwind;

// from rayon
pub enum JobResult<T> {
    None,
    Ok(T),
    Panic(Box<dyn Any + Send>),
}

// from rayon
pub struct StackJob<F, R> {
    func: UnsafeCell<Option<F>>,
    result: UnsafeCell<JobResult<R>>,
    latch: AtomicBool,
}

impl<F, R> StackJob<F, R> {
    pub fn new(f: F) -> Self
        where F: FnOnce() -> R + Send
    {
        Self {
            func: UnsafeCell::new(Some(f)),
            result: UnsafeCell::new(JobResult::None),
            latch: AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn into_result(self) -> R {
        unsafe {
            debug_assert!((*self.func.get()).is_none());
            match mem::replace(&mut *self.result.get(), JobResult::None) {
                JobResult::None => unreachable!(),
                JobResult::Ok(r) => r,
                JobResult::Panic(x) => unwind::resume_unwinding(x),
            }
        }
    }
    pub fn probe(&self) -> bool {
        self.latch.load(Ordering::Acquire)
    }
}

impl<F, R> Job for StackJob<F, R>
where
    //L: Latch + Sync,
    F: FnOnce() -> R + Send,
    R: Send,
{
    #[inline]
    unsafe fn execute(this: *const Self) {
        let this = &*this;
        let abort = unwind::AbortIfPanic;
        let func = (*this.func.get()).take().unwrap();
        (*this.result.get()) = match unwind::halt_unwinding(|| func()) {
            Ok(x) => JobResult::Ok(x),
            Err(x) => JobResult::Panic(x),
        };
        this.latch.store(true, Ordering::Release);
        //this.latch.set();
        mem::forget(abort);
    }
}


/// A `Job` is used to advertise work for other threads that they may
/// want to steal. In accordance with time honored tradition, jobs are
/// arranged in a deque, so that thieves can take from the top of the
/// deque while the main worker manages the bottom of the deque. This
/// deque is managed by the `thread_pool` module.
pub trait Job {
    /// Unsafe: this may be called from a different thread than the one
    /// which scheduled the job, so the implementer must ensure the
    /// appropriate traits are met, whether `Send`, `Sync`, or both.
    unsafe fn execute(this: *const Self);
}

/// Effectively a Job trait object. Each JobRef **must** be executed
/// exactly once, or else data may leak.
///
/// Internally, we store the job's data in a `*const ()` pointer.  The
/// true type is something like `*const StackJob<...>`, but we hide
/// it. We also carry the "execute fn" from the `Job` trait.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JobRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const ()),
}

unsafe impl Send for JobRef {}
unsafe impl Sync for JobRef {}

impl JobRef {
    /// Unsafe: caller asserts that `data` will remain valid until the
    /// job is executed.
    pub unsafe fn new<T>(data: *const T) -> JobRef
    where
        T: Job + Send,
    {
        let fn_ptr: unsafe fn(*const T) = <T as Job>::execute;

        // erase types:
        let fn_ptr: unsafe fn(*const ()) = mem::transmute(fn_ptr);
        let pointer = data as *const ();

        JobRef {
            pointer: pointer,
            execute_fn: fn_ptr,
        }
    }

    #[inline]
    pub unsafe fn execute(&self) {
        (self.execute_fn)(self.pointer)
    }
}

