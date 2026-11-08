//!
//! A hierarchical thread pool used for splitting work in a branching fashion.
//!
//! This thread pool is good for:
//!
//! - You want to split work recursively in jobs that use approximately the same time.
//! - You want thread pool overhead to be low
//!
//! This is not good for:
//!
//! - You need work stealing
//! - When you have jobs of uneven size
//!

// Stack jobs and job execution implementation based on rayon-core by Niko Matsakis and Josh Stone
//
use crossbeam_channel::{Sender, bounded};

use std::thread;

mod unwind;
mod job;

use crate::job::{JobRef, StackJob};

// ThreadTree message on the channel (is just a job ref)
type TTreeMessage = JobRef;

/// A hierarchical thread pool used for splitting work in a branching fashion.
///
/// See [`ThreadTree::new_with_level()`] to create a new thread tree,
/// and see [`ThreadTree::top()`] for a usage example.
///
/// The thread tree has the benefit that at each level, jobs can be sent directly to the thread
/// that is going to execute it - that means there is no contention between waiting threads. The
/// downside is that the structure of the thread tree is rather static.
#[derive(Debug)]
pub struct ThreadTree {
    sender: Option<Sender<TTreeMessage>>,
    child: Option<[Box<ThreadTree>; 2]>,
}

// Only three threads needed to have four leaves, see below.
//
//           (root)
//      (1)            2 
// (1.1)   1.2   (2.1)   2.2
//
// Leaves 1.1, 1.2, 2.1 and 2.2 but only 1.2, 2, and 2.2 are new threads - the others inherit the
// current thread from the parent.  That means we have a fanout of four (leaves 1.1 trough 2.2)
// using the current thread and three additional threads.
//
// The implementation is such that the root holds ownership of leaf 2, and the root contains a
// channel sender that passes jobs to the node 2.  Further nodes down continue the same way
// recursively.
//
// Idea for later: implement reservations of (parts of) the tree?
// So that a 2-2 tree can be used as two separate 1-2 trees simultaneously

impl ThreadTree {
    const BOTTOM: &'static Self = &ThreadTree::new_level0();

    /// Create a level 0 tree (with no parallelism)
    #[inline]
    pub const fn new_level0() -> Self {
        ThreadTree { sender: None, child: None }
    }

    /// Create an n-level thread tree with 2<sup>n</sup> leaves
    ///
    /// Level 0 has no parallelism
    /// Level 1 has two nodes
    /// Level 2 has four nodes (et.c.)
    ///
    /// Level must be <= 12; panics on invalid input
    pub fn new_with_level(level: usize) -> Box<Self> {
        assert!(level <= 12,
                "Input exceeds maximum level 12 (equivalent to 2**12 - 1 threads), got level='{}'",
                level);
        if level == 0 {
            Box::new(Self::new_level0())
        } else if level == 1 {
            Box::new(ThreadTree { sender: Some(Self::add_thread()), child: None })
        } else {
            let fork_2 = Self::new_with_level(level - 1);
            let fork_3 = Self::new_with_level(level - 1);
            Box::new(ThreadTree { sender: Some(Self::add_thread()), child: Some([fork_2, fork_3])})
        }
    }

    /// Return true if this is a non-dummy pool which will parallelize in join
    #[inline]
    pub fn is_parallel(&self) -> bool {
        self.sender.is_some()
    }

    /// Get the top thread tree context, where we can inject tasks with join.
    /// Each job gets a sub-context that can be used to inject tasks further down the corresponding
    /// branch of the tree.
    ///
    /// **Note** to avoid deadlocks, tasks should never be injected into a tree context that
    /// doesn't belong to the current level. To avoid this should be easy - only call .top() at the
    /// top level.
    ///
    /// The following example shows using a two-level tree and using context to spawn tasks.
    ///
    /// ```
    /// use thread_tree::{ThreadTree, ThreadTreeCtx};
    ///
    /// let tp = ThreadTree::new_with_level(2);
    ///
    /// fn f(index: i32, ctx: ThreadTreeCtx<'_>) -> i32 {
    ///     // do work in subtasks here
    ///     let (a, b) = ctx.join(move |_| index + 1, |_| index + 2);
    ///
    ///     return a + b;
    /// }
    ///
    /// let (r0, r1) = tp.top().join(|ctx| f(0, ctx), |ctx| f(1, ctx));
    ///
    /// assert_eq!(r0 + r1, (0 + 1) + (0 + 2) + (1 + 1) + (1 + 2));
    /// ```
    #[inline]
    pub fn top(&self) -> ThreadTreeCtx<'_> {
        ThreadTreeCtx::from(self)
    }

    // Create a new thread that executes jobs, and return the channel sender that feeds jobs to
    // this thread.
    fn add_thread() -> Sender<TTreeMessage> {
        let (sender, receiver) = bounded::<TTreeMessage>(1); // buffered, we know we have a connection
        std::thread::spawn(move || {
            for job in receiver {
                unsafe {
                    job.execute()
                }
            }
        });
        sender
    }
}

/// A level-specific handle to the thread tree, that can be used to inject jobs.
///
/// See [`ThreadTree::top()`] for more information.
#[derive(Debug, Copy, Clone)]
pub struct ThreadTreeCtx<'a> {
    tree: &'a ThreadTree,
    // This handle is marked as non-Send/Sync as a help - there is nothing safety critical about it
    // - but it helps the user to avoid deadlocks - see the top method.
    _not_send_sync: *const (),
}

impl ThreadTreeCtx<'_> {
    #[inline]
    pub(crate) fn get(&self) -> &ThreadTree { self.tree }

    #[inline]
    pub(crate) fn from(tree: &ThreadTree) -> ThreadTreeCtx<'_> {
        ThreadTreeCtx { tree, _not_send_sync: &() }
    }

    /// Return true if this level will parallelize in join (or if we are at the bottom of the tree)
    #[inline]
    pub fn is_parallel(&self) -> bool {
        self.get().is_parallel()
    }

    /// Branch out and run a and b simultaneously and return their results jointly.
    ///
    /// Job `a` runs on the current thread while `b` runs on the sibling thread; each is passed
    /// a lower level of the thread tree.
    /// If the bottom of the tree is reached, where no sibling threads are available, both `a` and
    /// `b` run on the current thread.
    ///
    /// If either `a` or `b` panics, the panic is propagated here. If both jobs are executing,
    /// the panic will not propagate until after both jobs have finished.
    /// 
    /// Warning: You must not .join() into the same tree from nested jobs. Nested jobs must
    /// be spawned using the context that each job receives as the first parameter.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
        where A: FnOnce(ThreadTreeCtx) -> RA + Send,
              B: FnOnce(ThreadTreeCtx) -> RB + Send,
              RA: Send,
              RB: Send,
    {
        let bottom_level = ThreadTree::BOTTOM;
        let self_ = self.get();
        let (fork_a, fork_b) = match &self_.child {
            None => (bottom_level, bottom_level),
            Some([fa, fb]) => (&**fa, &**fb),
        };
        //assert!(self_.sender.is_some());

        unsafe {
            let a = move || a(ThreadTreeCtx::from(fork_a));
            let b = move || b(ThreadTreeCtx::from(fork_b));

            // first send B to the sibling thread
            let b_job = StackJob::new(b); // plant this safely on the stack
            let b_job_ref = JobRef::new(&b_job);
            let b_runs_here = match self_.sender {
                Some(ref s) => { s.send(b_job_ref).unwrap(); None }
                None => Some(b_job_ref),
            };

            let a_result;
            {
                // Ensure that we will later wait for B, if it is running on
                // another thread. Both in the case of A panic or regular scope exit.
                //
                // If job A panics, we still cannot return until we are sure that job
                // B is complete. This is because it may contain references into the
                // enclosing stack frame(s).
                let _wait_for_b_guard = match b_runs_here {
                    None => Some(WaitForJobGuard::new(&b_job)),
                    Some(_) => None,
                };

                // Execute task A
                a_result = a();

                if let Some(b_job_ref) = b_runs_here {
                    b_job_ref.execute();
                }
                // wait for b here
            }
            (a_result, b_job.into_result())
        }
    }

    /// Branch out twice and join, running three different jobs
    ///
    /// Branches twice on the left side and once on the right.
    /// The closure is called with corresponding thread tree context and an index in 0..3 for the job.
    pub fn join3l<A, RA>(&self, a: &A) -> ((RA, RA), RA)
        where A: Fn(ThreadTreeCtx, usize) -> RA + Sync,
              RA: Send,
    {
        self.join(
            move |ctx| ctx.join(move |ctx| a(ctx, 0), move |ctx| a(ctx, 1)),
            move |ctx| a(ctx, 2))
    }

    /// Branch out twice and join, running three different jobs
    ///
    /// Branches once on the right side and twice on the right.
    /// The closure is called with corresponding thread tree context and an index in 0..3 for the job.
    pub fn join3r<A, RA>(&self, a: &A) -> (RA, (RA, RA))
        where A: Fn(ThreadTreeCtx, usize) -> RA + Sync,
              RA: Send,
    {
        self.join(
            move |ctx| a(ctx, 0),
            move |ctx| ctx.join(move |ctx| a(ctx, 1), move |ctx| a(ctx, 2)))
    }

    /// Branch out twice and join, running four different jobs.
    ///
    /// Branches twice on each side.
    /// The closure is called with corresponding thread tree context and an index in 0..4 for the job.
    pub fn join4<A, RA>(&self, a: &A) -> ((RA, RA), (RA, RA))
        where A: Fn(ThreadTreeCtx, usize) -> RA + Sync,
              RA: Send,
    {
        self.join(
            move |ctx| ctx.join(move |ctx| a(ctx, 0), move |ctx| a(ctx, 1)),
            move |ctx| ctx.join(move |ctx| a(ctx, 2), move |ctx| a(ctx, 3)))
    }
}


fn wait_for_job<F, R>(job: &StackJob<F, R>) {
    while !job.probe() {
        //spin_loop_hint();
        thread::yield_now();
    }
}

struct WaitForJobGuard<'a, F, R> {
    job: &'a StackJob<F, R>,
}

impl<'a, F, R> WaitForJobGuard<'a, F, R>
{
    fn new(job: &'a StackJob<F, R>) -> Self {
        Self { job }
    }
}

impl<'a, F, R> Drop for WaitForJobGuard<'a, F, R> {
    fn drop(&mut self) {
        wait_for_job(self.job)
    }
}

#[cfg(test)]
mod thread_tree_tests {
    use super::*;
    #[allow(deprecated)]

    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;
    use once_cell::sync::Lazy;
    use std::collections::HashSet;
    use std::thread;
    use std::thread::ThreadId;

    #[allow(deprecated)]
    fn sleep_ms(x: u32) {
        std::thread::sleep_ms(x)
    }

    #[test]
    fn stub() {
        let tp = ThreadTree::new_level0();
        let a = AtomicUsize::new(0);
        let b = AtomicUsize::new(0);

        tp.top().join(|_| a.fetch_add(1, Ordering::SeqCst),
                |_| b.fetch_add(1, Ordering::SeqCst));
        assert_eq!(a.load(Ordering::SeqCst), 1);
        assert_eq!(b.load(Ordering::SeqCst), 1);

        let f = || thread::current().id();
        let (aid, bid) = tp.top().join(|_| f(), |_| f());
        assert_eq!(aid, bid);
        assert!(!tp.top().is_parallel());
    }

    #[test]
    fn new_level_1() {
        let tp = ThreadTree::new_with_level(1);
        let a = AtomicUsize::new(0);
        let b = AtomicUsize::new(0);

        tp.top().join(|_| a.fetch_add(1, Ordering::SeqCst),
                |_| b.fetch_add(1, Ordering::SeqCst));
        assert_eq!(a.load(Ordering::SeqCst), 1);
        assert_eq!(b.load(Ordering::SeqCst), 1);

        let f = || thread::current().id();
        let (aid, bid) = tp.top().join(|_| f(), |_| f());
        assert_ne!(aid, bid);
        assert!(tp.top().is_parallel());
    }

    #[test]
    fn build_level_2() {
        let tp = ThreadTree::new_with_level(2);
        let a = AtomicUsize::new(0);
        let b = AtomicUsize::new(0);

        tp.top().join(|_| a.fetch_add(1, Ordering::SeqCst),
                |_| b.fetch_add(1, Ordering::SeqCst));
        assert_eq!(a.load(Ordering::SeqCst), 1);
        assert_eq!(b.load(Ordering::SeqCst), 1);

        let f = || thread::current().id();
        let ((aid, bid), (cid, did)) = tp.top().join(
            |tp1| tp1.join(|_| f(), |_| f()),
            |tp1| tp1.join(|_| f(), |_| f()));
        assert_ne!(aid, bid);
        assert_ne!(aid, cid);
        assert_ne!(aid, did);
        assert_ne!(bid, cid);
        assert_ne!(bid, did);
        assert_ne!(cid, did);
    }

    #[test]
    fn overload_2_2() {
        let global = ThreadTree::new_with_level(1);
        let tp = ThreadTree::new_with_level(2);
        let a = AtomicUsize::new(0);

        let range = 0..100;

        let work = |ctx: ThreadTreeCtx<'_>| {
            let subwork = || {
                for i in range.clone() {
                    a.fetch_add(i, Ordering::Relaxed);
                    sleep_ms(1);
                }
            };
            ctx.join(|_| subwork(), |_| subwork());
        };

        global.top().join(
            |_| tp.top().join(work, work),
            |_| tp.top().join(work, work));

        let sum = range.clone().sum::<usize>();

        assert_eq!(sum * 4 * 2, a.load(Ordering::SeqCst));

    }

    #[test]
    fn deep_tree() {
        static THREADS: Lazy<Mutex<HashSet<ThreadId>>> = Lazy::new(|| Mutex::default());
        const TREE_LEVEL: usize = 8;
        const MAX_DEPTH: usize = 12;

        static COUNT: AtomicUsize = AtomicUsize::new(0);

        let tp = ThreadTree::new_with_level(TREE_LEVEL);

        fn f(tp: ThreadTreeCtx<'_>, depth: usize) {
            COUNT.fetch_add(1, Ordering::SeqCst);
            THREADS.lock().unwrap().insert(thread::current().id());
            if depth >= MAX_DEPTH {
                return;
            }
            tp.join(
                |ctx| {
                    f(ctx, depth + 1);
                },
                |ctx| {
                    f(ctx, depth + 1);
                });
        }

        COUNT.fetch_add(2, Ordering::SeqCst); // for the two invocations below.
        tp.top().join(|ctx| f(ctx, 2), |ctx| f(ctx, 2));
        let visited_threads = THREADS.lock().unwrap().len();
        assert_eq!(visited_threads, 1 << TREE_LEVEL);
        assert_eq!(COUNT.load(Ordering::SeqCst), 1 << MAX_DEPTH);
    }

    #[test]
    #[should_panic]
    fn panic_a() {
        let pool = ThreadTree::new_with_level(1);
        pool.top().join(|_| panic!("Panic in A"), |_| 1 + 1);
    }

    #[test]
    #[should_panic]
    fn panic_b() {
        let pool = ThreadTree::new_with_level(1);
        pool.top().join(|_| 1 + 1, |_| panic!());
    }

    #[test]
    #[should_panic]
    fn panic_both_in_threads() {
        let pool = ThreadTree::new_with_level(1);
        pool.top().join(|_| { sleep_ms(50); panic!("Panic in A") }, |_| panic!("Panic in B"));
    }

    #[test]
    #[should_panic]
    fn panic_both_bottom() {
        let pool = ThreadTree::new_with_level(0);
        pool.top().join(|_| { sleep_ms(50); panic!("Panic in A") }, |_| panic!("Panic in B"));
    }

    #[test]
    fn on_panic_a_wait_for_b() {
        let pool = ThreadTree::new_with_level(1);
        for i in 0..3 {
            let start = AtomicUsize::new(0);
            let finish = AtomicUsize::new(0);
            let result = unwind::halt_unwinding(|| {
                pool.top().join(
                    |_| panic!("Panic in A"),
                    |_| {
                        start.fetch_add(1, Ordering::SeqCst);
                        sleep_ms(50);
                        finish.fetch_add(1, Ordering::SeqCst);
                    });
            });
            let start_count = start.load(Ordering::SeqCst);
            let finish_count = finish.load(Ordering::SeqCst);
            assert_eq!(start_count, finish_count);
            assert!(result.is_err());
            println!("Pass {} with start: {} == finish {}", i,
                     start_count, finish_count);
        }
    }
}
