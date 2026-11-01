#![cfg(feature = "std")]

use alloc::{Allocator, SliceWrapper};
use core::mem;
use std;
// in-place thread create
use std::sync::RwLock;
use std::sync::{Arc, Condvar, Mutex};

use crate::enc::backward_references::UnionHasher;
use crate::enc::fixed_queue::{FixedQueue, MAX_THREADS};
use crate::enc::threading::{
    BatchSpawnableLite, BrotliEncoderThreadError, CompressMulti, CompressionThreadResult,
    InternalOwned, InternalSendAlloc, Joinable, Owned, SendAlloc,
};
use crate::enc::{BrotliAlloc, BrotliEncoderParams};

struct JobReply<T: Send + 'static> {
    result: T,
    work_id: u64,
}

struct JobRequest<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
> {
    func: fn(ExtraInput, usize, usize, &U, Alloc) -> ReturnValue,
    extra_input: ExtraInput,
    index: usize,
    thread_size: usize,
    data: Arc<RwLock<U>>,
    alloc: Alloc,
    work_id: u64,
}

struct WorkQueue<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
> {
    jobs: FixedQueue<JobRequest<ReturnValue, ExtraInput, Alloc, U>>,
    results: FixedQueue<JobReply<ReturnValue>>,
    shutdown: bool,
    immediate_shutdown: bool,
    num_in_progress: usize,
    cur_work_id: u64,
}
impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > Default for WorkQueue<ReturnValue, ExtraInput, Alloc, U>
{
    fn default() -> Self {
        WorkQueue {
            jobs: FixedQueue::default(),
            results: FixedQueue::default(),
            num_in_progress: 0,
            immediate_shutdown: false,
            shutdown: false,
            cur_work_id: 0,
        }
    }
}

pub struct GuardedQueue<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
>(Arc<(Mutex<WorkQueue<ReturnValue, ExtraInput, Alloc, U>>, Condvar)>);
pub struct WorkerPool<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
> {
    queue: GuardedQueue<ReturnValue, ExtraInput, Alloc, U>,
    join: [Option<std::thread::JoinHandle<()>>; MAX_THREADS],
}

impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > Drop for WorkerPool<ReturnValue, ExtraInput, Alloc, U>
{
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.queue.0;
            let mut local_queue = lock.lock().unwrap();
            local_queue.immediate_shutdown = true;
            cvar.notify_all();
        }
        for thread_handle in self.join.iter_mut() {
            if let Some(th) = thread_handle.take() {
                th.join().unwrap();
            }
        }
    }
}
impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > WorkerPool<ReturnValue, ExtraInput, Alloc, U>
{
    fn do_work(queue: Arc<(Mutex<WorkQueue<ReturnValue, ExtraInput, Alloc, U>>, Condvar)>) {
        loop {
            let ret;
            {
                // need to drop possible job before the final lock is taken,
                // so refcount of possible_job Arc is 0 by the time the job is delivered
                // to the caller. We basically need a barrier (the lock) to happen
                // after the destructor that decrefs possible_job
                let possible_job;
                {
                    let (lock, cvar) = &*queue;
                    let mut local_queue = lock.lock().unwrap();
                    if local_queue.immediate_shutdown {
                        break;
                    }
                    possible_job = if let Some(res) = local_queue.jobs.pop() {
                        cvar.notify_all();
                        local_queue.num_in_progress += 1;
                        res
                    } else if local_queue.shutdown {
                        break;
                    } else {
                        let _lock = cvar.wait(local_queue); // unlock immediately, unfortunately
                        continue;
                    };
                }
                ret = if let Ok(job_data) = possible_job.data.read() {
                    JobReply {
                        result: (possible_job.func)(
                            possible_job.extra_input,
                            possible_job.index,
                            possible_job.thread_size,
                            &*job_data,
                            possible_job.alloc,
                        ),
                        work_id: possible_job.work_id,
                    }
                } else {
                    break; // poisoned lock
                };
            }
            {
                let (lock, cvar) = &*queue;
                let mut local_queue = lock.lock().unwrap();
                local_queue.num_in_progress -= 1;
                local_queue.results.push(ret).unwrap();
                cvar.notify_all();
            }
        }
    }
    fn _push_job(&mut self, job: JobRequest<ReturnValue, ExtraInput, Alloc, U>) {
        let (lock, cvar) = &*self.queue.0;
        let mut local_queue = lock.lock().unwrap();
        loop {
            if local_queue.jobs.size() + local_queue.num_in_progress + local_queue.results.size()
                < MAX_THREADS
            {
                local_queue.jobs.push(job).unwrap();
                cvar.notify_all();
                break;
            }
            local_queue = cvar.wait(local_queue).unwrap();
        }
    }
    fn _try_push_job(
        &mut self,
        job: JobRequest<ReturnValue, ExtraInput, Alloc, U>,
    ) -> Result<(), JobRequest<ReturnValue, ExtraInput, Alloc, U>> {
        let (lock, cvar) = &*self.queue.0;
        let mut local_queue = lock.lock().unwrap();
        if local_queue.jobs.size() + local_queue.num_in_progress + local_queue.results.size()
            < MAX_THREADS
        {
            local_queue.jobs.push(job).unwrap();
            cvar.notify_all();
            Ok(())
        } else {
            Err(job)
        }
    }
    fn start(
        queue: Arc<(Mutex<WorkQueue<ReturnValue, ExtraInput, Alloc, U>>, Condvar)>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || Self::do_work(queue))
    }
    pub fn new(num_threads: usize) -> Self {
        let queue = Arc::new((Mutex::new(WorkQueue::default()), Condvar::new()));
        WorkerPool {
            queue: GuardedQueue(queue.clone()),
            join: [
                Some(Self::start(queue.clone())),
                if 1 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 2 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 3 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 4 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 5 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 6 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 7 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 8 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 9 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 10 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 11 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 12 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 13 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 14 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
                if 15 < num_threads {
                    Some(Self::start(queue.clone()))
                } else {
                    None
                },
            ],
        }
    }
}

pub fn new_work_pool<
    Alloc: BrotliAlloc + Send + 'static,
    SliceW: SliceWrapper<u8> + Send + 'static + Sync,
>(
    num_threads: usize,
) -> WorkerPool<
    CompressionThreadResult<Alloc>,
    UnionHasher<Alloc>,
    Alloc,
    (SliceW, BrotliEncoderParams),
>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send + Sync,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send + Sync,
{
    WorkerPool::new(num_threads)
}

pub struct WorkerJoinable<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
> {
    queue: GuardedQueue<ReturnValue, ExtraInput, Alloc, U>,
    work_id: u64,
}
impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > Joinable<ReturnValue, BrotliEncoderThreadError>
    for WorkerJoinable<ReturnValue, ExtraInput, Alloc, U>
{
    fn join(self) -> Result<ReturnValue, BrotliEncoderThreadError> {
        let (lock, cvar) = &*self.queue.0;
        let mut local_queue = lock.lock().unwrap();
        loop {
            match local_queue
                .results
                .remove(|data: &Option<JobReply<ReturnValue>>| {
                    if let Some(ref item) = *data {
                        item.work_id == self.work_id
                    } else {
                        false
                    }
                }) {
                Some(matched) => return Ok(matched.result),
                None => local_queue = cvar.wait(local_queue).unwrap(),
            };
        }
    }
}

impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > BatchSpawnableLite<ReturnValue, ExtraInput, Alloc, U>
    for WorkerPool<ReturnValue, ExtraInput, Alloc, U>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send + Sync,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send + Sync,
{
    type FinalJoinHandle = Arc<RwLock<U>>;
    type JoinHandle = WorkerJoinable<ReturnValue, ExtraInput, Alloc, U>;

    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle {
        std::sync::Arc::<RwLock<U>>::new(RwLock::new(
            mem::replace(input, Owned(InternalOwned::Borrowed)).unwrap(),
        ))
    }
    fn spawn(
        &mut self,
        locked_input: &mut Self::FinalJoinHandle,
        work: &mut SendAlloc<ReturnValue, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: fn(ExtraInput, usize, usize, &U, Alloc) -> ReturnValue,
    ) {
        assert!(num_threads <= MAX_THREADS);
        let (lock, cvar) = &*self.queue.0;
        let mut local_queue = lock.lock().unwrap();
        loop {
            if local_queue.jobs.size() + local_queue.num_in_progress + local_queue.results.size()
                <= MAX_THREADS
            {
                let work_id = local_queue.cur_work_id;
                local_queue.cur_work_id += 1;
                let (local_alloc, local_extra) = work.replace_with_default();
                local_queue
                    .jobs
                    .push(JobRequest {
                        func: f,
                        extra_input: local_extra,
                        index,
                        thread_size: num_threads,
                        data: locked_input.clone(),
                        alloc: local_alloc,
                        work_id,
                    })
                    .unwrap();
                *work = SendAlloc(InternalSendAlloc::Join(WorkerJoinable {
                    queue: GuardedQueue(self.queue.0.clone()),
                    work_id,
                }));
                cvar.notify_all();
                break;
            } else {
                local_queue = cvar.wait(local_queue).unwrap(); // hope room frees up
            }
        }
    }
}

pub fn compress_worker_pool<
    Alloc: BrotliAlloc + Send + 'static,
    SliceW: SliceWrapper<u8> + Send + 'static + Sync,
>(
    params: &BrotliEncoderParams,
    owned_input: &mut Owned<SliceW>,
    output: &mut [u8],
    alloc_per_thread: &mut [SendAlloc<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        <WorkerPool<
            CompressionThreadResult<Alloc>,
            UnionHasher<Alloc>,
            Alloc,
            (SliceW, BrotliEncoderParams),
        > as BatchSpawnableLite<
            CompressionThreadResult<Alloc>,
            UnionHasher<Alloc>,
            Alloc,
            (SliceW, BrotliEncoderParams),
        >>::JoinHandle,
    >],
    work_pool: &mut WorkerPool<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        (SliceW, BrotliEncoderParams),
    >,
) -> Result<usize, BrotliEncoderThreadError>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send + Sync,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send + Sync,
{
    CompressMulti(params, owned_input, output, alloc_per_thread, work_pool)
}

// out of place thread create
