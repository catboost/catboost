use alloc::{Allocator, SliceWrapper};
use core::marker::PhantomData;
use core::mem;
#[cfg(feature = "std")]
use std;

use super::backward_references::UnionHasher;
use crate::enc::threading::{
    BatchSpawnable, BatchSpawnableLite, BrotliEncoderThreadError, CompressMulti,
    CompressionThreadResult, InternalOwned, InternalSendAlloc, Joinable, Owned, OwnedRetriever,
    PoisonedThreadError, SendAlloc,
};
use crate::enc::{BrotliAlloc, BrotliEncoderParams};

pub struct SingleThreadedJoinable<T: Send + 'static, U: Send + 'static> {
    result: Result<T, U>,
}
impl<T: Send + 'static, U: Send + 'static> Joinable<T, U> for SingleThreadedJoinable<T, U> {
    fn join(self) -> Result<T, U> {
        self.result
    }
}
#[cfg(feature = "std")]
pub struct SingleThreadedOwnedRetriever<U: Send + 'static>(std::sync::RwLock<U>);
#[cfg(feature = "std")]
impl<U: Send + 'static> OwnedRetriever<U> for SingleThreadedOwnedRetriever<U> {
    fn view<T, F: FnOnce(&U) -> T>(&self, f: F) -> Result<T, PoisonedThreadError> {
        Ok(f(&*self.0.read().unwrap()))
    }
    fn unwrap(self) -> Result<U, PoisonedThreadError> {
        Ok(self.0.into_inner().unwrap())
    }
}
#[cfg(feature = "std")]
impl<U: Send + 'static> SingleThreadedOwnedRetriever<U> {
    fn new(u: U) -> Self {
        SingleThreadedOwnedRetriever(std::sync::RwLock::new(u))
    }
}

#[cfg(not(feature = "std"))]
pub struct SingleThreadedOwnedRetriever<U: Send + 'static>(U);
#[cfg(not(feature = "std"))]
impl<U: Send + 'static> SingleThreadedOwnedRetriever<U> {
    fn new(u: U) -> Self {
        SingleThreadedOwnedRetriever(u)
    }
}
#[cfg(not(feature = "std"))]
impl<U: Send + 'static> OwnedRetriever<U> for SingleThreadedOwnedRetriever<U> {
    fn view<T, F: FnOnce(&U) -> T>(&self, f: F) -> Result<T, PoisonedThreadError> {
        Ok(f(&self.0))
    }
    fn unwrap(self) -> Result<U, PoisonedThreadError> {
        Ok(self.0)
    }
}

#[derive(Default)]
pub struct SingleThreadedSpawner {}

impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > BatchSpawnable<ReturnValue, ExtraInput, Alloc, U> for SingleThreadedSpawner
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    type JoinHandle = SingleThreadedJoinable<ReturnValue, BrotliEncoderThreadError>;
    type FinalJoinHandle = SingleThreadedOwnedRetriever<U>;
    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle {
        SingleThreadedOwnedRetriever::<U>::new(
            mem::replace(input, Owned(InternalOwned::Borrowed)).unwrap(),
        )
    }
    fn spawn<F: Fn(ExtraInput, usize, usize, &U, Alloc) -> ReturnValue + Send + 'static + Copy>(
        &mut self,
        handle: &mut Self::FinalJoinHandle,
        work: &mut SendAlloc<ReturnValue, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: F,
    ) {
        let (alloc, extra_input) = work.replace_with_default();
        let ret = handle.view(|sub_view| f(extra_input, index, num_threads, sub_view, alloc));
        *work = SendAlloc(InternalSendAlloc::Join(SingleThreadedJoinable {
            result: Ok(ret.unwrap()),
        }));
    }
}

impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > BatchSpawnableLite<ReturnValue, ExtraInput, Alloc, U> for SingleThreadedSpawner
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    type JoinHandle =
        <SingleThreadedSpawner as BatchSpawnable<ReturnValue, ExtraInput, Alloc, U>>::JoinHandle;
    type FinalJoinHandle = <SingleThreadedSpawner as BatchSpawnable<
        ReturnValue,
        ExtraInput,
        Alloc,
        U,
    >>::FinalJoinHandle;

    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle {
        <Self as BatchSpawnable<ReturnValue, ExtraInput, Alloc, U>>::make_spawner(self, input)
    }
    fn spawn(
        &mut self,
        handle: &mut Self::FinalJoinHandle,
        alloc_per_thread: &mut SendAlloc<ReturnValue, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: fn(ExtraInput, usize, usize, &U, Alloc) -> ReturnValue,
    ) {
        <Self as BatchSpawnable<ReturnValue, ExtraInput, Alloc, U>>::spawn(
            self,
            handle,
            alloc_per_thread,
            index,
            num_threads,
            f,
        )
    }
}

pub fn compress_multi<
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
        <SingleThreadedSpawner as BatchSpawnable<
            CompressionThreadResult<Alloc>,
            UnionHasher<Alloc>,
            Alloc,
            SliceW,
        >>::JoinHandle,
    >],
) -> Result<usize, BrotliEncoderThreadError>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send,
{
    CompressMulti(
        params,
        owned_input,
        output,
        alloc_per_thread,
        &mut SingleThreadedSpawner::default(),
    )
}

pub struct WorkerPool<A, B, C, D> {
    a: PhantomData<A>,
    b: PhantomData<B>,
    c: PhantomData<C>,
    d: PhantomData<D>,
}
pub fn new_work_pool<A, B, C, D>(_num_threads: usize) -> WorkerPool<A, B, C, D> {
    WorkerPool::<A, B, C, D> {
        a: PhantomData,
        b: PhantomData,
        c: PhantomData,
        d: PhantomData,
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
        <SingleThreadedSpawner as BatchSpawnable<
            CompressionThreadResult<Alloc>,
            UnionHasher<Alloc>,
            Alloc,
            SliceW,
        >>::JoinHandle,
    >],
    _worker_pool: &mut WorkerPool<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        (SliceW, BrotliEncoderParams),
    >,
) -> Result<usize, BrotliEncoderThreadError>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send,
{
    compress_multi(params, owned_input, output, alloc_per_thread)
}
