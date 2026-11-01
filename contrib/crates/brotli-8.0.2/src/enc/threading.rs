use alloc::{Allocator, SliceWrapper, SliceWrapperMut};
use core::marker::PhantomData;
use core::ops::Range;
use core::{any, mem};
#[cfg(feature = "std")]
use std;

use super::backward_references::{AnyHasher, BrotliEncoderParams, CloneWithAlloc, UnionHasher};
use super::encode::{
    hasher_setup, BrotliEncoderDestroyInstance, BrotliEncoderMaxCompressedSize,
    BrotliEncoderOperation, SanitizeParams,
};
use super::BrotliAlloc;
use crate::concat::{BroCatli, BroCatliResult};
use crate::enc::combined_alloc::{alloc_default, allocate};
use crate::enc::encode::BrotliEncoderStateStruct;

pub type PoisonedThreadError = ();

#[cfg(feature = "std")]
pub type LowLevelThreadError = std::boxed::Box<dyn any::Any + Send + 'static>;
#[cfg(not(feature = "std"))]
pub type LowLevelThreadError = ();

pub trait AnyBoxConstructor {
    fn new(data: LowLevelThreadError) -> Self;
}

pub trait Joinable<T: Send + 'static, U: Send + 'static>: Sized {
    fn join(self) -> Result<T, U>;
}
#[derive(Debug)]
pub enum BrotliEncoderThreadError {
    InsufficientOutputSpace,
    ConcatenationDidNotProcessFullFile,
    ConcatenationError(BroCatliResult),
    ConcatenationFinalizationError(BroCatliResult),
    OtherThreadPanic,
    ThreadExecError(LowLevelThreadError),
}

impl AnyBoxConstructor for BrotliEncoderThreadError {
    fn new(data: LowLevelThreadError) -> Self {
        BrotliEncoderThreadError::ThreadExecError(data)
    }
}

pub struct CompressedFileChunk<Alloc: BrotliAlloc + Send + 'static>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
{
    data_backing: <Alloc as Allocator<u8>>::AllocatedMemory,
    data_size: usize,
}
pub struct CompressionThreadResult<Alloc: BrotliAlloc + Send + 'static>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
{
    compressed: Result<CompressedFileChunk<Alloc>, BrotliEncoderThreadError>,
    alloc: Alloc,
}
pub enum InternalSendAlloc<
    ReturnVal: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    Join: Joinable<ReturnVal, BrotliEncoderThreadError>,
> where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
{
    A(Alloc, ExtraInput),
    Join(Join),
    SpawningOrJoining(PhantomData<ReturnVal>),
}
impl<
        ReturnVal: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        Join: Joinable<ReturnVal, BrotliEncoderThreadError>,
    > InternalSendAlloc<ReturnVal, ExtraInput, Alloc, Join>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
{
    fn unwrap_input(&mut self) -> (&mut Alloc, &mut ExtraInput) {
        match *self {
            InternalSendAlloc::A(ref mut alloc, ref mut extra) => (alloc, extra),
            _ => panic!("Bad state for allocator"),
        }
    }
}

pub struct SendAlloc<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    Join: Joinable<ReturnValue, BrotliEncoderThreadError>,
>(pub InternalSendAlloc<ReturnValue, ExtraInput, Alloc, Join>)
//FIXME pub
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send;

impl<
        ReturnValue: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        Join: Joinable<ReturnValue, BrotliEncoderThreadError>,
    > SendAlloc<ReturnValue, ExtraInput, Alloc, Join>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
{
    pub fn new(alloc: Alloc, extra_input: ExtraInput) -> Self {
        SendAlloc::<ReturnValue, ExtraInput, Alloc, Join>(InternalSendAlloc::A(alloc, extra_input))
    }
    pub fn unwrap_or(self, other: Alloc, other_extra: ExtraInput) -> (Alloc, ExtraInput) {
        match self.0 {
            InternalSendAlloc::A(alloc, extra_input) => (alloc, extra_input),
            InternalSendAlloc::SpawningOrJoining(_) | InternalSendAlloc::Join(_) => {
                (other, other_extra)
            }
        }
    }
    fn unwrap_view_mut(&mut self) -> (&mut Alloc, &mut ExtraInput) {
        match self.0 {
            InternalSendAlloc::A(ref mut alloc, ref mut extra_input) => (alloc, extra_input),
            InternalSendAlloc::SpawningOrJoining(_) | InternalSendAlloc::Join(_) => {
                panic!("Item permanently borrowed/leaked")
            }
        }
    }
    pub fn unwrap(self) -> (Alloc, ExtraInput) {
        match self.0 {
            InternalSendAlloc::A(alloc, extra_input) => (alloc, extra_input),
            InternalSendAlloc::SpawningOrJoining(_) | InternalSendAlloc::Join(_) => {
                panic!("Item permanently borrowed/leaked")
            }
        }
    }
    pub fn replace_with_default(&mut self) -> (Alloc, ExtraInput) {
        match mem::replace(
            &mut self.0,
            InternalSendAlloc::SpawningOrJoining(PhantomData),
        ) {
            InternalSendAlloc::A(alloc, extra_input) => (alloc, extra_input),
            InternalSendAlloc::SpawningOrJoining(_) | InternalSendAlloc::Join(_) => {
                panic!("Item permanently borrowed/leaked")
            }
        }
    }
}

pub enum InternalOwned<T> {
    // FIXME pub
    Item(T),
    Borrowed,
}

pub struct Owned<T>(pub InternalOwned<T>); // FIXME pub
impl<T> Owned<T> {
    pub fn new(data: T) -> Self {
        Owned::<T>(InternalOwned::Item(data))
    }
    pub fn unwrap_or(self, other: T) -> T {
        if let InternalOwned::Item(x) = self.0 {
            x
        } else {
            other
        }
    }
    pub fn unwrap(self) -> T {
        if let InternalOwned::Item(x) = self.0 {
            x
        } else {
            panic!("Item permanently borrowed")
        }
    }
    pub fn view(&self) -> &T {
        if let InternalOwned::Item(ref x) = self.0 {
            x
        } else {
            panic!("Item permanently borrowed")
        }
    }
}

pub trait OwnedRetriever<U: Send + 'static> {
    fn view<T, F: FnOnce(&U) -> T>(&self, f: F) -> Result<T, PoisonedThreadError>;
    fn unwrap(self) -> Result<U, PoisonedThreadError>;
}

#[cfg(feature = "std")]
impl<U: Send + 'static> OwnedRetriever<U> for std::sync::Arc<std::sync::RwLock<U>> {
    fn view<T, F: FnOnce(&U) -> T>(&self, f: F) -> Result<T, PoisonedThreadError> {
        match self.read() {
            Ok(ref u) => Ok(f(u)),
            Err(_) => Err(PoisonedThreadError::default()),
        }
    }
    fn unwrap(self) -> Result<U, PoisonedThreadError> {
        match std::sync::Arc::try_unwrap(self) {
            Ok(rwlock) => match rwlock.into_inner() {
                Ok(u) => Ok(u),
                Err(_) => Err(PoisonedThreadError::default()),
            },
            Err(_) => Err(PoisonedThreadError::default()),
        }
    }
}

pub trait BatchSpawnable<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
> where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    type JoinHandle: Joinable<ReturnValue, BrotliEncoderThreadError>;
    type FinalJoinHandle: OwnedRetriever<U>;
    // this function takes in an input slice
    // a SendAlloc per thread and converts them all into JoinHandle
    // the input is borrowed until the joins complete
    // owned is set to borrowed
    // the final join handle is a r/w lock which will return the SliceW to the owner
    // the FinalJoinHandle is only to be called when each individual JoinHandle has been examined
    // the function is called with the thread_index, the num_threads, a reference to the slice under a read lock,
    // and an allocator from the alloc_per_thread
    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle;
    fn spawn<F: Fn(ExtraInput, usize, usize, &U, Alloc) -> ReturnValue + Send + 'static + Copy>(
        &mut self,
        handle: &mut Self::FinalJoinHandle,
        alloc: &mut SendAlloc<ReturnValue, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: F,
    );
}

pub trait BatchSpawnableLite<
    ReturnValue: Send + 'static,
    ExtraInput: Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
> where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    type JoinHandle: Joinable<ReturnValue, BrotliEncoderThreadError>;
    type FinalJoinHandle: OwnedRetriever<U>;
    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle;
    fn spawn(
        &mut self,
        handle: &mut Self::FinalJoinHandle,
        alloc_per_thread: &mut SendAlloc<ReturnValue, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: fn(ExtraInput, usize, usize, &U, Alloc) -> ReturnValue,
    );
}
/*
impl<ReturnValue:Send+'static,
     ExtraInput:Send+'static,
     Alloc:BrotliAlloc+Send+'static,
     U:Send+'static+Sync>
     BatchSpawnableLite<T, Alloc, U> for BatchSpawnable<T, Alloc, U> {
  type JoinHandle = <Self as BatchSpawnable<T, Alloc, U>>::JoinHandle;
  type FinalJoinHandle = <Self as BatchSpawnable<T, Alloc, U>>::FinalJoinHandle;
  fn batch_spawn(
    &mut self,
    input: &mut Owned<U>,
    alloc_per_thread:&mut [SendAlloc<ReturnValue, ExtraInput, Alloc, Self::JoinHandle>],
    f: fn(usize, usize, &U, Alloc) -> T,
  ) -> Self::FinalJoinHandle {
   <Self as BatchSpawnable<ReturnValue, ExtraInput,  Alloc, U>>::batch_spawn(self, input, alloc_per_thread, f)
  }
}*/

pub fn CompressMultiSlice<
    Alloc: BrotliAlloc + Send + 'static,
    Spawner: BatchSpawnableLite<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        (
            <Alloc as Allocator<u8>>::AllocatedMemory,
            BrotliEncoderParams,
        ),
    >,
>(
    params: &BrotliEncoderParams,
    input_slice: &[u8],
    output: &mut [u8],
    alloc_per_thread: &mut [SendAlloc<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        Spawner::JoinHandle,
    >],
    thread_spawner: &mut Spawner,
) -> Result<usize, BrotliEncoderThreadError>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + Sync,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send + Sync,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send + Sync,
{
    let input = if let InternalSendAlloc::A(ref mut alloc, ref _extra) = alloc_per_thread[0].0 {
        let mut input = allocate::<u8, _>(alloc, input_slice.len());
        input.slice_mut().clone_from_slice(input_slice);
        input
    } else {
        alloc_default::<u8, Alloc>()
    };
    let mut owned_input = Owned::new(input);
    let ret = CompressMulti(
        params,
        &mut owned_input,
        output,
        alloc_per_thread,
        thread_spawner,
    );
    if let InternalSendAlloc::A(ref mut alloc, ref _extra) = alloc_per_thread[0].0 {
        <Alloc as Allocator<u8>>::free_cell(alloc, owned_input.unwrap());
    }
    ret
}

fn get_range(thread_index: usize, num_threads: usize, file_size: usize) -> Range<usize> {
    ((thread_index * file_size) / num_threads)..(((thread_index + 1) * file_size) / num_threads)
}

fn compress_part<Alloc: BrotliAlloc + Send + 'static, SliceW: SliceWrapper<u8>>(
    hasher: UnionHasher<Alloc>,
    thread_index: usize,
    num_threads: usize,
    input_and_params: &(SliceW, BrotliEncoderParams),
    mut alloc: Alloc,
) -> CompressionThreadResult<Alloc>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    let mut range = get_range(thread_index, num_threads, input_and_params.0.len());
    let mut mem = allocate::<u8, _>(
        &mut alloc,
        BrotliEncoderMaxCompressedSize(range.end - range.start),
    );
    let mut state = BrotliEncoderStateStruct::new(alloc);
    state.params = input_and_params.1.clone();
    if thread_index != 0 {
        state.params.catable = true; // make sure we can concatenate this to the other work results
        state.params.magic_number = false; // no reason to pepper this around
    }
    state.params.appendable = true; // make sure we are at least appendable, so that future items can be catted in
    if thread_index != 0 {
        state.set_custom_dictionary_with_optional_precomputed_hasher(
            range.start,
            &input_and_params.0.slice()[..range.start],
            hasher,
            true,
        );
    }
    let mut out_offset = 0usize;
    let compression_result;
    let mut available_out = mem.len();
    loop {
        let mut next_in_offset = 0usize;
        let mut available_in = range.end - range.start;
        let result = state.compress_stream(
            BrotliEncoderOperation::BROTLI_OPERATION_FINISH,
            &mut available_in,
            &input_and_params.0.slice()[range.clone()],
            &mut next_in_offset,
            &mut available_out,
            mem.slice_mut(),
            &mut out_offset,
            &mut None,
            &mut |_a, _b, _c, _d| (),
        );
        let new_range = range.start + next_in_offset..range.end;
        range = new_range;
        if result {
            compression_result = Ok(out_offset);
            break;
        } else if available_out == 0 {
            compression_result = Err(BrotliEncoderThreadError::InsufficientOutputSpace); // mark no space??
            break;
        }
    }
    BrotliEncoderDestroyInstance(&mut state);
    match compression_result {
        Ok(size) => CompressionThreadResult::<Alloc> {
            compressed: Ok(CompressedFileChunk {
                data_backing: mem,
                data_size: size,
            }),
            alloc: state.m8,
        },
        Err(e) => {
            <Alloc as Allocator<u8>>::free_cell(&mut state.m8, mem);
            CompressionThreadResult::<Alloc> {
                compressed: Err(e),
                alloc: state.m8,
            }
        }
    }
}

pub fn CompressMulti<
    Alloc: BrotliAlloc + Send + 'static,
    SliceW: SliceWrapper<u8> + Send + 'static + Sync,
    Spawner: BatchSpawnableLite<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        (SliceW, BrotliEncoderParams),
    >,
>(
    params: &BrotliEncoderParams,
    owned_input: &mut Owned<SliceW>,
    output: &mut [u8],
    alloc_per_thread: &mut [SendAlloc<
        CompressionThreadResult<Alloc>,
        UnionHasher<Alloc>,
        Alloc,
        Spawner::JoinHandle,
    >],
    thread_spawner: &mut Spawner,
) -> Result<usize, BrotliEncoderThreadError>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send,
    <Alloc as Allocator<u16>>::AllocatedMemory: Send,
    <Alloc as Allocator<u32>>::AllocatedMemory: Send,
{
    let num_threads = alloc_per_thread.len();
    let actually_owned_mem = mem::replace(owned_input, Owned(InternalOwned::Borrowed));
    let mut owned_input_pair = Owned::new((actually_owned_mem.unwrap(), params.clone()));
    // start thread spawner
    let mut spawner_and_input = thread_spawner.make_spawner(&mut owned_input_pair);
    if num_threads > 1 {
        // spawn first thread without "custom dictionary" while we compute the custom dictionary for other work items
        thread_spawner.spawn(
            &mut spawner_and_input,
            &mut alloc_per_thread[0],
            0,
            num_threads,
            compress_part,
        );
    }
    // populate all hashers at once, cloning them one by one
    let mut compression_last_thread_result;
    if num_threads > 1 && params.favor_cpu_efficiency {
        let mut local_params = params.clone();
        SanitizeParams(&mut local_params);
        let mut hasher = UnionHasher::Uninit;
        hasher_setup(
            alloc_per_thread[num_threads - 1].0.unwrap_input().0,
            &mut hasher,
            &mut local_params,
            None, // No unwrappable custom dict used here.
            &[],
            0,
            0,
            false,
        );
        for thread_index in 1..num_threads {
            let res = spawner_and_input.view(|input_and_params: &(SliceW, BrotliEncoderParams)| {
                let range = get_range(thread_index - 1, num_threads, input_and_params.0.len());
                let overlap = hasher.StoreLookahead().wrapping_sub(1);
                if range.end - range.start > overlap {
                    hasher.BulkStoreRange(
                        input_and_params.0.slice(),
                        usize::MAX,
                        if range.start > overlap {
                            range.start - overlap
                        } else {
                            0
                        },
                        range.end - overlap,
                    );
                }
            });
            if let Err(_e) = res {
                return Err(BrotliEncoderThreadError::OtherThreadPanic);
            }
            if thread_index + 1 != num_threads {
                {
                    let (alloc, out_hasher) = alloc_per_thread[thread_index].unwrap_view_mut();
                    *out_hasher = hasher.clone_with_alloc(alloc);
                }
                thread_spawner.spawn(
                    &mut spawner_and_input,
                    &mut alloc_per_thread[thread_index],
                    thread_index,
                    num_threads,
                    compress_part,
                );
            }
        }
        let (alloc, _extra) = alloc_per_thread[num_threads - 1].replace_with_default();
        compression_last_thread_result = spawner_and_input.view(move |input_and_params:&(SliceW, BrotliEncoderParams)| -> CompressionThreadResult<Alloc> {
        compress_part(hasher,
                      num_threads - 1,
                      num_threads,
                      input_and_params,
                      alloc,
        )
      });
    } else {
        if num_threads > 1 {
            for thread_index in 1..num_threads - 1 {
                thread_spawner.spawn(
                    &mut spawner_and_input,
                    &mut alloc_per_thread[thread_index],
                    thread_index,
                    num_threads,
                    compress_part,
                );
            }
        }
        let (alloc, _extra) = alloc_per_thread[num_threads - 1].replace_with_default();
        compression_last_thread_result = spawner_and_input.view(move |input_and_params:&(SliceW, BrotliEncoderParams)| -> CompressionThreadResult<Alloc> {
        compress_part(UnionHasher::Uninit,
                      num_threads - 1,
                      num_threads,
                      input_and_params,
                      alloc,
        )
      });
    }
    let mut compression_result = Err(BrotliEncoderThreadError::InsufficientOutputSpace);
    let mut out_file_size = 0usize;
    let mut bro_cat_li = BroCatli::new();
    for (index, thread) in alloc_per_thread.iter_mut().enumerate() {
        let mut cur_result = if index + 1 == num_threads {
            match mem::replace(&mut compression_last_thread_result, Err(())) {
                Ok(result) => result,
                Err(_err) => return Err(BrotliEncoderThreadError::OtherThreadPanic),
            }
        } else {
            match mem::replace(
                &mut thread.0,
                InternalSendAlloc::SpawningOrJoining(PhantomData),
            ) {
                InternalSendAlloc::A(_, _) | InternalSendAlloc::SpawningOrJoining(_) => {
                    panic!("Thread not properly spawned")
                }
                InternalSendAlloc::Join(join) => match join.join() {
                    Ok(result) => result,
                    Err(err) => {
                        return Err(err);
                    }
                },
            }
        };
        match cur_result.compressed {
            Ok(compressed_out) => {
                bro_cat_li.new_brotli_file();
                let mut in_offset = 0usize;
                let cat_result = bro_cat_li.stream(
                    &compressed_out.data_backing.slice()[..compressed_out.data_size],
                    &mut in_offset,
                    output,
                    &mut out_file_size,
                );
                match cat_result {
                    BroCatliResult::Success | BroCatliResult::NeedsMoreInput => {
                        compression_result = Ok(out_file_size);
                    }
                    BroCatliResult::NeedsMoreOutput => {
                        compression_result = Err(BrotliEncoderThreadError::InsufficientOutputSpace);
                        // not enough space
                    }
                    err => {
                        compression_result = Err(BrotliEncoderThreadError::ConcatenationError(err));
                        // misc error
                    }
                }
                <Alloc as Allocator<u8>>::free_cell(
                    &mut cur_result.alloc,
                    compressed_out.data_backing,
                );
            }
            Err(e) => {
                compression_result = Err(e);
            }
        }
        thread.0 = InternalSendAlloc::A(cur_result.alloc, UnionHasher::Uninit);
    }
    compression_result?;
    match bro_cat_li.finish(output, &mut out_file_size) {
        BroCatliResult::Success => compression_result = Ok(out_file_size),
        err => {
            compression_result = Err(BrotliEncoderThreadError::ConcatenationFinalizationError(
                err,
            ))
        }
    }
    if let Ok(retrieved_owned_input) = spawner_and_input.unwrap() {
        *owned_input = Owned::new(retrieved_owned_input.0); // return the input to its rightful owner before returning
    } else if compression_result.is_ok() {
        compression_result = Err(BrotliEncoderThreadError::OtherThreadPanic);
    }
    compression_result
}
