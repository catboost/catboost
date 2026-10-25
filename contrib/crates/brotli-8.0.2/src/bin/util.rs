use core::marker::PhantomData;
use core::mem;
use std;
use std::collections::BTreeMap;
use std::fmt;
use std::thread::JoinHandle;

use alloc_no_stdlib::{Allocator, SliceWrapper};
use brotli::dictionary::{
    kBrotliDictionary, kBrotliDictionaryOffsetsByLength, kBrotliDictionarySizeBitsByLength,
};
use brotli::enc::threading::{
    AnyBoxConstructor, BatchSpawnable, BatchSpawnableLite, BrotliEncoderThreadError, InternalOwned,
    InternalSendAlloc, Joinable, Owned, OwnedRetriever, PoisonedThreadError, SendAlloc,
};
use brotli::enc::BrotliAlloc;
use brotli::interface;
use brotli::transform::TransformDictionaryWord;

struct HexSlice<'a>(&'a [u8]);

impl<'a> fmt::Display for HexSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for byte in self.0 {
            if let Err(e) = write!(f, "{:02X}", byte) {
                return Err(e);
            }
        }
        Ok(())
    }
}
pub fn permute_dictionary() -> BTreeMap<Vec<u8>, ()> {
    let mut ret = BTreeMap::<Vec<u8>, ()>::new();
    let mut transformed = [0u8; 38];
    for wordlen in 4..kBrotliDictionaryOffsetsByLength.len() {
        let offset = kBrotliDictionaryOffsetsByLength[wordlen] as usize;
        for index in 0..(1 << kBrotliDictionarySizeBitsByLength[wordlen]) {
            let word = &kBrotliDictionary[offset + index..offset + index + wordlen];
            for transform in 0..121 {
                let final_size =
                    TransformDictionaryWord(&mut transformed[..], word, wordlen as i32, transform)
                        as usize;
                let vec: Vec<u8> = transformed[..final_size].to_vec();
                ret.insert(vec, ());
            }
        }
    }
    ret
}

pub fn print_dictionary(dict: BTreeMap<Vec<u8>, ()>) {
    for (key, _) in dict {
        println!("{}", HexSlice(&key[..]));
    }
}
macro_rules! println_stderr(
    ($($val:tt)*) => { {
        writeln!(&mut ::std::io::stderr(), $($val)*).unwrap();
    } }
);

fn prediction_mode_str(
    prediction_mode_nibble: interface::LiteralPredictionModeNibble,
) -> &'static str {
    match prediction_mode_nibble.prediction_mode() {
        interface::LITERAL_PREDICTION_MODE_SIGN => "sign",
        interface::LITERAL_PREDICTION_MODE_LSB6 => "lsb6",
        interface::LITERAL_PREDICTION_MODE_MSB6 => "msb6",
        interface::LITERAL_PREDICTION_MODE_UTF8 => "utf8",
        _ => "unknown",
    }
}

struct SliceU8Ref<'a>(pub &'a [u8]);

impl<'a> fmt::LowerHex for SliceU8Ref<'a> {
    fn fmt(&self, fmtr: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for item in self.0 {
            fmtr.write_fmt(format_args!("{:02x}", item))?
        }
        Ok(())
    }
}

pub fn write_one<T: SliceWrapper<u8>>(cmd: &interface::Command<T>) {
    use std::io::Write;
    match cmd {
        interface::Command::BlockSwitchLiteral(bsl) => {
            println_stderr!("ltype {} {}", bsl.0.block_type(), bsl.1);
        }
        interface::Command::BlockSwitchCommand(bsc) => {
            println_stderr!("ctype {}", bsc.0);
        }
        interface::Command::BlockSwitchDistance(bsd) => {
            println_stderr!("dtype {}", bsd.0);
        }
        interface::Command::PredictionMode(prediction) => {
            let prediction_mode = prediction_mode_str(prediction.literal_prediction_mode());
            let lit_cm = prediction
                .literal_context_map
                .slice()
                .iter()
                .fold(::std::string::String::new(), |res, &val| {
                    res + " " + &val.to_string()
                });
            let dist_cm = prediction
                .distance_context_map()
                .iter()
                .fold(::std::string::String::new(), |res, &val| {
                    res + " " + &val.to_string()
                });
            let mixing_values = prediction
                .get_mixing_values()
                .iter()
                .fold(::std::string::String::new(), |res, &val| {
                    res + " " + &val.to_string()
                });
            if prediction.has_context_speeds() {
                println_stderr!("prediction {} lcontextmap{} dcontextmap{} mixingvalues{} cmspeedinc {} {} cmspeedmax {} {} stspeedinc {} {} stspeedmax {} {} mxspeedinc {} {} mxspeedmax {} {}",
                                prediction_mode,
                                lit_cm,
                                dist_cm,
                                mixing_values,
                                prediction.context_map_speed()[0].0,
                                prediction.context_map_speed()[1].0,
                                prediction.context_map_speed()[0].1,
                                prediction.context_map_speed()[1].1,
                                prediction.stride_context_speed()[0].0,
                                prediction.stride_context_speed()[1].0,
                                prediction.stride_context_speed()[0].1,
                                prediction.stride_context_speed()[1].1,
                                prediction.combined_stride_context_speed()[0].0,
                                prediction.combined_stride_context_speed()[1].0,
                                prediction.combined_stride_context_speed()[0].1,
                                prediction.combined_stride_context_speed()[0].1,
                                );
            } else {
                println_stderr!(
                    "prediction {} lcontextmap{} dcontextmap{} mixingvalues{}",
                    prediction_mode,
                    lit_cm,
                    dist_cm,
                    mixing_values,
                );
            }
        }
        interface::Command::Copy(copy) => {
            println_stderr!("copy {} from {}", copy.num_bytes, copy.distance);
        }
        interface::Command::Dict(dict) => {
            let mut transformed_word = [0u8; 38];
            let word_index = dict.word_id as usize * dict.word_size as usize
                + kBrotliDictionaryOffsetsByLength[dict.word_size as usize] as usize;
            let raw_word = &kBrotliDictionary[word_index..(word_index + dict.word_size as usize)];
            let actual_copy_len = TransformDictionaryWord(
                &mut transformed_word[..],
                raw_word,
                dict.word_size as i32,
                dict.transform as i32,
            ) as usize;

            assert_eq!(dict.final_size as usize, actual_copy_len);
            println_stderr!(
                "dict {} word {},{} {:x} func {} {:x}",
                actual_copy_len,
                dict.word_size,
                dict.word_id,
                SliceU8Ref(raw_word),
                dict.transform,
                SliceU8Ref(transformed_word.split_at(actual_copy_len).0)
            );
        }
        interface::Command::Literal(lit) => {
            println_stderr!(
                "{} {} {:x}",
                if lit.high_entropy { "rndins" } else { "insert" },
                lit.data.slice().len(),
                SliceU8Ref(lit.data.slice())
            );
        }
    }
}

// in-place thread create

use std::sync::RwLock;

pub struct MTJoinable<T: Send + 'static, U: Send + 'static>(JoinHandle<T>, PhantomData<U>);
#[cfg(not(feature = "std"))]
impl<T: Send + 'static, U: Send + 'static + AnyBoxConstructor> Joinable<T, U> for MTJoinable<T, U> {
    fn join(self) -> Result<T, U> {
        match self.0.join() {
            Ok(t) => Ok(t),
            Err(_e) => Err(<U as AnyBoxConstructor>::new(())),
        }
    }
}
#[cfg(feature = "std")]
impl<T: Send + 'static, U: Send + 'static + AnyBoxConstructor> Joinable<T, U> for MTJoinable<T, U> {
    fn join(self) -> Result<T, U> {
        match self.0.join() {
            Ok(t) => Ok(t),
            Err(e) => Err(<U as AnyBoxConstructor>::new(e)),
        }
    }
}
pub struct MTOwnedRetriever<U: Send + 'static>(std::sync::Arc<RwLock<U>>);
impl<U: Send + 'static> Clone for MTOwnedRetriever<U> {
    fn clone(&self) -> Self {
        MTOwnedRetriever(self.0.clone())
    }
}
impl<U: Send + 'static> OwnedRetriever<U> for MTOwnedRetriever<U> {
    fn view<T, F: FnOnce(&U) -> T>(&self, f: F) -> Result<T, PoisonedThreadError> {
        match self.0.read() {
            Ok(u) => Ok(f(&*u)),
            Err(_) => Err(PoisonedThreadError::default()),
        }
    }
    fn unwrap(self) -> Result<U, PoisonedThreadError> {
        match std::sync::Arc::try_unwrap(self.0) {
            Ok(rwlock) => match rwlock.into_inner() {
                Ok(u) => Ok(u),
                Err(_) => Err(PoisonedThreadError::default()),
            },
            Err(_) => Err(PoisonedThreadError::default()),
        }
    }
}

#[derive(Default)]
pub struct MTSpawner {}

fn spawn_work<
    T: Send + 'static,
    ExtraInput: Send + 'static,
    F: Fn(ExtraInput, usize, usize, &U, Alloc) -> T + Send + 'static,
    Alloc: BrotliAlloc + Send + 'static,
    U: Send + 'static + Sync,
>(
    extra_input: ExtraInput,
    index: usize,
    num_threads: usize,
    locked_input: MTOwnedRetriever<U>,
    alloc: Alloc,
    f: F,
) -> std::thread::JoinHandle<T>
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    std::thread::spawn(move || {
        locked_input
            .view(move |guard: &U| -> T { f(extra_input, index, num_threads, guard, alloc) })
            .unwrap()
    })
}

impl<
        T: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > BatchSpawnable<T, ExtraInput, Alloc, U> for MTSpawner
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    type JoinHandle = MTJoinable<T, BrotliEncoderThreadError>;
    type FinalJoinHandle = MTOwnedRetriever<U>;
    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle {
        MTOwnedRetriever(std::sync::Arc::<RwLock<U>>::new(RwLock::new(
            mem::replace(input, Owned(InternalOwned::Borrowed)).unwrap(),
        )))
    }
    fn spawn<F: Fn(ExtraInput, usize, usize, &U, Alloc) -> T + Send + 'static + Copy>(
        &mut self,
        locked_input: &mut Self::FinalJoinHandle,
        work: &mut SendAlloc<T, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: F,
    ) {
        let (alloc, extra_input) = work.replace_with_default();
        let ret = spawn_work(
            extra_input,
            index,
            num_threads,
            locked_input.clone(),
            alloc,
            f,
        );
        *work = SendAlloc(InternalSendAlloc::Join(MTJoinable(ret, PhantomData)));
    }
}
impl<
        T: Send + 'static,
        ExtraInput: Send + 'static,
        Alloc: BrotliAlloc + Send + 'static,
        U: Send + 'static + Sync,
    > BatchSpawnableLite<T, ExtraInput, Alloc, U> for MTSpawner
where
    <Alloc as Allocator<u8>>::AllocatedMemory: Send + 'static,
{
    type JoinHandle = <MTSpawner as BatchSpawnable<T, ExtraInput, Alloc, U>>::JoinHandle;
    type FinalJoinHandle = <MTSpawner as BatchSpawnable<T, ExtraInput, Alloc, U>>::FinalJoinHandle;
    fn make_spawner(&mut self, input: &mut Owned<U>) -> Self::FinalJoinHandle {
        <Self as BatchSpawnable<T, ExtraInput, Alloc, U>>::make_spawner(self, input)
    }
    fn spawn(
        &mut self,
        handle: &mut Self::FinalJoinHandle,
        alloc_per_thread: &mut SendAlloc<T, ExtraInput, Alloc, Self::JoinHandle>,
        index: usize,
        num_threads: usize,
        f: fn(ExtraInput, usize, usize, &U, Alloc) -> T,
    ) {
        <Self as BatchSpawnable<T, ExtraInput, Alloc, U>>::spawn(
            self,
            handle,
            alloc_per_thread,
            index,
            num_threads,
            f,
        )
    }
}
