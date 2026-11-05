#![cfg_attr(feature = "benchmark", feature(test))]

extern crate alloc_no_stdlib;
extern crate brotli;
extern crate brotli_decompressor;
extern crate core;
#[cfg(feature = "validation")]
extern crate sha2;

pub mod integration_tests;
mod test_broccoli;
mod test_custom_dict;
mod test_threading;
mod tests;
mod util;
mod validate;

use core::cmp::{max, min};
use core::ops;
use std::env;
use std::fs::File;
use std::io::{self, Error, ErrorKind, Read, Seek, SeekFrom, Write};

use alloc_no_stdlib::{Allocator, SliceWrapper, SliceWrapperMut};
use brotli::enc::backward_references::BrotliEncoderMode;
use brotli::enc::threading::{
    BrotliEncoderThreadError, CompressMulti, CompressionThreadResult, Owned, SendAlloc,
};
use brotli::enc::{
    compress_worker_pool, new_work_pool, BrotliEncoderMaxCompressedSizeMulti, BrotliEncoderParams,
    UnionHasher, WorkerPool,
};
use brotli::CustomRead;

const MAX_THREADS: usize = 16;

pub struct Rebox<T> {
    b: Box<[T]>,
}

impl<T> From<Vec<T>> for Rebox<T> {
    fn from(data: Vec<T>) -> Self {
        Rebox::<T> {
            b: data.into_boxed_slice(),
        }
    }
}

impl<T> core::default::Default for Rebox<T> {
    fn default() -> Self {
        let v: Vec<T> = Vec::new();
        let b = v.into_boxed_slice();
        Rebox::<T> { b }
    }
}

impl<T> ops::Index<usize> for Rebox<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &(*self.b)[index]
    }
}

impl<T> ops::IndexMut<usize> for Rebox<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut (*self.b)[index]
    }
}

impl<T> alloc_no_stdlib::SliceWrapper<T> for Rebox<T> {
    fn slice(&self) -> &[T] {
        &self.b
    }
}

impl<T> alloc_no_stdlib::SliceWrapperMut<T> for Rebox<T> {
    fn slice_mut(&mut self) -> &mut [T] {
        &mut self.b
    }
}
#[derive(Clone, Copy, Default)]
pub struct HeapAllocator {}

impl<T: core::clone::Clone + Default> alloc_no_stdlib::Allocator<T> for HeapAllocator {
    type AllocatedMemory = Rebox<T>;
    fn alloc_cell(self: &mut HeapAllocator, len: usize) -> Rebox<T> {
        let v: Vec<T> = vec![T::default(); len];
        let b = v.into_boxed_slice();
        Rebox::<T> { b }
    }
    fn free_cell(self: &mut HeapAllocator, _data: Rebox<T>) {}
}

/*
type HeapBrotliAlloc = brotli::CombiningAllocator<
                      HeapAllocator<u8>,
                      HeapAllocator<u16>,
                      HeapAllocator<i32>,
                      HeapAllocator<u32>,
                      HeapAllocator<u64>,
                      HeapAllocator<Command>,
                      HeapAllocator<brotli::enc::floatX>,
                      HeapAllocator<v8>,
                      HeapAllocator<s16>,
                      HeapAllocator<brotli::enc::PDF>,
                      HeapAllocator<StaticCommand>,
                      HeapAllocator<HistogramLiteral>,
                      HeapAllocator<HistogramCommand>,
                      HeapAllocator<HistogramDistance>,
                      HeapAllocator<HistogramPair>,
                      HeapAllocator<ContextType>,
                      HeapAllocator<HuffmanTree>,
                      HeapAllocator<ZopfliNode>,
                  >;
*/

macro_rules! println_stderr(
    ($($val:tt)*) => { {
        writeln!(&mut ::std::io::stderr(), $($val)*).unwrap();
    } }
);

use std::path::Path;

// declare_stack_allocator_struct!(MemPool, 4096, global);

pub struct IoWriterWrapper<'a, OutputType: Write + 'a>(&'a mut OutputType);

pub struct IoReaderWrapper<'a, OutputType: Read + 'a>(&'a mut OutputType);

impl<'a, OutputType: Write> brotli::CustomWrite<io::Error> for IoWriterWrapper<'a, OutputType> {
    fn flush(&mut self) -> Result<(), io::Error> {
        loop {
            match self.0.flush() {
                Err(e) => match e.kind() {
                    ErrorKind::Interrupted => continue,
                    _ => return Err(e),
                },
                Ok(_) => return Ok(()),
            }
        }
    }
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        loop {
            match self.0.write(buf) {
                Err(e) => match e.kind() {
                    ErrorKind::Interrupted => continue,
                    _ => return Err(e),
                },
                Ok(cur_written) => return Ok(cur_written),
            }
        }
    }
}

impl<'a, InputType: Read> brotli::CustomRead<io::Error> for IoReaderWrapper<'a, InputType> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        loop {
            match self.0.read(buf) {
                Err(e) => match e.kind() {
                    ErrorKind::Interrupted => continue,
                    _ => return Err(e),
                },
                Ok(cur_read) => return Ok(cur_read),
            }
        }
    }
}

struct IntoIoReader<OutputType: Read>(OutputType);

impl<InputType: Read> brotli::CustomRead<io::Error> for IntoIoReader<InputType> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
        loop {
            match self.0.read(buf) {
                Err(e) => match e.kind() {
                    ErrorKind::Interrupted => continue,
                    _ => return Err(e),
                },
                Ok(cur_read) => return Ok(cur_read),
            }
        }
    }
}
#[cfg(not(feature = "seccomp"))]
pub fn decompress<InputType, OutputType>(
    r: &mut InputType,
    w: &mut OutputType,
    buffer_size: usize,
    custom_dictionary: Rebox<u8>,
) -> Result<(), io::Error>
where
    InputType: Read,
    OutputType: Write,
{
    let mut alloc_u8 = HeapAllocator::default();
    let mut input_buffer = alloc_u8.alloc_cell(buffer_size);
    let mut output_buffer = alloc_u8.alloc_cell(buffer_size);
    brotli::BrotliDecompressCustomIoCustomDict(
        &mut IoReaderWrapper::<InputType>(r),
        &mut IoWriterWrapper::<OutputType>(w),
        input_buffer.slice_mut(),
        output_buffer.slice_mut(),
        alloc_u8,
        HeapAllocator::default(),
        HeapAllocator::default(),
        custom_dictionary,
        Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF"),
    )
}
#[cfg(feature = "seccomp")]
extern "C" {
    fn calloc(n_elem: usize, el_size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn syscall(value: i32) -> i32;
    fn prctl(operation: i32, flags: u32) -> i32;
}
#[cfg(feature = "seccomp")]
const PR_SET_SECCOMP: i32 = 22;
#[cfg(feature = "seccomp")]
const SECCOMP_MODE_STRICT: u32 = 1;

#[cfg(feature = "seccomp")]
declare_stack_allocator_struct!(CallocAllocatedFreelist, 8192, calloc);

#[cfg(feature = "seccomp")]
pub fn decompress<InputType, OutputType>(
    r: &mut InputType,
    w: &mut OutputType,
    buffer_size: usize,
    custom_dictionary: Rebox<u8>,
) -> Result<(), io::Error>
where
    InputType: Read,
    OutputType: Write,
{
    if custom_dictionary.len() != 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "Not allowed to have a custom_dictionary with SECCOMP",
        ));
    }
    core::mem::drop(custom_dictionary);
    let u8_buffer = unsafe { define_allocator_memory_pool!(4, u8, [0; 1024 * 1024 * 200], calloc) };
    let u32_buffer = unsafe { define_allocator_memory_pool!(4, u32, [0; 16384], calloc) };
    let hc_buffer =
        unsafe { define_allocator_memory_pool!(4, HuffmanCode, [0; 1024 * 1024 * 16], calloc) };
    let mut alloc_u8 = CallocAllocatedFreelist::<u8>::new_allocator(u8_buffer.data, bzero);
    let alloc_u32 = CallocAllocatedFreelist::<u32>::new_allocator(u32_buffer.data, bzero);
    let alloc_hc = CallocAllocatedFreelist::<HuffmanCode>::new_allocator(hc_buffer.data, bzero);
    let ret = unsafe { prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT) };
    if ret != 0 {
        panic!("Unable to activate seccomp");
    }
    match brotli::BrotliDecompressCustomIo(
        &mut IoReaderWrapper::<InputType>(r),
        &mut IoWriterWrapper::<OutputType>(w),
        &mut alloc_u8.alloc_cell(buffer_size).slice_mut(),
        &mut alloc_u8.alloc_cell(buffer_size).slice_mut(),
        alloc_u8,
        alloc_u32,
        alloc_hc,
        Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF"),
    ) {
        Err(e) => Err(e),
        Ok(()) => {
            unsafe {
                syscall(60);
            };
            unreachable!()
        }
    }
}
pub fn new_brotli_heap_alloc() -> HeapAllocator {
    HeapAllocator::default()
}
impl brotli::enc::BrotliAlloc for HeapAllocator {}
pub fn compress_multi_nostd(
    input: Vec<u8>,
    output: &mut [u8],
    params: &BrotliEncoderParams,
    mut num_threads: usize,
) -> Result<usize, BrotliEncoderThreadError> {
    let mut alloc_array = [
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
    ];
    if num_threads > alloc_array.len() {
        num_threads = alloc_array.len();
    }
    CompressMulti(
        params,
        &mut Owned::new(Rebox::from(input)),
        output,
        &mut alloc_array[..num_threads],
        &mut util::MTSpawner::default(),
    )
}
pub fn compress_multi<InputType: Read, OutputType: Write>(
    r: &mut InputType,
    w: &mut OutputType,
    params: &BrotliEncoderParams,
    mut num_threads: usize,
    work_pool: Option<
        &mut WorkerPool<
            CompressionThreadResult<HeapAllocator>,
            UnionHasher<HeapAllocator>,
            HeapAllocator,
            (Rebox<u8>, BrotliEncoderParams),
        >,
    >,
) -> Result<usize, io::Error> {
    let mut input: Vec<u8> = Vec::<u8>::new();
    r.read_to_end(&mut input)?;
    let mut output = Rebox::from(vec![
        0u8;
        BrotliEncoderMaxCompressedSizeMulti(
            input.len(),
            num_threads
        )
    ]);
    let res = if let Some(worker_pool) = work_pool {
        let mut alloc_array = [
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
            SendAlloc::new(HeapAllocator::default(), UnionHasher::Uninit),
        ];
        if num_threads > alloc_array.len() {
            num_threads = alloc_array.len();
        }
        compress_worker_pool(
            params,
            &mut Owned::new(Rebox::from(input)),
            output.slice_mut(),
            &mut alloc_array[..num_threads],
            worker_pool,
        )
    } else {
        compress_multi_nostd(input, output.slice_mut(), params, num_threads)
    };

    match res {
        Ok(size) => {
            if let Err(err) = w.write_all(&output.slice()[..size]) {
                Err(err)
            } else {
                Ok(size)
            }
        }
        Err(err) => Err(io::Error::new(ErrorKind::Other, format!("{:?}", err))),
    }
}

pub fn compress<InputType, OutputType>(
    r: &mut InputType,
    w: &mut OutputType,
    buffer_size: usize,
    params: &brotli::enc::BrotliEncoderParams,
    custom_dictionary: &[u8],
    num_threads: usize,
) -> Result<usize, io::Error>
where
    InputType: Read,
    OutputType: Write,
{
    if num_threads > 1 && custom_dictionary.is_empty() && !params.log_meta_block {
        if has_stdlib() {
            return compress_multi(
                r,
                w,
                params,
                num_threads,
                Some(&mut new_work_pool(num_threads - 1)),
            );
        } else {
            return compress_multi(r, w, params, num_threads, None);
        }
    }
    let mut alloc_u8 = HeapAllocator::default();
    let mut input_buffer = alloc_u8.alloc_cell(buffer_size);
    let mut output_buffer = alloc_u8.alloc_cell(buffer_size);
    let mut log =
        |pm: &mut brotli::interface::PredictionModeContextMap<brotli::InputReferenceMut>,
         data: &mut [brotli::interface::Command<brotli::SliceOffset>],
         mb: brotli::InputPair,
         _mfv: &mut HeapAllocator| {
            let tmp = brotli::interface::Command::PredictionMode(
                brotli::interface::PredictionModeContextMap::<brotli::InputReference> {
                    literal_context_map: brotli::InputReference::from(&pm.literal_context_map),
                    predmode_speed_and_distance_context_map: brotli::InputReference::from(
                        &pm.predmode_speed_and_distance_context_map,
                    ),
                },
            );
            util::write_one(&tmp);
            for cmd in data.iter() {
                util::write_one(&cmd.thaw_pair(&mb));
            }
        };
    if params.log_meta_block {
        println_stderr!("window {} 0 0 0", params.lgwin);
    }
    brotli::BrotliCompressCustomIoCustomDict(
        &mut IoReaderWrapper::<InputType>(r),
        &mut IoWriterWrapper::<OutputType>(w),
        input_buffer.slice_mut(),
        output_buffer.slice_mut(),
        params,
        new_brotli_heap_alloc(),
        &mut log,
        custom_dictionary,
        Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF"),
    )
}

// This decompressor is defined unconditionally on whether std is defined
// so we can exercise the code in any case
pub struct BrotliDecompressor<R: Read>(
    brotli::DecompressorCustomIo<
        io::Error,
        IntoIoReader<R>,
        Rebox<u8>,
        HeapAllocator,
        HeapAllocator,
        HeapAllocator,
    >,
);

impl<R: Read> BrotliDecompressor<R> {
    pub fn new(r: R, buffer_size: usize) -> Self {
        let mut alloc_u8 = HeapAllocator::default();
        let buffer = alloc_u8.alloc_cell(buffer_size);
        let alloc_u32 = HeapAllocator::default();
        let alloc_hc = HeapAllocator::default();
        BrotliDecompressor::<R>(brotli::DecompressorCustomIo::<
            Error,
            IntoIoReader<R>,
            Rebox<u8>,
            HeapAllocator,
            HeapAllocator,
            HeapAllocator,
        >::new(
            IntoIoReader::<R>(r),
            buffer,
            alloc_u8,
            alloc_u32,
            alloc_hc,
            io::Error::new(ErrorKind::InvalidData, "Invalid Data"),
        ))
    }
}

impl<R: Read> Read for BrotliDecompressor<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, Error> {
        self.0.read(buf)
    }
}

#[cfg(test)]
fn writeln0<OutputType: Write>(
    strm: &mut OutputType,
    data: &str,
) -> core::result::Result<(), io::Error> {
    writeln!(strm, "{:}", data)
}
#[cfg(test)]
fn writeln_time<OutputType: Write>(
    strm: &mut OutputType,
    data: &str,
    v0: u64,
    v1: u64,
    v2: u32,
) -> core::result::Result<(), io::Error> {
    writeln!(strm, "{:} {:} {:}.{:09}", v0, data, v1, v2)
}

#[allow(non_upper_case_globals)]
pub const kMaxDictionarySize: usize = 50331660;

fn read_custom_dictionary(filename: &str) -> Vec<u8> {
    let mut dict = match File::open(Path::new(&filename)) {
        Err(why) => panic!("couldn't open custom dictionary {:}\n{:}", filename, why),
        Ok(file) => file,
    };
    let mut ret = Vec::<u8>::new();
    dict.read_to_end(&mut ret).unwrap();
    if ret.len() > kMaxDictionarySize {
        panic!("dictionary [{}] is larger than maximum allowed: {}\n",
               filename,
               kMaxDictionarySize);
    }
    ret
}

#[cfg(feature = "std")]
fn has_stdlib() -> bool {
    true
}
#[cfg(not(feature = "std"))]
fn has_stdlib() -> bool {
    false
}

fn main() {
    let mut buffer_size = 65536;
    let mut do_compress = false;
    let mut do_validate = false;
    let mut params = brotli::enc::BrotliEncoderInitParams();
    let mut custom_dictionary = Vec::<u8>::new();
    let mut use_work_pool = has_stdlib();
    params.quality = 11; // default
    let mut filenames = [std::string::String::new(), std::string::String::new()];
    let mut num_benchmarks = 1;
    let mut double_dash = false;
    let mut num_threads = 1;
    if env::args_os().len() > 1 {
        let mut first = true;
        for argument in env::args() {
            if first {
                first = false;
                continue;
            }
            if argument == "--" {
                double_dash = true;
                continue;
            }
            if (argument == "-catable" || argument == "--catable") && !double_dash {
                params.catable = true;
                params.use_dictionary = false;
                params.appendable = true;
                continue;
            }
            if (argument == "-nothreadpool" || argument == "--nothreadpool") && !double_dash {
                use_work_pool = false;
                continue;
            }

            if (argument == "-appendable" || argument == "--appendable") && !double_dash {
                params.appendable = true;
                continue;
            }
            if (argument.starts_with("-magic") || argument.starts_with("--magic")) && !double_dash {
                params.magic_number = true;
                continue;
            }
            if argument.starts_with("-customdictionary=") && !double_dash {
                for item in argument.splitn(2, |c| c == '=').skip(1) {
                    custom_dictionary = read_custom_dictionary(item);
                }
                continue;
            }
            if argument == "--dump-dictionary" && !double_dash {
                util::print_dictionary(util::permute_dictionary());
                return;
            }
            if argument == "-utf8" && !double_dash {
                params.mode = BrotliEncoderMode::BROTLI_FORCE_UTF8_PRIOR;
                continue;
            }
            if argument == "-msb" && !double_dash {
                params.mode = BrotliEncoderMode::BROTLI_FORCE_MSB_PRIOR;
                continue;
            }
            if argument == "-lsb" && !double_dash {
                params.mode = BrotliEncoderMode::BROTLI_FORCE_LSB_PRIOR;
                continue;
            }
            if argument == "-signed" && !double_dash {
                params.mode = BrotliEncoderMode::BROTLI_FORCE_SIGNED_PRIOR;
                continue;
            }
            if argument == "-efficient" && !double_dash {
                params.favor_cpu_efficiency = true;
                continue;
            }
            if argument == "-lowlatency" && !double_dash {
                params.favor_cpu_efficiency = false;
                continue;
            }
            if argument == "-i" && !double_dash {
                // display the intermediate representation of metablocks
                params.log_meta_block = true;
                continue;
            }
            if (argument == "-0" || argument == "-q0") && !double_dash {
                params.quality = 0;
                continue;
            }
            if (argument == "-1" || argument == "-q1") && !double_dash {
                params.quality = 1;
                continue;
            }
            if (argument == "-2" || argument == "-q2") && !double_dash {
                params.quality = 2;
                continue;
            }
            if (argument == "-3" || argument == "-q3") && !double_dash {
                params.quality = 3;
                continue;
            }
            if (argument == "-4" || argument == "-q4") && !double_dash {
                params.quality = 4;
                continue;
            }
            if (argument == "-5" || argument == "-q5") && !double_dash {
                params.quality = 5;
                continue;
            }
            if (argument == "-6" || argument == "-q6") && !double_dash {
                params.quality = 6;
                continue;
            }
            if (argument == "-7" || argument == "-q7") && !double_dash {
                params.quality = 7;
                continue;
            }
            if (argument == "-8" || argument == "-q8") && !double_dash {
                params.quality = 8;
                continue;
            }
            if (argument == "-9" || argument == "-q9") && !double_dash {
                params.quality = 9;
                continue;
            }
            if (argument == "-9.5" || argument == "-q9.5") && !double_dash {
                params.quality = 10;
                params.q9_5 = true;
                continue;
            }
            if (argument == "-9.5x" || argument == "-q9.5x") && !double_dash {
                params.quality = 11;
                params.q9_5 = true;
                continue;
            }
            if (argument == "-10" || argument == "-q10") && !double_dash {
                params.quality = 10;
                continue;
            }
            if (argument == "-11" || argument == "-q11") && !double_dash {
                params.quality = 11;
                continue;
            }
            if (argument == "-q9.5y") && !double_dash {
                params.quality = 12;
                params.q9_5 = true;
                continue;
            }
            if argument.starts_with("-l") && !double_dash {
                params.lgblock = argument
                    .trim_matches('-')
                    .trim_matches('l')
                    .parse::<i32>()
                    .unwrap();
                continue;
            }
            if argument.starts_with("-j") && !double_dash {
                num_threads = min(
                    max(
                        1,
                        argument
                            .trim_matches('-')
                            .trim_matches('j')
                            .parse::<i32>()
                            .unwrap() as usize,
                    ),
                    MAX_THREADS,
                );
                continue;
            }
            if argument.starts_with("-bytescore=") && !double_dash {
                params.hasher.literal_byte_score = argument
                    .trim_matches('-')
                    .trim_matches('b')
                    .trim_matches('y')
                    .trim_matches('t')
                    .trim_matches('e')
                    .trim_matches('s')
                    .trim_matches('c')
                    .trim_matches('o')
                    .trim_matches('r')
                    .trim_matches('e')
                    .trim_matches('=')
                    .parse::<i32>()
                    .unwrap();
                continue;
            }
            if argument.starts_with("-w") && !double_dash {
                params.lgwin = argument
                    .trim_matches('-')
                    .trim_matches('w')
                    .parse::<i32>()
                    .unwrap();
                continue;
            }
            if (argument == "-validate" || argument == "--validate") && !double_dash {
                do_validate = true;
                continue;
            }
            if argument.starts_with("-bs") && !double_dash {
                buffer_size = argument
                    .trim_matches('-')
                    .trim_matches('b')
                    .trim_matches('s')
                    .trim_matches('=')
                    .parse::<usize>()
                    .unwrap();
                continue;
            }
            if argument.starts_with("-findprior") && !double_dash {
                params.prior_bitmask_detection = 1;
                continue;
            }
            if argument.starts_with("-findspeed=") && !double_dash {
                params.cdf_adaptation_detection = argument
                    .trim_matches('-')
                    .trim_matches('f')
                    .trim_matches('i')
                    .trim_matches('n')
                    .trim_matches('d')
                    .trim_matches('r')
                    .trim_matches('a')
                    .trim_matches('n')
                    .trim_matches('d')
                    .trim_matches('o')
                    .trim_matches('m')
                    .trim_matches('=')
                    .parse::<u32>()
                    .unwrap() as u8;
                continue;
            } else if argument == "-findspeed" && !double_dash {
                params.cdf_adaptation_detection = 1;
                continue;
            }
            if argument == "-basicstride" && !double_dash {
                params.stride_detection_quality = 1;
                continue;
            } else if argument == "-advstride" && !double_dash {
                params.stride_detection_quality = 3;
                continue;
            } else if argument == "-stride" && !double_dash {
                params.stride_detection_quality = 2;
                continue;
            } else if (argument.starts_with("-s") && !argument.starts_with("-speed="))
                && !double_dash
            {
                params.size_hint = argument
                    .trim_matches('-')
                    .trim_matches('s')
                    .parse::<usize>()
                    .unwrap();
                continue;
            }
            if argument.starts_with("-speed=") && !double_dash {
                let comma_string = argument
                    .trim_matches('-')
                    .trim_matches('s')
                    .trim_matches('p')
                    .trim_matches('e')
                    .trim_matches('e')
                    .trim_matches('d')
                    .trim_matches('=');
                let split = comma_string.split(',');
                for (index, s) in split.enumerate() {
                    let data = s.parse::<u16>().unwrap();
                    if data > 16384 {
                        println_stderr!("Speed must be <= 16384, not {}", data);
                    }
                    if index == 0 {
                        for item in params.literal_adaptation.iter_mut() {
                            item.0 = data;
                        }
                    } else if index == 1 {
                        for item in params.literal_adaptation.iter_mut() {
                            item.1 = data;
                        }
                    } else if (index & 1) == 0 {
                        params.literal_adaptation[index / 2].0 = data;
                    } else {
                        params.literal_adaptation[index / 2].1 = data;
                    }
                }
                continue;
            }
            if argument == "-avoiddistanceprefixsearch" && !double_dash {
                params.avoid_distance_prefix_search = true;
            }
            if argument.starts_with("-b") && !double_dash {
                num_benchmarks = argument
                    .trim_matches('-')
                    .trim_matches('b')
                    .parse::<usize>()
                    .unwrap();
                continue;
            }
            if argument == "-c" && !double_dash {
                do_compress = true;
                continue;
            }
            if argument == "-h" || argument == "-help" || argument == "--help" && !double_dash {
                println_stderr!("Decompression:\nbrotli [input_file] [output_file]\nCompression:brotli -c -q9.5 -w22 [input_file] [output_file]\nQuality may be one of -q9.5 -q9.5x -q9.5y or -q[0-11] for standard brotli settings.\nOptional size hint -s<size> to direct better compression\n\nThe -i parameter produces a cross human readdable IR representation of the file.\nThis can be ingested by other compressors.\nIR-specific options include:\n-findprior\n-speed=<inc,max,inc,max,inc,max,inc,max>");
                return;
            }
            if filenames[0].is_empty() {
                filenames[0] = argument.clone();
                continue;
            }
            if filenames[1].is_empty() {
                filenames[1] = argument.clone();
                continue;
            }
            panic!("Unknown Argument {:}", argument);
        }
        if !filenames[0].is_empty() {
            let mut input = match File::open(Path::new(&filenames[0])) {
                Err(why) => panic!("couldn't open {:}\n{:}", filenames[0], why),
                Ok(file) => file,
            };
            if !filenames[1].is_empty() {
                let mut output = match File::create(Path::new(&filenames[1])) {
                    Err(why) => panic!(
                        "couldn't open file for writing: {:}\n{:}",
                        filenames[1], why
                    ),
                    Ok(file) => file,
                };
                let mut worker_pool =
                    if num_threads != 1 && do_compress && use_work_pool && !do_validate {
                        Some(new_work_pool(num_threads - 1))
                    } else {
                        None
                    };
                for i in 0..num_benchmarks {
                    if do_validate {
                        let dict = core::mem::take(&mut custom_dictionary);
                        if num_benchmarks > 0 {
                            custom_dictionary = dict.clone();
                        }
                        match validate::compress_validate(
                            &mut input,
                            &mut output,
                            buffer_size,
                            &params,
                            dict.into(),
                            num_threads,
                        ) {
                            Ok(_) => {}
                            Err(e) => panic!("Error {:?}", e),
                        }
                    } else if do_compress {
                        if let Some(ref mut work_pool) = worker_pool {
                            match compress_multi(
                                &mut input,
                                &mut output,
                                &params,
                                num_threads,
                                Some(work_pool),
                            ) {
                                Ok(_) => {}
                                Err(e) => panic!("Error {:?}", e),
                            }
                        } else if num_threads != 1 {
                            match compress_multi(
                                &mut input,
                                &mut output,
                                &params,
                                num_threads,
                                None,
                            ) {
                                Ok(_) => {}
                                Err(e) => panic!("Error {:?}", e),
                            }
                        } else {
                            match compress(
                                &mut input,
                                &mut output,
                                buffer_size,
                                &params,
                                &custom_dictionary[..],
                                num_threads,
                            ) {
                                Ok(_) => {}
                                Err(e) => panic!("Error {:?}", e),
                            }
                        }
                    } else {
                        let dict = core::mem::take(&mut custom_dictionary);
                        if num_benchmarks > 0 {
                            custom_dictionary = dict.clone();
                        }
                        match decompress(&mut input, &mut output, buffer_size, dict.into()) {
              Ok(_) => {}
              Err(e) => panic!("Error: {:} during brotli decompress\nTo compress with Brotli, specify the -c flag.", e),
            }
                    }
                    if i + 1 != num_benchmarks {
                        input.seek(SeekFrom::Start(0)).unwrap();
                        output.seek(SeekFrom::Start(0)).unwrap();
                    }
                }
                drop(output);
            } else {
                assert_eq!(num_benchmarks, 1);
                if do_validate {
                    if do_compress {
                        match validate::compress_validate(
                            &mut input,
                            &mut io::stdout(),
                            buffer_size,
                            &params,
                            custom_dictionary.into(),
                            num_threads,
                        ) {
                            Ok(_) => {}
                            Err(e) => panic!("Error {:?}", e),
                        }
                    } else {
                        match validate::compress_validate(
                            &mut input,
                            &mut io::sink(),
                            buffer_size,
                            &params,
                            custom_dictionary.into(),
                            num_threads,
                        ) {
                            Ok(_) => {}
                            Err(e) => panic!("Error {:?}", e),
                        }
                    }
                } else if do_compress {
                    match compress(
                        &mut input,
                        &mut io::stdout(),
                        buffer_size,
                        &params,
                        &custom_dictionary[..],
                        num_threads,
                    ) {
                        Ok(_) => {}
                        Err(e) => panic!("Error {:?}", e),
                    }
                } else {
                    match decompress(&mut input, &mut io::stdout(), buffer_size, custom_dictionary.into()) {
            Ok(_) => {}
            Err(e) => panic!("Error: {:} during brotli decompress\nTo compress with Brotli, specify the -c flag.", e),
          }
                }
            }
            drop(input);
        } else {
            assert_eq!(num_benchmarks, 1);
            if do_validate {
                if do_compress {
                    match validate::compress_validate(
                        &mut io::stdin(),
                        &mut io::stdout(),
                        buffer_size,
                        &params,
                        custom_dictionary.into(),
                        num_threads,
                    ) {
                        Ok(_) => {}
                        Err(e) => panic!("Error {:?}", e),
                    }
                } else {
                    match validate::compress_validate(
                        &mut io::stdin(),
                        &mut io::sink(),
                        buffer_size,
                        &params,
                        custom_dictionary.into(),
                        num_threads,
                    ) {
                        Ok(_) => {}
                        Err(e) => panic!("Error {:?}", e),
                    }
                }
            } else if do_compress {
                match compress(
                    &mut io::stdin(),
                    &mut io::stdout(),
                    buffer_size,
                    &params,
                    &custom_dictionary[..],
                    num_threads,
                ) {
                    Ok(_) => return,
                    Err(e) => panic!("Error {:?}", e),
                }
            } else {
                match decompress(&mut io::stdin(), &mut io::stdout(), buffer_size, custom_dictionary.into()) {
          Ok(_) => return,
          Err(e) => panic!("Error: {:} during brotli decompress\nTo compress with Brotli, specify the -c flag.", e),
        }
            }
        }
    } else {
        assert_eq!(num_benchmarks, 1);
        match decompress(&mut io::stdin(), &mut io::stdout(), buffer_size, custom_dictionary.into()) {
      Ok(_) => (),
      Err(e) => panic!("Error: {:} during brotli decompress\nTo compress with Brotli, specify the -c flag.", e),
    }
    }
}
