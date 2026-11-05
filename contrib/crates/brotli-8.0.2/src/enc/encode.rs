use alloc::Allocator;
use core;
use core::cmp::{max, min};

use super::super::alloc;
use super::super::alloc::{SliceWrapper, SliceWrapperMut};
use super::backward_references::{
    AdvHashSpecialization, AdvHasher, AnyHasher, BasicHasher, BrotliCreateBackwardReferences,
    BrotliEncoderMode, BrotliEncoderParams, BrotliHasherParams, H2Sub, H3Sub, H4Sub, H54Sub, H5Sub,
    H6Sub, HQ5Sub, HQ7Sub, HowPrepared, StoreLookaheadThenStore, Struct1, UnionHasher, H9,
    H9_BLOCK_BITS, H9_BLOCK_SIZE, H9_BUCKET_BITS, H9_NUM_LAST_DISTANCES_TO_CHECK,
};
use super::bit_cost::{shannon_entropy, BitsEntropy};
use super::brotli_bit_stream::{
    store_meta_block, store_meta_block_fast, store_meta_block_trivial,
    store_uncompressed_meta_block, BrotliWriteEmptyLastMetaBlock, BrotliWriteMetadataMetaBlock,
    MetaBlockSplit, RecoderState,
};
use super::combined_alloc::BrotliAlloc;
use super::command::{get_length_code, BrotliDistanceParams, Command};
use super::compress_fragment::compress_fragment_fast;
use super::compress_fragment_two_pass::{compress_fragment_two_pass, BrotliWriteBits};
use super::constants::{
    BROTLI_CONTEXT, BROTLI_CONTEXT_LUT, BROTLI_MAX_NDIRECT, BROTLI_MAX_NPOSTFIX,
    BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS, BROTLI_WINDOW_GAP,
};
use super::hash_to_binary_tree::InitializeH10;
use super::histogram::{
    ContextType, CostAccessors, HistogramCommand, HistogramDistance, HistogramLiteral,
};
use super::interface;
use super::metablock::{
    BrotliBuildMetaBlock, BrotliBuildMetaBlockGreedy, BrotliInitDistanceParams,
    BrotliOptimizeHistograms,
};
pub use super::parameters::BrotliEncoderParameter;
use super::static_dict::{kNumDistanceCacheEntries, BrotliGetDictionary};
use super::util::{floatX, Log2FloorNonZero};
use crate::enc::combined_alloc::{alloc_default, allocate};
use crate::enc::input_pair::InputReferenceMut;
use crate::enc::utf8_util::is_mostly_utf8;

//fn BrotliCreateHqZopfliBackwardReferences(m: &mut [MemoryManager],
//                                          dictionary: &[BrotliDictionary],
//                                          num_bytes: usize,
//                                          position: usize,
//                                          ringbuffer: &[u8],
//                                          ringbuffer_mask: usize,
//                                          params: &[BrotliEncoderParams],
//                                          hasher: &mut [u8],
//                                          dist_cache: &mut [i32],
//                                          last_insert_len: &mut [usize],
//                                          commands: &mut [Command],
//                                          num_commands: &mut [usize],
//                                          num_literals: &mut [usize]);
//fn BrotliCreateZopfliBackwardReferences(m: &mut [MemoryManager],
//                                       dictionary: &[BrotliDictionary],
//                                      num_bytes: usize,
//                                        position: usize,
//                                        ringbuffer: &[u8],
//                                        ringbuffer_mask: usize,
//                                        params: &[BrotliEncoderParams],
//                                        hasher: &mut [u8],
//                                        dist_cache: &mut [i32],
//                                        last_insert_len: &mut [usize],
//                                        commands: &mut [Command],
//                                        num_commands: &mut [usize],
//                                        num_literals: &mut [usize]);
//fn BrotliInitBlockSplit(xself: &mut BlockSplit);
//fn BrotliInitMemoryManager(m: &mut [MemoryManager],
//                           alloc_func: fn(&mut [::std::os::raw::c_void], usize)
//                                          -> *mut ::std::os::raw::c_void,
//                           free_func: fn(*mut ::std::os::raw::c_void,
//                                         *mut ::std::os::raw::c_void),
//                           opaque: *mut ::std::os::raw::c_void);
//fn BrotliInitZopfliNodes(array: &mut [ZopfliNode], length: usize);
//fn BrotliWipeOutMemoryManager(m: &mut [MemoryManager]);

static kCompressFragmentTwoPassBlockSize: usize = (1i32 << 17) as usize;

static kMinUTF8Ratio: floatX = 0.75;

pub struct RingBuffer<AllocU8: alloc::Allocator<u8>> {
    pub size_: u32,
    pub mask_: u32,
    pub tail_size_: u32,
    pub total_size_: u32,
    pub cur_size_: u32,
    pub pos_: u32,
    pub data_mo: AllocU8::AllocatedMemory,
    pub buffer_index: usize,
}

#[derive(PartialEq, Eq, Copy, Clone)]
#[repr(i32)]
pub enum BrotliEncoderStreamState {
    BROTLI_STREAM_PROCESSING = 0,
    BROTLI_STREAM_FLUSH_REQUESTED = 1,
    BROTLI_STREAM_FINISHED = 2,
    BROTLI_STREAM_METADATA_HEAD = 3,
    BROTLI_STREAM_METADATA_BODY = 4,
}

#[derive(Clone, Copy, Debug)]
enum NextOut {
    DynamicStorage(u32),
    TinyBuf(u32),
    None,
}
fn GetNextOutInternal<'a>(
    next_out: &NextOut,
    storage: &'a mut [u8],
    tiny_buf: &'a mut [u8; 16],
) -> &'a mut [u8] {
    match next_out {
        &NextOut::DynamicStorage(offset) => &mut storage[offset as usize..],
        &NextOut::TinyBuf(offset) => &mut tiny_buf[offset as usize..],
        &NextOut::None => &mut [],
    }
}
macro_rules! GetNextOut {
    ($s : expr) => {
        GetNextOutInternal(&$s.next_out_, $s.storage_.slice_mut(), &mut $s.tiny_buf_)
    };
}
fn NextOutIncrement(next_out: &NextOut, inc: i32) -> NextOut {
    match next_out {
        &NextOut::DynamicStorage(offset) => NextOut::DynamicStorage((offset as i32 + inc) as u32),
        &NextOut::TinyBuf(offset) => NextOut::TinyBuf((offset as i32 + inc) as u32),
        &NextOut::None => NextOut::None,
    }
}
fn IsNextOutNull(next_out: &NextOut) -> bool {
    match next_out {
        &NextOut::DynamicStorage(_) => false,
        &NextOut::TinyBuf(_) => false,
        &NextOut::None => true,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IsFirst {
    NothingWritten,
    HeaderWritten,
    FirstCatableByteWritten,
    BothCatableBytesWritten,
}

pub struct BrotliEncoderStateStruct<Alloc: BrotliAlloc> {
    pub params: BrotliEncoderParams,
    pub m8: Alloc,
    pub hasher_: UnionHasher<Alloc>,
    pub input_pos_: u64,
    pub ringbuffer_: RingBuffer<Alloc>,
    pub cmd_alloc_size_: usize,
    pub commands_: <Alloc as Allocator<Command>>::AllocatedMemory, // not sure about this one
    pub num_commands_: usize,
    pub num_literals_: usize,
    pub last_insert_len_: usize,
    pub last_flush_pos_: u64,
    pub last_processed_pos_: u64,
    pub dist_cache_: [i32; 16],
    pub saved_dist_cache_: [i32; kNumDistanceCacheEntries],
    pub last_bytes_: u16,
    pub last_bytes_bits_: u8,
    pub prev_byte_: u8,
    pub prev_byte2_: u8,
    pub storage_size_: usize,
    pub storage_: <Alloc as Allocator<u8>>::AllocatedMemory,
    pub small_table_: [i32; 1024],
    pub large_table_: <Alloc as Allocator<i32>>::AllocatedMemory,
    //  pub large_table_size_: usize, // <-- get this by doing large_table_.len()
    pub cmd_depths_: [u8; 128],
    pub cmd_bits_: [u16; 128],
    pub cmd_code_: [u8; 512],
    pub cmd_code_numbits_: usize,
    pub command_buf_: <Alloc as Allocator<u32>>::AllocatedMemory,
    pub literal_buf_: <Alloc as Allocator<u8>>::AllocatedMemory,
    next_out_: NextOut,
    pub available_out_: usize,
    pub total_out_: u64,
    pub tiny_buf_: [u8; 16],
    pub remaining_metadata_bytes_: u32,
    pub stream_state_: BrotliEncoderStreamState,
    pub is_last_block_emitted_: bool,
    pub is_initialized_: bool,
    pub is_first_mb: IsFirst,
    pub literal_scratch_space: <HistogramLiteral as CostAccessors>::i32vec,
    pub command_scratch_space: <HistogramCommand as CostAccessors>::i32vec,
    pub distance_scratch_space: <HistogramDistance as CostAccessors>::i32vec,
    pub recoder_state: RecoderState,
    custom_dictionary_size: Option<core::num::NonZeroUsize>,
    custom_dictionary: bool,
}

pub fn set_parameter(
    params: &mut BrotliEncoderParams,
    p: BrotliEncoderParameter,
    value: u32,
) -> bool {
    use crate::enc::parameters::BrotliEncoderParameter::*;
    match p {
        BROTLI_PARAM_MODE => {
            params.mode = match value {
                0 => BrotliEncoderMode::BROTLI_MODE_GENERIC,
                1 => BrotliEncoderMode::BROTLI_MODE_TEXT,
                2 => BrotliEncoderMode::BROTLI_MODE_FONT,
                3 => BrotliEncoderMode::BROTLI_FORCE_LSB_PRIOR,
                4 => BrotliEncoderMode::BROTLI_FORCE_MSB_PRIOR,
                5 => BrotliEncoderMode::BROTLI_FORCE_UTF8_PRIOR,
                6 => BrotliEncoderMode::BROTLI_FORCE_SIGNED_PRIOR,
                _ => BrotliEncoderMode::BROTLI_MODE_GENERIC,
            };
        }
        BROTLI_PARAM_QUALITY => params.quality = value as i32,
        BROTLI_PARAM_STRIDE_DETECTION_QUALITY => params.stride_detection_quality = value as u8,
        BROTLI_PARAM_HIGH_ENTROPY_DETECTION_QUALITY => {
            params.high_entropy_detection_quality = value as u8
        }
        BROTLI_PARAM_CDF_ADAPTATION_DETECTION => params.cdf_adaptation_detection = value as u8,
        BROTLI_PARAM_Q9_5 => params.q9_5 = (value != 0),
        BROTLI_PARAM_PRIOR_BITMASK_DETECTION => params.prior_bitmask_detection = value as u8,
        BROTLI_PARAM_SPEED => {
            params.literal_adaptation[1].0 = value as u16;
            if params.literal_adaptation[0] == (0, 0) {
                params.literal_adaptation[0].0 = value as u16;
            }
        }
        BROTLI_PARAM_SPEED_MAX => {
            params.literal_adaptation[1].1 = value as u16;
            if params.literal_adaptation[0].1 == 0 {
                params.literal_adaptation[0].1 = value as u16;
            }
        }
        BROTLI_PARAM_CM_SPEED => {
            params.literal_adaptation[3].0 = value as u16;
            if params.literal_adaptation[2] == (0, 0) {
                params.literal_adaptation[2].0 = value as u16;
            }
        }
        BROTLI_PARAM_CM_SPEED_MAX => {
            params.literal_adaptation[3].1 = value as u16;
            if params.literal_adaptation[2].1 == 0 {
                params.literal_adaptation[2].1 = value as u16;
            }
        }
        BROTLI_PARAM_SPEED_LOW => params.literal_adaptation[0].0 = value as u16,
        BROTLI_PARAM_SPEED_LOW_MAX => params.literal_adaptation[0].1 = value as u16,
        BROTLI_PARAM_CM_SPEED_LOW => params.literal_adaptation[2].0 = value as u16,
        BROTLI_PARAM_CM_SPEED_LOW_MAX => params.literal_adaptation[2].1 = value as u16,
        BROTLI_PARAM_LITERAL_BYTE_SCORE => params.hasher.literal_byte_score = value as i32,
        BROTLI_METABLOCK_CALLBACK => params.log_meta_block = value != 0,
        BROTLI_PARAM_LGWIN => params.lgwin = value as i32,
        BROTLI_PARAM_LGBLOCK => params.lgblock = value as i32,
        BROTLI_PARAM_DISABLE_LITERAL_CONTEXT_MODELING => {
            if value != 0 && value != 1 {
                return false;
            }
            params.disable_literal_context_modeling = if value != 0 { 1 } else { 0 };
        }
        BROTLI_PARAM_SIZE_HINT => params.size_hint = value as usize,
        BROTLI_PARAM_LARGE_WINDOW => params.large_window = value != 0,
        BROTLI_PARAM_AVOID_DISTANCE_PREFIX_SEARCH => {
            params.avoid_distance_prefix_search = value != 0
        }
        BROTLI_PARAM_CATABLE => {
            params.catable = value != 0;
            if !params.appendable {
                params.appendable = value != 0;
            }
            params.use_dictionary = (value == 0);
        }
        BROTLI_PARAM_APPENDABLE => params.appendable = value != 0,
        BROTLI_PARAM_MAGIC_NUMBER => params.magic_number = value != 0,
        BROTLI_PARAM_FAVOR_EFFICIENCY => params.favor_cpu_efficiency = value != 0,
        _ => return false,
    }
    true
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    pub fn set_parameter(&mut self, p: BrotliEncoderParameter, value: u32) -> bool {
        if self.is_initialized_ {
            false
        } else {
            set_parameter(&mut self.params, p, value)
        }
    }
}

/* "Large Window Brotli" */
pub const BROTLI_LARGE_MAX_DISTANCE_BITS: u32 = 62;
pub const BROTLI_LARGE_MIN_WBITS: u32 = 10;
pub const BROTLI_LARGE_MAX_WBITS: u32 = 30;

pub const BROTLI_MAX_DISTANCE_BITS: u32 = 24;
pub const BROTLI_MAX_WINDOW_BITS: usize = BROTLI_MAX_DISTANCE_BITS as usize;
pub const BROTLI_MAX_DISTANCE: usize = 0x03ff_fffc;
pub const BROTLI_MAX_ALLOWED_DISTANCE: usize = 0x07ff_fffc;
pub const BROTLI_NUM_DISTANCE_SHORT_CODES: u32 = 16;
pub fn BROTLI_DISTANCE_ALPHABET_SIZE(NPOSTFIX: u32, NDIRECT: u32, MAXNBITS: u32) -> u32 {
    BROTLI_NUM_DISTANCE_SHORT_CODES + (NDIRECT) + ((MAXNBITS) << ((NPOSTFIX) + 1))
}

//#define BROTLI_NUM_DISTANCE_SYMBOLS \
//    BROTLI_DISTANCE_ALPHABET_SIZE(  \
//        BROTLI_MAX_NDIRECT, BROTLI_MAX_NPOSTFIX, BROTLI_LARGE_MAX_DISTANCE_BITS)

pub const BROTLI_NUM_DISTANCE_SYMBOLS: usize = 1128;

pub fn BrotliEncoderInitParams() -> BrotliEncoderParams {
    BrotliEncoderParams {
        dist: BrotliDistanceParams {
            distance_postfix_bits: 0,
            num_direct_distance_codes: 0,
            alphabet_size: BROTLI_DISTANCE_ALPHABET_SIZE(0, 0, BROTLI_MAX_DISTANCE_BITS),
            max_distance: BROTLI_MAX_DISTANCE,
        },
        mode: BrotliEncoderMode::BROTLI_MODE_GENERIC,
        log_meta_block: false,
        large_window: false,
        avoid_distance_prefix_search: false,
        quality: 11,
        q9_5: false,
        lgwin: 22i32,
        lgblock: 0i32,
        size_hint: 0usize,
        disable_literal_context_modeling: 0i32,
        stride_detection_quality: 0,
        high_entropy_detection_quality: 0,
        cdf_adaptation_detection: 0,
        prior_bitmask_detection: 0,
        literal_adaptation: [(0, 0); 4],
        catable: false,
        use_dictionary: true,
        appendable: false,
        magic_number: false,
        favor_cpu_efficiency: false,
        hasher: BrotliHasherParams {
            type_: 6,
            block_bits: 9 - 1,
            bucket_bits: 15,
            hash_len: 5,
            num_last_distances_to_check: 16,
            literal_byte_score: 0,
        },
    }
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn extend_last_command(&mut self, bytes: &mut u32, wrapped_last_processed_pos: &mut u32) {
        let last_command = &mut self.commands_.slice_mut()[self.num_commands_ - 1];

        let mask = self.ringbuffer_.mask_;
        let max_backward_distance: u64 = (1u64 << self.params.lgwin) - BROTLI_WINDOW_GAP as u64;
        let last_copy_len = u64::from(last_command.copy_len_) & 0x01ff_ffff;
        let last_processed_pos: u64 = self.last_processed_pos_ - last_copy_len;
        let max_distance: u64 = if last_processed_pos < max_backward_distance {
            last_processed_pos
        } else {
            max_backward_distance
        };
        let cmd_dist: u64 = self.dist_cache_[0] as u64;
        let distance_code: u32 = last_command.restore_distance_code(&self.params.dist);
        if (distance_code < BROTLI_NUM_DISTANCE_SHORT_CODES
            || distance_code as u64 - (BROTLI_NUM_DISTANCE_SHORT_CODES - 1) as u64 == cmd_dist)
        {
            if (cmd_dist <= max_distance) {
                while (*bytes != 0
                    && self.ringbuffer_.data_mo.slice()[self.ringbuffer_.buffer_index
                        + (*wrapped_last_processed_pos as usize & mask as usize)]
                        == self.ringbuffer_.data_mo.slice()[self.ringbuffer_.buffer_index
                            + (((*wrapped_last_processed_pos as usize)
                                .wrapping_sub(cmd_dist as usize))
                                & mask as usize)])
                {
                    last_command.copy_len_ += 1;
                    (*bytes) -= 1;
                    (*wrapped_last_processed_pos) += 1;
                }
            }
            /* The copy length is at most the metablock size, and thus expressible. */
            get_length_code(
                last_command.insert_len_ as usize,
                ((last_command.copy_len_ & 0x01ff_ffff) as i32
                    + (last_command.copy_len_ >> 25) as i32) as usize,
                (last_command.dist_prefix_ & 0x03ff) == 0,
                &mut last_command.cmd_prefix_,
            );
        }
    }
}

fn RingBufferInit<AllocU8: alloc::Allocator<u8>>() -> RingBuffer<AllocU8> {
    RingBuffer {
        size_: 0,
        mask_: 0, // 0xff??
        tail_size_: 0,
        total_size_: 0,

        cur_size_: 0,
        pos_: 0,
        data_mo: AllocU8::AllocatedMemory::default(),
        buffer_index: 0usize,
    }
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    pub fn new(m8: Alloc) -> Self {
        let cache: [i32; 16] = [4, 11, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        Self {
            params: BrotliEncoderInitParams(),
            input_pos_: 0,
            num_commands_: 0,
            num_literals_: 0,
            last_insert_len_: 0,
            last_flush_pos_: 0,
            last_processed_pos_: 0,
            prev_byte_: 0,
            prev_byte2_: 0,
            storage_size_: 0,
            storage_: alloc_default::<u8, Alloc>(),
            hasher_: UnionHasher::<Alloc>::default(),
            large_table_: alloc_default::<i32, Alloc>(),
            //    large_table_size_: 0,
            cmd_code_numbits_: 0,
            command_buf_: alloc_default::<u32, Alloc>(),
            literal_buf_: alloc_default::<u8, Alloc>(),
            next_out_: NextOut::None,
            available_out_: 0,
            total_out_: 0,
            is_first_mb: IsFirst::NothingWritten,
            stream_state_: BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING,
            is_last_block_emitted_: false,
            is_initialized_: false,
            ringbuffer_: RingBufferInit(),
            commands_: alloc_default::<Command, Alloc>(),
            cmd_alloc_size_: 0,
            dist_cache_: cache,
            saved_dist_cache_: [cache[0], cache[1], cache[2], cache[3]],
            cmd_bits_: [0; 128],
            cmd_depths_: [0; 128],
            last_bytes_: 0,
            last_bytes_bits_: 0,
            cmd_code_: [0; 512],
            m8,
            remaining_metadata_bytes_: 0,
            small_table_: [0; 1024],
            tiny_buf_: [0; 16],
            literal_scratch_space: HistogramLiteral::make_nnz_storage(),
            command_scratch_space: HistogramCommand::make_nnz_storage(),
            distance_scratch_space: HistogramDistance::make_nnz_storage(),
            recoder_state: RecoderState::new(),
            custom_dictionary: false,
            custom_dictionary_size: None,
        }
    }
}

fn RingBufferFree<AllocU8: alloc::Allocator<u8>>(m: &mut AllocU8, rb: &mut RingBuffer<AllocU8>) {
    m.free_cell(core::mem::take(&mut rb.data_mo));
}
fn DestroyHasher<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m16: &mut Alloc,
    handle: &mut UnionHasher<Alloc>,
) {
    handle.free(m16);
}
/*
fn DestroyHasher<AllocU16:alloc::Allocator<u16>, AllocU32:alloc::Allocator<u32>>(
m16: &mut AllocU16, m32:&mut AllocU32, handle: &mut UnionHasher<AllocU16, AllocU32>){
  match handle {
    &mut UnionHasher::H2(ref mut hasher) => {
        m32.free_cell(core::mem::replace(&mut hasher.buckets_.buckets_, alloc_default::<u32, Alloc>()));
    }
    &mut UnionHasher::H3(ref mut hasher) => {
        m32.free_cell(core::mem::replace(&mut hasher.buckets_.buckets_, alloc_default::<u32, Alloc>()));
    }
    &mut UnionHasher::H4(ref mut hasher) => {
        m32.free_cell(core::mem::replace(&mut hasher.buckets_.buckets_, alloc_default::<u32, Alloc>()));
    }
    &mut UnionHasher::H54(ref mut hasher) => {
        m32.free_cell(core::mem::replace(&mut hasher.buckets_.buckets_, alloc_default::<u32, Alloc>()));
    }
    &mut UnionHasher::H5(ref mut hasher) => {
      m16.free_cell(core::mem::replace(&mut hasher.num, AllocU16::AllocatedMemory::default()));
      m32.free_cell(core::mem::replace(&mut hasher.buckets, alloc_default::<u32, Alloc>()));
    }
    &mut UnionHasher::H6(ref mut hasher) => {
      m16.free_cell(core::mem::replace(&mut hasher.num, AllocU16::AllocatedMemory::default()));
      m32.free_cell(core::mem::replace(&mut hasher.buckets, alloc_default::<u32, Alloc>()));
    }
    &mut UnionHasher::H9(ref mut hasher) => {
      m16.free_cell(core::mem::replace(&mut hasher.num_, AllocU16::AllocatedMemory::default()));
      m32.free_cell(core::mem::replace(&mut hasher.buckets_, alloc_default::<u32, Alloc>()));
    }
    _ => {}
  }
  *handle = UnionHasher::<AllocU16, AllocU32>::default();
}
*/

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn cleanup(&mut self) {
        <Alloc as Allocator<u8>>::free_cell(&mut self.m8, core::mem::take(&mut self.storage_));
        <Alloc as Allocator<Command>>::free_cell(
            &mut self.m8,
            core::mem::take(&mut self.commands_),
        );
        RingBufferFree(&mut self.m8, &mut self.ringbuffer_);
        DestroyHasher(&mut self.m8, &mut self.hasher_);
        <Alloc as Allocator<i32>>::free_cell(&mut self.m8, core::mem::take(&mut self.large_table_));
        <Alloc as Allocator<u32>>::free_cell(&mut self.m8, core::mem::take(&mut self.command_buf_));
        <Alloc as Allocator<u8>>::free_cell(&mut self.m8, core::mem::take(&mut self.literal_buf_));
    }
}

// TODO: use drop trait instead
// impl<Alloc: BrotliAlloc> Drop for BrotliEncoderStateStruct<Alloc> {
//     fn drop(&mut self) {
//         self.cleanup()
//     }
// }
pub fn BrotliEncoderDestroyInstance<Alloc: BrotliAlloc>(s: &mut BrotliEncoderStateStruct<Alloc>) {
    s.cleanup()
}

#[cfg(not(feature = "disallow_large_window_size"))]
fn check_large_window_ok() -> bool {
    true
}
#[cfg(feature = "disallow_large_window_size")]
fn check_large_window_ok() -> bool {
    false
}

pub fn SanitizeParams(params: &mut BrotliEncoderParams) {
    params.quality = min(11i32, max(0i32, params.quality));
    if params.lgwin < 10i32 {
        params.lgwin = 10i32;
    } else if params.lgwin > 24i32 {
        if params.large_window && check_large_window_ok() {
            if params.lgwin > 30i32 {
                params.lgwin = 30i32;
            }
        } else {
            params.lgwin = 24i32;
        }
    }
    if params.catable {
        params.appendable = true;
    }
}

fn ComputeLgBlock(params: &BrotliEncoderParams) -> i32 {
    let mut lgblock: i32 = params.lgblock;
    if params.quality == 0i32 || params.quality == 1i32 {
        lgblock = params.lgwin;
    } else if params.quality < 4i32 {
        lgblock = 14i32;
    } else if lgblock == 0i32 {
        lgblock = 16i32;
        if params.quality >= 9i32 && (params.lgwin > lgblock) {
            lgblock = min(18i32, params.lgwin);
        }
    } else {
        lgblock = min(24i32, max(16i32, lgblock));
    }
    lgblock
}

fn ComputeRbBits(params: &BrotliEncoderParams) -> i32 {
    1i32 + max(params.lgwin, params.lgblock)
}

fn RingBufferSetup<AllocU8: alloc::Allocator<u8>>(
    params: &BrotliEncoderParams,
    rb: &mut RingBuffer<AllocU8>,
) {
    let window_bits: i32 = ComputeRbBits(params);
    let tail_bits: i32 = params.lgblock;
    rb.size_ = 1u32 << window_bits;
    rb.mask_ = (1u32 << window_bits).wrapping_sub(1);
    rb.tail_size_ = 1u32 << tail_bits;
    rb.total_size_ = rb.size_.wrapping_add(rb.tail_size_);
}

fn EncodeWindowBits(
    lgwin: i32,
    large_window: bool,
    last_bytes: &mut u16,
    last_bytes_bits: &mut u8,
) {
    if large_window {
        *last_bytes = (((lgwin & 0x3F) << 8) | 0x11) as u16;
        *last_bytes_bits = 14;
    } else if lgwin == 16i32 {
        *last_bytes = 0u16;
        *last_bytes_bits = 1u8;
    } else if lgwin == 17i32 {
        *last_bytes = 1u16;
        *last_bytes_bits = 7u8;
    } else if lgwin > 17i32 {
        *last_bytes = ((lgwin - 17i32) << 1 | 1i32) as u16;
        *last_bytes_bits = 4u8;
    } else {
        *last_bytes = ((lgwin - 8i32) << 4 | 1i32) as u16;
        *last_bytes_bits = 7u8;
    }
}

fn InitCommandPrefixCodes(
    cmd_depths: &mut [u8],
    cmd_bits: &mut [u16],
    cmd_code: &mut [u8],
    cmd_code_numbits: &mut usize,
) {
    static kDefaultCommandDepths: [u8; 128] = [
        0, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6,
        7, 7, 7, 7, 10, 10, 10, 10, 10, 10, 0, 4, 4, 5, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6,
        6, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 10, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0,
    ];
    static kDefaultCommandBits: [u16; 128] = [
        0, 0, 8, 9, 3, 35, 7, 71, 39, 103, 23, 47, 175, 111, 239, 31, 0, 0, 0, 4, 12, 2, 10, 6, 13,
        29, 11, 43, 27, 59, 87, 55, 15, 79, 319, 831, 191, 703, 447, 959, 0, 14, 1, 25, 5, 21, 19,
        51, 119, 159, 95, 223, 479, 991, 63, 575, 127, 639, 383, 895, 255, 767, 511, 1023, 14, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 59, 7, 39, 23, 55, 30, 1, 17, 9, 25, 5, 0, 8,
        4, 12, 2, 10, 6, 21, 13, 29, 3, 19, 11, 15, 47, 31, 95, 63, 127, 255, 767, 2815, 1791,
        3839, 511, 2559, 1535, 3583, 1023, 3071, 2047, 4095, 0, 0, 0, 0,
    ];
    static kDefaultCommandCode: [u8; 57] = [
        0xff, 0x77, 0xd5, 0xbf, 0xe7, 0xde, 0xea, 0x9e, 0x51, 0x5d, 0xde, 0xc6, 0x70, 0x57, 0xbc,
        0x58, 0x58, 0x58, 0xd8, 0xd8, 0x58, 0xd5, 0xcb, 0x8c, 0xea, 0xe0, 0xc3, 0x87, 0x1f, 0x83,
        0xc1, 0x60, 0x1c, 0x67, 0xb2, 0xaa, 0x6, 0x83, 0xc1, 0x60, 0x30, 0x18, 0xcc, 0xa1, 0xce,
        0x88, 0x54, 0x94, 0x46, 0xe1, 0xb0, 0xd0, 0x4e, 0xb2, 0xf7, 0x4, 0x0,
    ];
    static kDefaultCommandCodeNumBits: usize = 448usize;
    cmd_depths[..].clone_from_slice(&kDefaultCommandDepths[..]);
    cmd_bits[..].clone_from_slice(&kDefaultCommandBits[..]);
    cmd_code[..kDefaultCommandCode.len()].clone_from_slice(&kDefaultCommandCode[..]);
    *cmd_code_numbits = kDefaultCommandCodeNumBits;
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn ensure_initialized(&mut self) -> bool {
        if self.is_initialized_ {
            return true;
        }
        SanitizeParams(&mut self.params);
        self.params.lgblock = ComputeLgBlock(&mut self.params);
        ChooseDistanceParams(&mut self.params);
        self.remaining_metadata_bytes_ = u32::MAX;
        RingBufferSetup(&mut self.params, &mut self.ringbuffer_);
        {
            let mut lgwin: i32 = self.params.lgwin;
            if self.params.quality == 0i32 || self.params.quality == 1i32 {
                lgwin = max(lgwin, 18i32);
            }
            EncodeWindowBits(
                lgwin,
                self.params.large_window,
                &mut self.last_bytes_,
                &mut self.last_bytes_bits_,
            );
        }
        if self.params.quality == 0i32 {
            InitCommandPrefixCodes(
                &mut self.cmd_depths_[..],
                &mut self.cmd_bits_[..],
                &mut self.cmd_code_[..],
                &mut self.cmd_code_numbits_,
            );
        }
        if self.params.catable {
            // if we want to properly concatenate, then we need to ignore any distances
            // this value 0x7ffffff0 was chosen to be larger than max_distance + gap
            // but small enough so that +/-3 will not overflow (due to distance modifications)
            for item in self.dist_cache_.iter_mut() {
                *item = 0x7ffffff0;
            }
            for item in self.saved_dist_cache_.iter_mut() {
                *item = 0x7ffffff0;
            }
        }
        self.is_initialized_ = true;
        true
    }
}

fn RingBufferInitBuffer<AllocU8: alloc::Allocator<u8>>(
    m: &mut AllocU8,
    buflen: u32,
    rb: &mut RingBuffer<AllocU8>,
) {
    static kSlackForEightByteHashingEverywhere: usize = 7usize;
    let mut new_data = m.alloc_cell(
        ((2u32).wrapping_add(buflen) as usize).wrapping_add(kSlackForEightByteHashingEverywhere),
    );
    if !rb.data_mo.slice().is_empty() {
        let lim: usize = ((2u32).wrapping_add(rb.cur_size_) as usize)
            .wrapping_add(kSlackForEightByteHashingEverywhere);
        new_data.slice_mut()[..lim].clone_from_slice(&rb.data_mo.slice()[..lim]);
        m.free_cell(core::mem::take(&mut rb.data_mo));
    }
    let _ = core::mem::replace(&mut rb.data_mo, new_data);
    rb.cur_size_ = buflen;
    rb.buffer_index = 2usize;
    rb.data_mo.slice_mut()[(rb.buffer_index.wrapping_sub(2))] = 0;
    rb.data_mo.slice_mut()[(rb.buffer_index.wrapping_sub(1))] = 0;
    for i in 0usize..kSlackForEightByteHashingEverywhere {
        rb.data_mo.slice_mut()[rb
            .buffer_index
            .wrapping_add(rb.cur_size_ as usize)
            .wrapping_add(i)] = 0;
    }
}

fn RingBufferWriteTail<AllocU8: alloc::Allocator<u8>>(
    bytes: &[u8],
    n: usize,
    rb: &mut RingBuffer<AllocU8>,
) {
    let masked_pos: usize = (rb.pos_ & rb.mask_) as usize;
    if masked_pos < rb.tail_size_ as usize {
        let p: usize = (rb.size_ as usize).wrapping_add(masked_pos);
        let begin = rb.buffer_index.wrapping_add(p);
        let lim = min(n, (rb.tail_size_ as usize).wrapping_sub(masked_pos));
        rb.data_mo.slice_mut()[begin..(begin + lim)].clone_from_slice(&bytes[..lim]);
    }
}

fn RingBufferWrite<AllocU8: alloc::Allocator<u8>>(
    m: &mut AllocU8,
    bytes: &[u8],
    n: usize,
    rb: &mut RingBuffer<AllocU8>,
) {
    if rb.pos_ == 0u32 && (n < rb.tail_size_ as usize) {
        rb.pos_ = n as u32;
        RingBufferInitBuffer(m, rb.pos_, rb);
        rb.data_mo.slice_mut()[rb.buffer_index..(rb.buffer_index + n)]
            .clone_from_slice(&bytes[..n]);
        return;
    }
    if rb.cur_size_ < rb.total_size_ {
        RingBufferInitBuffer(m, rb.total_size_, rb);
        rb.data_mo.slice_mut()[rb
            .buffer_index
            .wrapping_add(rb.size_ as usize)
            .wrapping_sub(2)] = 0u8;
        rb.data_mo.slice_mut()[rb
            .buffer_index
            .wrapping_add(rb.size_ as usize)
            .wrapping_sub(1)] = 0u8;
    }
    {
        let masked_pos: usize = (rb.pos_ & rb.mask_) as usize;
        RingBufferWriteTail(bytes, n, rb);
        if masked_pos.wrapping_add(n) <= rb.size_ as usize {
            // a single write fits
            let start = rb.buffer_index.wrapping_add(masked_pos);
            rb.data_mo.slice_mut()[start..(start + n)].clone_from_slice(&bytes[..n]);
        } else {
            {
                let start = rb.buffer_index.wrapping_add(masked_pos);
                let mid = min(n, (rb.total_size_ as usize).wrapping_sub(masked_pos));
                rb.data_mo.slice_mut()[start..(start + mid)].clone_from_slice(&bytes[..mid]);
            }
            let xstart = rb.buffer_index.wrapping_add(0);
            let size = n.wrapping_sub((rb.size_ as usize).wrapping_sub(masked_pos));
            let bytes_start = (rb.size_ as usize).wrapping_sub(masked_pos);
            rb.data_mo.slice_mut()[xstart..(xstart + size)]
                .clone_from_slice(&bytes[bytes_start..(bytes_start + size)]);
        }
    }
    let data_2 = rb.data_mo.slice()[rb
        .buffer_index
        .wrapping_add(rb.size_ as usize)
        .wrapping_sub(2)];
    rb.data_mo.slice_mut()[rb.buffer_index.wrapping_sub(2)] = data_2;
    let data_1 = rb.data_mo.slice()[rb
        .buffer_index
        .wrapping_add(rb.size_ as usize)
        .wrapping_sub(1)];
    rb.data_mo.slice_mut()[rb.buffer_index.wrapping_sub(1)] = data_1;
    rb.pos_ = rb.pos_.wrapping_add(n as u32);
    if rb.pos_ > 1u32 << 30 {
        rb.pos_ = rb.pos_ & (1u32 << 30).wrapping_sub(1) | 1u32 << 30;
    }
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    pub fn copy_input_to_ring_buffer(&mut self, input_size: usize, input_buffer: &[u8]) {
        if !self.ensure_initialized() {
            return;
        }
        RingBufferWrite(
            &mut self.m8,
            input_buffer,
            input_size,
            &mut self.ringbuffer_,
        );
        self.input_pos_ = self.input_pos_.wrapping_add(input_size as u64);
        if (self.ringbuffer_).pos_ <= (self.ringbuffer_).mask_ {
            let start = (self.ringbuffer_)
                .buffer_index
                .wrapping_add((self.ringbuffer_).pos_ as usize);
            for item in (self.ringbuffer_).data_mo.slice_mut()[start..(start + 7)].iter_mut() {
                *item = 0;
            }
        }
    }
}

fn ChooseHasher(params: &mut BrotliEncoderParams) {
    let hparams = &mut params.hasher;
    if params.quality >= 10 && !params.q9_5 {
        hparams.type_ = 10;
    } else if params.quality == 10 {
        // we are using quality 10 as a proxy for "9.5"
        hparams.type_ = 9;
        hparams.num_last_distances_to_check = H9_NUM_LAST_DISTANCES_TO_CHECK as i32;
        hparams.block_bits = H9_BLOCK_BITS as i32;
        hparams.bucket_bits = H9_BUCKET_BITS as i32;
        hparams.hash_len = 4;
    } else if params.quality == 9 {
        hparams.type_ = 9;
        hparams.num_last_distances_to_check = H9_NUM_LAST_DISTANCES_TO_CHECK as i32;
        hparams.block_bits = H9_BLOCK_BITS as i32;
        hparams.bucket_bits = H9_BUCKET_BITS as i32;
        hparams.hash_len = 4;
    } else if params.quality == 4 && (params.size_hint >= (1i32 << 20) as usize) {
        hparams.type_ = 54i32;
    } else if params.quality < 5 {
        hparams.type_ = params.quality;
    } else if params.lgwin <= 16 {
        hparams.type_ = if params.quality < 7 {
            40i32
        } else if params.quality < 9 {
            41i32
        } else {
            42i32
        };
    } else if ((params.q9_5 && params.size_hint > (1 << 20)) || params.size_hint > (1 << 22))
        && (params.lgwin >= 19i32)
    {
        hparams.type_ = 6i32;
        hparams.block_bits = min(params.quality - 1, 9);
        hparams.bucket_bits = 15i32;
        hparams.hash_len = 5i32;
        hparams.num_last_distances_to_check = if params.quality < 7 {
            4i32
        } else if params.quality < 9 {
            10i32
        } else {
            16i32
        };
    } else {
        hparams.type_ = 5i32;
        hparams.block_bits = min(params.quality - 1, 9);
        hparams.bucket_bits = if params.quality < 7 && params.size_hint <= (1 << 20) {
            14i32
        } else {
            15i32
        };
        hparams.num_last_distances_to_check = if params.quality < 7 {
            4i32
        } else if params.quality < 9 {
            10i32
        } else {
            16i32
        };
    }
}

fn InitializeH2<AllocU32: alloc::Allocator<u32>>(
    m32: &mut AllocU32,
    params: &BrotliEncoderParams,
) -> BasicHasher<H2Sub<AllocU32>> {
    BasicHasher {
        GetHasherCommon: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        buckets_: H2Sub {
            buckets_: m32.alloc_cell(65537 + 8),
        },
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
    }
}
fn InitializeH3<AllocU32: alloc::Allocator<u32>>(
    m32: &mut AllocU32,
    params: &BrotliEncoderParams,
) -> BasicHasher<H3Sub<AllocU32>> {
    BasicHasher {
        GetHasherCommon: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        buckets_: H3Sub {
            buckets_: m32.alloc_cell(65538 + 8),
        },
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
    }
}
fn InitializeH4<AllocU32: alloc::Allocator<u32>>(
    m32: &mut AllocU32,
    params: &BrotliEncoderParams,
) -> BasicHasher<H4Sub<AllocU32>> {
    BasicHasher {
        GetHasherCommon: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        buckets_: H4Sub {
            buckets_: m32.alloc_cell(131072 + 8),
        },
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
    }
}
fn InitializeH54<AllocU32: alloc::Allocator<u32>>(
    m32: &mut AllocU32,
    params: &BrotliEncoderParams,
) -> BasicHasher<H54Sub<AllocU32>> {
    BasicHasher {
        GetHasherCommon: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        buckets_: H54Sub {
            buckets_: m32.alloc_cell(1048580 + 8),
        },
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
    }
}

fn InitializeH9<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m16: &mut Alloc,
    params: &BrotliEncoderParams,
) -> H9<Alloc> {
    H9 {
        dict_search_stats_: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        num_: allocate::<u16, _>(m16, 1 << H9_BUCKET_BITS),
        buckets_: allocate::<u32, _>(m16, H9_BLOCK_SIZE << H9_BUCKET_BITS),
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
    }
}

fn InitializeH5<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m16: &mut Alloc,
    params: &BrotliEncoderParams,
) -> UnionHasher<Alloc> {
    let block_size = 1u64 << params.hasher.block_bits;
    let bucket_size = 1u64 << params.hasher.bucket_bits;
    let buckets: <Alloc as Allocator<u32>>::AllocatedMemory =
        allocate::<u32, _>(m16, (bucket_size * block_size) as usize);
    let num: <Alloc as Allocator<u16>>::AllocatedMemory =
        allocate::<u16, _>(m16, bucket_size as usize);

    if params.hasher.block_bits == (HQ5Sub {}).block_bits()
        && (1 << params.hasher.bucket_bits) == (HQ5Sub {}).bucket_size()
    {
        return UnionHasher::H5q5(AdvHasher {
            buckets,
            h9_opts: super::backward_references::H9Opts::new(&params.hasher),
            num,
            GetHasherCommon: Struct1 {
                params: params.hasher,
                is_prepared_: 1,
                dict_num_lookups: 0,
                dict_num_matches: 0,
            },
            specialization: HQ5Sub {},
        });
    }
    if params.hasher.block_bits == (HQ7Sub {}).block_bits()
        && (1 << params.hasher.bucket_bits) == (HQ7Sub {}).bucket_size()
    {
        return UnionHasher::H5q7(AdvHasher {
            buckets,
            h9_opts: super::backward_references::H9Opts::new(&params.hasher),
            num,
            GetHasherCommon: Struct1 {
                params: params.hasher,
                is_prepared_: 1,
                dict_num_lookups: 0,
                dict_num_matches: 0,
            },
            specialization: HQ7Sub {},
        });
    }
    UnionHasher::H5(AdvHasher {
        buckets,
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
        num,
        GetHasherCommon: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: H5Sub {
            hash_shift_: 32i32 - params.hasher.bucket_bits,
            bucket_size_: bucket_size as u32,
            block_bits_: params.hasher.block_bits,
            block_mask_: block_size.wrapping_sub(1) as u32,
        },
    })
}
fn InitializeH6<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m16: &mut Alloc,
    params: &BrotliEncoderParams,
) -> UnionHasher<Alloc> {
    let block_size = 1u64 << params.hasher.block_bits;
    let bucket_size = 1u64 << params.hasher.bucket_bits;
    let buckets: <Alloc as Allocator<u32>>::AllocatedMemory =
        allocate::<u32, _>(m16, (bucket_size * block_size) as usize);
    let num: <Alloc as Allocator<u16>>::AllocatedMemory =
        allocate::<u16, _>(m16, bucket_size as usize);
    UnionHasher::H6(AdvHasher {
        buckets,
        num,
        h9_opts: super::backward_references::H9Opts::new(&params.hasher),
        GetHasherCommon: Struct1 {
            params: params.hasher,
            is_prepared_: 1,
            dict_num_lookups: 0,
            dict_num_matches: 0,
        },
        specialization: H6Sub {
            bucket_size_: 1u32 << params.hasher.bucket_bits,
            block_bits_: params.hasher.block_bits,
            block_mask_: block_size.wrapping_sub(1) as u32,
            hash_mask: 0xffffffffffffffffu64 >> (64i32 - 8i32 * params.hasher.hash_len),
            hash_shift_: 64i32 - params.hasher.bucket_bits,
        },
    })
}

fn BrotliMakeHasher<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m: &mut Alloc,
    params: &BrotliEncoderParams,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
) -> UnionHasher<Alloc> {
    let hasher_type: i32 = params.hasher.type_;
    if hasher_type == 2i32 {
        return UnionHasher::H2(InitializeH2(m, params));
    }
    if hasher_type == 3i32 {
        return UnionHasher::H3(InitializeH3(m, params));
    }
    if hasher_type == 4i32 {
        return UnionHasher::H4(InitializeH4(m, params));
    }
    if hasher_type == 5i32 {
        return InitializeH5(m, params);
    }
    if hasher_type == 6i32 {
        return InitializeH6(m, params);
    }
    if hasher_type == 9i32 {
        return UnionHasher::H9(InitializeH9(m, params));
    }
    /*
        if hasher_type == 40i32 {
          return InitializeH40(params);
        }
        if hasher_type == 41i32 {
          return InitializeH41(params);
        }
        if hasher_type == 42i32 {
          return InitializeH42(params);
        }
    */
    if hasher_type == 54i32 {
        return UnionHasher::H54(InitializeH54(m, params));
    }
    if hasher_type == 10i32 {
        return UnionHasher::H10(InitializeH10(m, false, params, ringbuffer_break, 0));
    }
    // since we don't support all of these, fall back to something sane
    InitializeH6(m, params)

    //  return UnionHasher::Uninit;
}
fn HasherReset<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(t: &mut UnionHasher<Alloc>) {
    match t {
        &mut UnionHasher::Uninit => {}
        _ => (t.GetHasherCommon()).is_prepared_ = 0i32,
    };
}

pub(crate) fn hasher_setup<Alloc: Allocator<u16> + Allocator<u32>>(
    m16: &mut Alloc,
    handle: &mut UnionHasher<Alloc>,
    params: &mut BrotliEncoderParams,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    data: &[u8],
    position: usize,
    input_size: usize,
    is_last: bool,
) {
    let one_shot = position == 0 && is_last;
    let is_uninit = match (handle) {
        &mut UnionHasher::Uninit => true,
        _ => false,
    };
    if is_uninit {
        //let alloc_size: usize;
        ChooseHasher(&mut (*params));
        //alloc_size = HasherSize(params, one_shot, input_size);
        //xself = BrotliAllocate(m, alloc_size.wrapping_mul(::core::mem::size_of::<u8>()))
        *handle = BrotliMakeHasher(m16, params, ringbuffer_break);
        handle.GetHasherCommon().params = params.hasher;
        HasherReset(handle); // this sets everything to zero, unlike in C
        handle.GetHasherCommon().is_prepared_ = 1;
    } else {
        match handle.Prepare(one_shot, input_size, data) {
            HowPrepared::ALREADY_PREPARED => {}
            HowPrepared::NEWLY_PREPARED => {
                if position == 0usize {
                    let common = handle.GetHasherCommon();
                    common.dict_num_lookups = 0usize;
                    common.dict_num_matches = 0usize;
                }
            }
        }
    }
}

fn HasherPrependCustomDictionary<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m: &mut Alloc,
    handle: &mut UnionHasher<Alloc>,
    params: &mut BrotliEncoderParams,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    size: usize,
    dict: &[u8],
) {
    hasher_setup(
        m,
        handle,
        params,
        ringbuffer_break,
        dict,
        0usize,
        size,
        false,
    );
    match handle {
        &mut UnionHasher::H2(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H3(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H4(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H5(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H5q7(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H5q5(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H6(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H9(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H54(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::H10(ref mut hasher) => StoreLookaheadThenStore(hasher, size, dict),
        &mut UnionHasher::Uninit => panic!("Uninitialized"),
    }
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    pub fn set_custom_dictionary(&mut self, size: usize, dict: &[u8]) {
        self.set_custom_dictionary_with_optional_precomputed_hasher(
            size,
            dict,
            UnionHasher::Uninit,
            false,
        )
    }

    pub fn set_custom_dictionary_with_optional_precomputed_hasher(
        &mut self,
        size: usize,
        mut dict: &[u8],
        opt_hasher: UnionHasher<Alloc>,
        is_multithreading_file_continue: bool,
    ) {
        self.params.use_dictionary = false;

        self.prev_byte_ = 0;
        self.prev_byte2_ = 0;
        if is_multithreading_file_continue {
            if size > 0 {
                self.prev_byte_ = dict[size.wrapping_sub(1)];
            }
            if size > 1 {
                self.prev_byte2_ = dict[size.wrapping_sub(2)];
            }
        }

        let has_optional_hasher = if let UnionHasher::Uninit = opt_hasher {
            false
        } else {
            true
        };
        let max_dict_size: usize = (1usize << self.params.lgwin).wrapping_sub(16);
        self.hasher_ = opt_hasher;
        let mut dict_size: usize = size;
        if !self.ensure_initialized() {
            return;
        }
        if dict_size == 0 || self.params.quality == 0 || self.params.quality == 1 || size <= 1 {
            self.params.catable = true; // don't risk a too-short dictionary
            self.params.appendable = true; // don't risk a too-short dictionary
            return;
        }
        self.custom_dictionary = true;
        if size > max_dict_size {
            dict = &dict[size.wrapping_sub(max_dict_size)..];
            dict_size = max_dict_size;
        }
        self.custom_dictionary_size = core::num::NonZeroUsize::new(dict_size);
        self.copy_input_to_ring_buffer(dict_size, dict);
        self.last_flush_pos_ = dict_size as u64;
        self.last_processed_pos_ = dict_size as u64;
        let m16 = &mut self.m8;
        if cfg!(debug_assertions) || !has_optional_hasher {
            let mut orig_hasher = UnionHasher::Uninit;
            if has_optional_hasher {
                orig_hasher = core::mem::replace(&mut self.hasher_, UnionHasher::Uninit);
            }
            HasherPrependCustomDictionary(
                m16,
                &mut self.hasher_,
                &mut self.params,
                self.custom_dictionary_size,
                dict_size,
                dict,
            );
            if has_optional_hasher {
                debug_assert!(orig_hasher == self.hasher_);
                DestroyHasher(m16, &mut orig_hasher);
            }
        }
    }
}

pub fn BrotliEncoderMaxCompressedSizeMulti(input_size: usize, num_threads: usize) -> usize {
    BrotliEncoderMaxCompressedSize(input_size) + num_threads * 8
}

pub fn BrotliEncoderMaxCompressedSize(input_size: usize) -> usize {
    let magic_size = 16usize;
    let num_large_blocks: usize = input_size >> 14;
    let tail: usize = input_size.wrapping_sub(num_large_blocks << 24);
    let tail_overhead: usize = (if tail > (1i32 << 20) as usize {
        4i32
    } else {
        3i32
    }) as usize;
    let overhead: usize = (2usize)
        .wrapping_add((4usize).wrapping_mul(num_large_blocks))
        .wrapping_add(tail_overhead)
        .wrapping_add(1);
    let result: usize = input_size.wrapping_add(overhead);
    if input_size == 0usize {
        return 1 + magic_size;
    }
    if result < input_size {
        0usize
    } else {
        result + magic_size
    }
}

fn InitOrStitchToPreviousBlock<Alloc: alloc::Allocator<u16> + alloc::Allocator<u32>>(
    m: &mut Alloc,
    handle: &mut UnionHasher<Alloc>,
    data: &[u8],
    mask: usize,
    ringbuffer_break: Option<core::num::NonZeroUsize>,
    params: &mut BrotliEncoderParams,
    position: usize,
    input_size: usize,
    is_last: bool,
) {
    hasher_setup(
        m,
        handle,
        params,
        ringbuffer_break,
        data,
        position,
        input_size,
        is_last,
    );
    handle.StitchToPreviousBlock(input_size, position, data, mask);
}

fn should_compress(
    data: &[u8],
    mask: usize,
    last_flush_pos: u64,
    bytes: usize,
    num_literals: usize,
    num_commands: usize,
) -> bool {
    const K_SAMPLE_RATE: u32 = 13;
    const K_MIN_ENTROPY: floatX = 7.92;

    if num_commands < (bytes >> 8) + 2 && num_literals as floatX > 0.99 * bytes as floatX {
        let mut literal_histo = [0u32; 256];
        let bit_cost_threshold = (bytes as floatX) * K_MIN_ENTROPY / (K_SAMPLE_RATE as floatX);
        let t = bytes
            .wrapping_add(K_SAMPLE_RATE as usize)
            .wrapping_sub(1)
            .wrapping_div(K_SAMPLE_RATE as usize);
        let mut pos = last_flush_pos as u32;
        for _ in 0..t {
            let value = &mut literal_histo[data[pos as usize & mask] as usize];
            *value = value.wrapping_add(1);
            pos = pos.wrapping_add(K_SAMPLE_RATE);
        }
        if BitsEntropy(&literal_histo[..], 256) > bit_cost_threshold {
            return false;
        }
    }
    true
}

/* Chooses the literal context mode for a metablock */
fn ChooseContextMode(
    params: &BrotliEncoderParams,
    data: &[u8],
    pos: usize,
    mask: usize,
    length: usize,
) -> ContextType {
    /* We only do the computation for the option of something else than
    CONTEXT_UTF8 for the highest qualities */
    match params.mode {
        BrotliEncoderMode::BROTLI_FORCE_LSB_PRIOR => return ContextType::CONTEXT_LSB6,
        BrotliEncoderMode::BROTLI_FORCE_MSB_PRIOR => return ContextType::CONTEXT_MSB6,
        BrotliEncoderMode::BROTLI_FORCE_UTF8_PRIOR => return ContextType::CONTEXT_UTF8,
        BrotliEncoderMode::BROTLI_FORCE_SIGNED_PRIOR => return ContextType::CONTEXT_SIGNED,
        _ => {}
    }
    if (params.quality >= 10 && !is_mostly_utf8(data, pos, mask, length, kMinUTF8Ratio)) {
        return ContextType::CONTEXT_SIGNED;
    }
    ContextType::CONTEXT_UTF8
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum BrotliEncoderOperation {
    BROTLI_OPERATION_PROCESS = 0,
    BROTLI_OPERATION_FLUSH = 1,
    BROTLI_OPERATION_FINISH = 2,
    BROTLI_OPERATION_EMIT_METADATA = 3,
}

#[allow(unused)]
fn MakeUncompressedStream(input: &[u8], input_size: usize, output: &mut [u8]) -> usize {
    let mut size: usize = input_size;
    let mut result: usize = 0usize;
    let mut offset: usize = 0usize;
    if input_size == 0usize {
        output[0] = 6u8;
        return 1;
    }
    output[result] = 0x21u8;
    result = result.wrapping_add(1);
    output[result] = 0x3u8;
    result = result.wrapping_add(1);
    while size > 0usize {
        let mut nibbles: u32 = 0u32;

        let chunk_size: u32 = if size > (1u32 << 24) as usize {
            1u32 << 24
        } else {
            size as u32
        };
        if chunk_size > 1u32 << 16 {
            nibbles = if chunk_size > 1u32 << 20 { 2i32 } else { 1i32 } as u32;
        }
        let bits: u32 = nibbles << 1
            | chunk_size.wrapping_sub(1) << 3
            | 1u32 << (19u32).wrapping_add((4u32).wrapping_mul(nibbles));
        output[result] = bits as u8;
        result = result.wrapping_add(1);
        output[result] = (bits >> 8) as u8;
        result = result.wrapping_add(1);
        output[result] = (bits >> 16) as u8;
        result = result.wrapping_add(1);
        if nibbles == 2u32 {
            output[result] = (bits >> 24) as u8;
            result = result.wrapping_add(1);
        }
        output[result..(result + chunk_size as usize)]
            .clone_from_slice(&input[offset..(offset + chunk_size as usize)]);
        result = result.wrapping_add(chunk_size as usize);
        offset = offset.wrapping_add(chunk_size as usize);
        size = size.wrapping_sub(chunk_size as usize);
    }
    output[result] = 3u8;
    result = result.wrapping_add(1);
    result
}

#[cfg_attr(not(feature = "ffi-api"), cfg(test))]
pub(crate) fn encoder_compress<
    Alloc: BrotliAlloc,
    MetablockCallback: FnMut(
        &mut interface::PredictionModeContextMap<InputReferenceMut>,
        &mut [interface::StaticCommand],
        interface::InputPair,
        &mut Alloc,
    ),
>(
    empty_m8: Alloc,
    m8: &mut Alloc,
    mut quality: i32,
    lgwin: i32,
    mode: BrotliEncoderMode,
    input_size: usize,
    input_buffer: &[u8],
    encoded_size: &mut usize,
    encoded_buffer: &mut [u8],
    metablock_callback: &mut MetablockCallback,
) -> bool {
    let out_size: usize = *encoded_size;
    let input_start = input_buffer;
    let output_start = encoded_buffer;
    let max_out_size: usize = BrotliEncoderMaxCompressedSize(input_size);
    if out_size == 0 {
        return false;
    }
    if input_size == 0 {
        *encoded_size = 1;
        output_start[0] = 6;
        return true;
    }
    let mut is_fallback = false;
    let mut is_9_5 = false;
    if quality == 10 {
        quality = 9;
        is_9_5 = true;
    }
    if !is_fallback {
        let mut s_orig = BrotliEncoderStateStruct::new(core::mem::replace(m8, empty_m8));
        if is_9_5 {
            let mut params = BrotliEncoderParams::default();
            params.q9_5 = true;
            params.quality = 10;
            ChooseHasher(&mut params);
            s_orig.hasher_ = BrotliMakeHasher(m8, &params, None /*no custom dict */);
        }
        let mut result: bool;
        {
            let s = &mut s_orig;
            let mut available_in: usize = input_size;
            let next_in_array: &[u8] = input_buffer;
            let mut next_in_offset: usize = 0;
            let mut available_out: usize = *encoded_size;
            let next_out_array: &mut [u8] = output_start;
            let mut next_out_offset: usize = 0;
            let mut total_out = Some(0);
            s.set_parameter(BrotliEncoderParameter::BROTLI_PARAM_QUALITY, quality as u32);
            s.set_parameter(BrotliEncoderParameter::BROTLI_PARAM_LGWIN, lgwin as u32);
            s.set_parameter(BrotliEncoderParameter::BROTLI_PARAM_MODE, mode as u32);
            s.set_parameter(
                BrotliEncoderParameter::BROTLI_PARAM_SIZE_HINT,
                input_size as u32,
            );
            if lgwin > BROTLI_MAX_WINDOW_BITS as i32 {
                s.set_parameter(BrotliEncoderParameter::BROTLI_PARAM_LARGE_WINDOW, 1);
            }
            result = s.compress_stream(
                BrotliEncoderOperation::BROTLI_OPERATION_FINISH,
                &mut available_in,
                next_in_array,
                &mut next_in_offset,
                &mut available_out,
                next_out_array,
                &mut next_out_offset,
                &mut total_out,
                metablock_callback,
            );
            if !s.is_finished() {
                result = false;
            }

            *encoded_size = total_out.unwrap();
            BrotliEncoderDestroyInstance(s);
        }
        let _ = core::mem::replace(m8, s_orig.m8);
        if !result || max_out_size != 0 && (*encoded_size > max_out_size) {
            is_fallback = true;
        } else {
            return true;
        }
    }
    assert_ne!(is_fallback, false);
    *encoded_size = 0;
    if max_out_size == 0 {
        return false;
    }
    if out_size >= max_out_size {
        *encoded_size = MakeUncompressedStream(input_start, input_size, output_start);
        return true;
    }
    false
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn inject_byte_padding_block(&mut self) {
        let mut seal: u32 = self.last_bytes_ as u32;
        let mut seal_bits: usize = self.last_bytes_bits_ as usize;
        let destination: &mut [u8];
        self.last_bytes_ = 0;
        self.last_bytes_bits_ = 0;
        seal |= 0x6u32 << seal_bits;

        seal_bits = seal_bits.wrapping_add(6);
        if !IsNextOutNull(&self.next_out_) {
            destination = &mut GetNextOut!(*self)[self.available_out_..];
        } else {
            destination = &mut self.tiny_buf_[..];
            self.next_out_ = NextOut::TinyBuf(0);
        }
        destination[0] = seal as u8;
        if seal_bits > 8usize {
            destination[1] = (seal >> 8) as u8;
        }
        if seal_bits > 16usize {
            destination[2] = (seal >> 16) as u8;
        }
        self.available_out_ = self
            .available_out_
            .wrapping_add(seal_bits.wrapping_add(7) >> 3);
    }

    fn inject_flush_or_push_output(
        &mut self,
        available_out: &mut usize,
        next_out_array: &mut [u8],
        next_out_offset: &mut usize,
        total_out: &mut Option<usize>,
    ) -> bool {
        if self.stream_state_ as i32
            == BrotliEncoderStreamState::BROTLI_STREAM_FLUSH_REQUESTED as i32
            && (self.last_bytes_bits_ as i32 != 0i32)
        {
            self.inject_byte_padding_block();
            return true;
        }
        if self.available_out_ != 0usize && (*available_out != 0usize) {
            let copy_output_size: usize = min(self.available_out_, *available_out);
            (*next_out_array)[(*next_out_offset)..(*next_out_offset + copy_output_size)]
                .clone_from_slice(&GetNextOut!(self)[..copy_output_size]);
            //memcpy(*next_out, s.next_out_, copy_output_size);
            *next_out_offset = next_out_offset.wrapping_add(copy_output_size);
            *available_out = available_out.wrapping_sub(copy_output_size);
            self.next_out_ = NextOutIncrement(&self.next_out_, (copy_output_size as i32));
            self.available_out_ = self.available_out_.wrapping_sub(copy_output_size);
            self.total_out_ = self.total_out_.wrapping_add(copy_output_size as u64);
            if let &mut Some(ref mut total_out_inner) = total_out {
                *total_out_inner = self.total_out_ as usize;
            }
            return true;
        }
        false
    }

    fn unprocessed_input_size(&self) -> u64 {
        self.input_pos_.wrapping_sub(self.last_processed_pos_)
    }

    fn update_size_hint(&mut self, available_in: usize) {
        if self.params.size_hint == 0usize {
            let delta: u64 = self.unprocessed_input_size();
            let tail: u64 = available_in as u64;
            let limit: u32 = 1u32 << 30;
            let total: u32;
            if delta >= u64::from(limit)
                || tail >= u64::from(limit)
                || delta.wrapping_add(tail) >= u64::from(limit)
            {
                total = limit;
            } else {
                total = delta.wrapping_add(tail) as u32;
            }
            self.params.size_hint = total as usize;
        }
    }
}

fn WrapPosition(position: u64) -> u32 {
    let mut result: u32 = position as u32;
    let gb: u64 = position >> 30;
    if gb > 2 {
        result = result & (1u32 << 30).wrapping_sub(1)
            | ((gb.wrapping_sub(1) & 1) as u32).wrapping_add(1) << 30;
    }
    result
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn get_brotli_storage(&mut self, size: usize) {
        if self.storage_size_ < size {
            <Alloc as Allocator<u8>>::free_cell(&mut self.m8, core::mem::take(&mut self.storage_));
            self.storage_ = allocate::<u8, _>(&mut self.m8, size);
            self.storage_size_ = size;
        }
    }
}

fn MaxHashTableSize(quality: i32) -> usize {
    (if quality == 0i32 {
        1i32 << 15
    } else {
        1i32 << 17
    }) as usize
}

fn HashTableSize(max_table_size: usize, input_size: usize) -> usize {
    let mut htsize: usize = 256usize;
    while htsize < max_table_size && (htsize < input_size) {
        htsize <<= 1i32;
    }
    htsize
}

macro_rules! GetHashTable {
    ($s : expr, $quality: expr, $input_size : expr, $table_size : expr) => {
        GetHashTableInternal(
            &mut $s.m8,
            &mut $s.small_table_,
            &mut $s.large_table_,
            $quality,
            $input_size,
            $table_size,
        )
    };
}
fn GetHashTableInternal<'a, AllocI32: alloc::Allocator<i32>>(
    mi32: &mut AllocI32,
    small_table_: &'a mut [i32; 1024],
    large_table_: &'a mut AllocI32::AllocatedMemory,
    quality: i32,
    input_size: usize,
    table_size: &mut usize,
) -> &'a mut [i32] {
    let max_table_size: usize = MaxHashTableSize(quality);
    let mut htsize: usize = HashTableSize(max_table_size, input_size);
    let table: &mut [i32];
    if quality == 0i32 && htsize & 0xaaaaausize == 0usize {
        htsize <<= 1i32;
    }
    if htsize <= small_table_.len() {
        table = &mut small_table_[..];
    } else {
        if htsize > large_table_.slice().len() {
            //s.large_table_size_ = htsize;
            {
                mi32.free_cell(core::mem::take(large_table_));
            }
            *large_table_ = mi32.alloc_cell(htsize);
        }
        table = large_table_.slice_mut();
    }
    *table_size = htsize;
    for item in table[..htsize].iter_mut() {
        *item = 0;
    }
    table // FIXME: probably need a macro to do this without borrowing the whole EncoderStateStruct
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn update_last_processed_pos(&mut self) -> bool {
        let wrapped_last_processed_pos: u32 = WrapPosition(self.last_processed_pos_);
        let wrapped_input_pos: u32 = WrapPosition(self.input_pos_);
        self.last_processed_pos_ = self.input_pos_;
        wrapped_input_pos < wrapped_last_processed_pos
    }
}

fn MaxMetablockSize(params: &BrotliEncoderParams) -> usize {
    1 << min(ComputeRbBits(params), 24)
}

fn ChooseContextMap(
    quality: i32,
    bigram_histo: &mut [u32],
    num_literal_contexts: &mut usize,
    literal_context_map: &mut &[u32],
) {
    static kStaticContextMapContinuation: [u32; 64] = [
        1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    static kStaticContextMapSimpleUTF8: [u32; 64] = [
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    let mut monogram_histo = [0u32; 3];
    let mut two_prefix_histo = [0u32; 6];

    let mut i: usize;
    let mut entropy = [0.0 as floatX; 4];
    i = 0usize;
    while i < 9usize {
        {
            {
                let _rhs = bigram_histo[i];
                let _lhs = &mut monogram_histo[i.wrapping_rem(3)];
                *_lhs = (*_lhs).wrapping_add(_rhs);
            }
            {
                let _rhs = bigram_histo[i];
                let _lhs = &mut two_prefix_histo[i.wrapping_rem(6)];
                *_lhs = (*_lhs).wrapping_add(_rhs);
            }
        }
        i = i.wrapping_add(1);
    }
    entropy[1] = shannon_entropy(&monogram_histo[..], 3).0;
    entropy[2] =
        shannon_entropy(&two_prefix_histo[..], 3).0 + shannon_entropy(&two_prefix_histo[3..], 3).0;
    entropy[3] = 0.0;
    for i in 0usize..3usize {
        entropy[3] += shannon_entropy(&bigram_histo[(3usize).wrapping_mul(i)..], 3).0;
    }
    let total: usize = monogram_histo[0]
        .wrapping_add(monogram_histo[1])
        .wrapping_add(monogram_histo[2]) as usize;
    entropy[0] = 1.0 / (total as floatX);
    entropy[1] *= entropy[0];
    entropy[2] *= entropy[0];
    entropy[3] *= entropy[0];
    if quality < 7i32 {
        entropy[3] = entropy[1] * 10.0;
    }
    if entropy[1] - entropy[2] < 0.2 && entropy[1] - entropy[3] < 0.2 {
        *num_literal_contexts = 1;
    } else if entropy[2] - entropy[3] < 0.02 {
        *num_literal_contexts = 2usize;
        *literal_context_map = &kStaticContextMapSimpleUTF8[..];
    } else {
        *num_literal_contexts = 3usize;
        *literal_context_map = &kStaticContextMapContinuation[..];
    }
}

static kStaticContextMapComplexUTF8: [u32; 64] = [
    11, 11, 12, 12, /* 0 special */
    0, 0, 0, 0, /* 4 lf */
    1, 1, 9, 9, /* 8 space */
    2, 2, 2, 2, /* !, first after space/lf and after something else. */
    1, 1, 1, 1, /* " */
    8, 3, 3, 3, /* % */
    1, 1, 1, 1, /* ({[ */
    2, 2, 2, 2, /* }]) */
    8, 4, 4, 4, /* :; */
    8, 7, 4, 4, /* . */
    8, 0, 0, 0, /* > */
    3, 3, 3, 3, /* [0..9] */
    5, 5, 10, 5, /* [A-Z] */
    5, 5, 10, 5, 6, 6, 6, 6, /* [a-z] */
    6, 6, 6, 6,
];
/* Decide if we want to use a more complex static context map containing 13
context values, based on the entropy reduction of histograms over the
first 5 bits of literals. */
fn ShouldUseComplexStaticContextMap(
    input: &[u8],
    mut start_pos: usize,
    length: usize,
    mask: usize,
    quality: i32,
    size_hint: usize,
    num_literal_contexts: &mut usize,
    literal_context_map: &mut &[u32],
) -> bool {
    let _ = quality;
    //BROTLI_UNUSED(quality);
    /* Try the more complex static context map only for long data. */
    if (size_hint < (1 << 20)) {
        false
    } else {
        let end_pos = start_pos + length;
        /* To make entropy calculations faster and to fit on the stack, we collect
        histograms over the 5 most significant bits of literals. One histogram
        without context and 13 additional histograms for each context value. */
        let mut combined_histo: [u32; 32] = [0; 32];
        let mut context_histo: [[u32; 32]; 13] = [[0; 32]; 13];
        let mut total = 0u32;
        let mut entropy = [0.0 as floatX; 3];
        let utf8_lut = BROTLI_CONTEXT_LUT(ContextType::CONTEXT_UTF8);
        while start_pos + 64 <= end_pos {
            let stride_end_pos = start_pos + 64;
            let mut prev2 = input[start_pos & mask];
            let mut prev1 = input[(start_pos + 1) & mask];

            /* To make the analysis of the data faster we only examine 64 byte long
            strides at every 4kB intervals. */
            for pos in start_pos + 2..stride_end_pos {
                let literal = input[pos & mask];
                let context = kStaticContextMapComplexUTF8
                    [BROTLI_CONTEXT(prev1, prev2, utf8_lut) as usize]
                    as u8;
                total += 1;
                combined_histo[(literal >> 3) as usize] += 1;
                context_histo[context as usize][(literal >> 3) as usize] += 1;
                prev2 = prev1;
                prev1 = literal;
            }
            start_pos += 4096;
        }
        entropy[1] = shannon_entropy(&combined_histo[..], 32).0;
        entropy[2] = 0.0;
        for i in 0..13 {
            assert!(i < 13);
            entropy[2] += shannon_entropy(&context_histo[i][..], 32).0;
        }
        entropy[0] = 1.0 / (total as floatX);
        entropy[1] *= entropy[0];
        entropy[2] *= entropy[0];
        /* The triggering heuristics below were tuned by compressing the individual
        files of the silesia corpus. If we skip this kind of context modeling
        for not very well compressible input (i.e. entropy using context modeling
        is 60% of maximal entropy) or if expected savings by symbol are less
        than 0.2 bits, then in every case when it triggers, the final compression
        ratio is improved. Note however that this heuristics might be too strict
        for some cases and could be tuned further. */
        if (entropy[2] > 3.0 || entropy[1] - entropy[2] < 0.2) {
            false
        } else {
            *num_literal_contexts = 13;
            *literal_context_map = &kStaticContextMapComplexUTF8;
            true
        }
    }
}

fn DecideOverLiteralContextModeling(
    input: &[u8],
    mut start_pos: usize,
    length: usize,
    mask: usize,
    quality: i32,
    size_hint: usize,
    num_literal_contexts: &mut usize,
    literal_context_map: &mut &[u32],
) {
    if quality < 5i32 || length < 64usize {
    } else if ShouldUseComplexStaticContextMap(
        input,
        start_pos,
        length,
        mask,
        quality,
        size_hint,
        num_literal_contexts,
        literal_context_map,
    ) {
    } else {
        let end_pos: usize = start_pos.wrapping_add(length);
        let mut bigram_prefix_histo = [0u32; 9];
        while start_pos.wrapping_add(64) <= end_pos {
            {
                static lut: [i32; 4] = [0, 0, 1, 2];
                let stride_end_pos: usize = start_pos.wrapping_add(64);
                let mut prev: i32 = lut[(input[(start_pos & mask)] as i32 >> 6) as usize] * 3i32;
                let mut pos: usize;
                pos = start_pos.wrapping_add(1);
                while pos < stride_end_pos {
                    {
                        let literal: u8 = input[(pos & mask)];
                        {
                            let _rhs = 1;
                            let cur_ind = (prev + lut[(literal as i32 >> 6) as usize]);
                            let _lhs = &mut bigram_prefix_histo[cur_ind as usize];
                            *_lhs = (*_lhs).wrapping_add(_rhs as u32);
                        }
                        prev = lut[(literal as i32 >> 6) as usize] * 3i32;
                    }
                    pos = pos.wrapping_add(1);
                }
            }
            start_pos = start_pos.wrapping_add(4096);
        }
        ChooseContextMap(
            quality,
            &mut bigram_prefix_histo[..],
            num_literal_contexts,
            literal_context_map,
        );
    }
}
fn WriteMetaBlockInternal<Alloc: BrotliAlloc, Cb>(
    alloc: &mut Alloc,
    data: &[u8],
    mask: usize,
    last_flush_pos: u64,
    bytes: usize,
    mut is_last: bool,
    literal_context_mode: ContextType,
    params: &BrotliEncoderParams,
    lit_scratch_space: &mut <HistogramLiteral as CostAccessors>::i32vec,
    cmd_scratch_space: &mut <HistogramCommand as CostAccessors>::i32vec,
    dst_scratch_space: &mut <HistogramDistance as CostAccessors>::i32vec,
    prev_byte: u8,
    prev_byte2: u8,
    num_literals: usize,
    num_commands: usize,
    commands: &mut [Command],
    saved_dist_cache: &[i32; kNumDistanceCacheEntries],
    dist_cache: &mut [i32; 16],
    recoder_state: &mut RecoderState,
    storage_ix: &mut usize,
    storage: &mut [u8],
    cb: &mut Cb,
) where
    Cb: FnMut(
        &mut interface::PredictionModeContextMap<InputReferenceMut>,
        &mut [interface::StaticCommand],
        interface::InputPair,
        &mut Alloc,
    ),
{
    let actual_is_last = is_last;
    if params.appendable {
        is_last = false;
    } else {
        assert!(!params.catable); // Sanitize Params senforces this constraint
    }
    let wrapped_last_flush_pos: u32 = WrapPosition(last_flush_pos);

    let literal_context_lut = BROTLI_CONTEXT_LUT(literal_context_mode);
    let mut block_params = params.clone();
    if bytes == 0usize {
        BrotliWriteBits(2usize, 3, storage_ix, storage);
        *storage_ix = storage_ix.wrapping_add(7u32 as usize) & !7u32 as usize;
        return;
    }
    if !should_compress(
        data,
        mask,
        last_flush_pos,
        bytes,
        num_literals,
        num_commands,
    ) {
        dist_cache[..4].clone_from_slice(&saved_dist_cache[..4]);
        store_uncompressed_meta_block(
            alloc,
            is_last,
            data,
            wrapped_last_flush_pos as usize,
            mask,
            params,
            bytes,
            recoder_state,
            storage_ix,
            storage,
            false,
            cb,
        );
        if actual_is_last != is_last {
            BrotliWriteEmptyLastMetaBlock(storage_ix, storage)
        }
        return;
    }
    let saved_byte_location = (*storage_ix) >> 3;
    let last_bytes: u16 =
        ((storage[saved_byte_location + 1] as u16) << 8) | storage[saved_byte_location] as u16;
    let last_bytes_bits: u8 = *storage_ix as u8;
    /*if params.dist.num_direct_distance_codes != 0 ||
                      params.dist.distance_postfix_bits != 0 {
      RecomputeDistancePrefixes(commands,
                                num_commands,
                                params.dist.num_direct_distance_codes,
                                params.dist.distance_postfix_bits);
    }*/
    // why was this removed??
    if params.quality <= 2 {
        store_meta_block_fast(
            alloc,
            data,
            wrapped_last_flush_pos as usize,
            bytes,
            mask,
            is_last,
            params,
            saved_dist_cache,
            commands,
            num_commands,
            recoder_state,
            storage_ix,
            storage,
            cb,
        );
    } else if params.quality < 4 {
        store_meta_block_trivial(
            alloc,
            data,
            wrapped_last_flush_pos as usize,
            bytes,
            mask,
            is_last,
            params,
            saved_dist_cache,
            commands,
            num_commands,
            recoder_state,
            storage_ix,
            storage,
            cb,
        );
    } else {
        //let mut literal_context_mode: ContextType = ContextType::CONTEXT_UTF8;

        let mut mb = MetaBlockSplit::<Alloc>::new();
        if params.quality < 10i32 {
            let mut num_literal_contexts: usize = 1;
            let mut literal_context_map: &[u32] = &[];
            if params.disable_literal_context_modeling == 0 {
                DecideOverLiteralContextModeling(
                    data,
                    wrapped_last_flush_pos as usize,
                    bytes,
                    mask,
                    params.quality,
                    params.size_hint,
                    &mut num_literal_contexts,
                    &mut literal_context_map,
                );
            }
            BrotliBuildMetaBlockGreedy(
                alloc,
                data,
                wrapped_last_flush_pos as usize,
                mask,
                prev_byte,
                prev_byte2,
                literal_context_mode,
                literal_context_lut,
                num_literal_contexts,
                literal_context_map,
                commands,
                num_commands,
                &mut mb,
            );
        } else {
            BrotliBuildMetaBlock(
                alloc,
                data,
                wrapped_last_flush_pos as usize,
                mask,
                &mut block_params,
                prev_byte,
                prev_byte2,
                commands,
                num_commands,
                literal_context_mode,
                lit_scratch_space,
                cmd_scratch_space,
                dst_scratch_space,
                &mut mb,
            );
        }
        if params.quality >= 4i32 {
            let mut num_effective_dist_codes = block_params.dist.alphabet_size;
            if num_effective_dist_codes > BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS as u32 {
                num_effective_dist_codes = BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS as u32;
            }
            BrotliOptimizeHistograms(num_effective_dist_codes as usize, &mut mb);
        }
        store_meta_block(
            alloc,
            data,
            wrapped_last_flush_pos as usize,
            bytes,
            mask,
            prev_byte,
            prev_byte2,
            is_last,
            &block_params,
            literal_context_mode,
            saved_dist_cache,
            commands,
            num_commands,
            &mut mb,
            recoder_state,
            storage_ix,
            storage,
            cb,
        );
        mb.destroy(alloc);
    }
    if bytes + 4 + saved_byte_location < (*storage_ix >> 3) {
        dist_cache[..4].clone_from_slice(&saved_dist_cache[..4]);
        //memcpy(dist_cache,
        //     saved_dist_cache,
        //     (4usize).wrapping_mul(::core::mem::size_of::<i32>()));
        storage[saved_byte_location] = last_bytes as u8;
        storage[saved_byte_location + 1] = (last_bytes >> 8) as u8;
        *storage_ix = last_bytes_bits as usize;
        store_uncompressed_meta_block(
            alloc,
            is_last,
            data,
            wrapped_last_flush_pos as usize,
            mask,
            params,
            bytes,
            recoder_state,
            storage_ix,
            storage,
            true,
            cb,
        );
    }
    if actual_is_last != is_last {
        BrotliWriteEmptyLastMetaBlock(storage_ix, storage)
    }
}

fn ChooseDistanceParams(params: &mut BrotliEncoderParams) {
    let mut num_direct_distance_codes = 0u32;
    let mut distance_postfix_bits = 0u32;

    if params.quality >= 4 {
        if params.mode == BrotliEncoderMode::BROTLI_MODE_FONT {
            distance_postfix_bits = 1;
            num_direct_distance_codes = 12;
        } else {
            distance_postfix_bits = params.dist.distance_postfix_bits;
            num_direct_distance_codes = params.dist.num_direct_distance_codes;
        }
        let ndirect_msb = (num_direct_distance_codes >> distance_postfix_bits) & 0x0f;
        if distance_postfix_bits > BROTLI_MAX_NPOSTFIX as u32
            || num_direct_distance_codes > BROTLI_MAX_NDIRECT as u32
            || (ndirect_msb << distance_postfix_bits) != num_direct_distance_codes
        {
            distance_postfix_bits = 0;
            num_direct_distance_codes = 0;
        }
    }
    BrotliInitDistanceParams(params, distance_postfix_bits, num_direct_distance_codes);
    /*(
    if (params.large_window) {
        max_distance = BROTLI_MAX_ALLOWED_DISTANCE;
        if (num_direct_distance_codes != 0 || distance_postfix_bits != 0) {
            max_distance = (3 << 29) - 4;
        }
        alphabet_size = BROTLI_DISTANCE_ALPHABET_SIZE(
            num_direct_distance_codes, distance_postfix_bits,
            BROTLI_LARGE_MAX_DISTANCE_BITS);
    } else {
        alphabet_size = BROTLI_DISTANCE_ALPHABET_SIZE(
            num_direct_distance_codes, distance_postfix_bits,
            BROTLI_MAX_DISTANCE_BITS);

    }

    params.dist.num_direct_distance_codes = num_direct_distance_codes;
    params.dist.distance_postfix_bits = distance_postfix_bits;
    params.dist.alphabet_size = alphabet_size;
    params.dist.max_distance = max_distance;*/
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn encode_data<MetablockCallback>(
        &mut self,
        is_last: bool,
        force_flush: bool,
        out_size: &mut usize,
        callback: &mut MetablockCallback,
        // mut output: &'a mut &'a mut [u8]
    ) -> bool
    where
        MetablockCallback: FnMut(
            &mut interface::PredictionModeContextMap<InputReferenceMut>,
            &mut [interface::StaticCommand],
            interface::InputPair,
            &mut Alloc,
        ),
    {
        let mut delta: u64 = self.unprocessed_input_size();
        let mut bytes: u32 = delta as u32;
        let mask = self.ringbuffer_.mask_;
        if !self.ensure_initialized() {
            return false;
        }
        let dictionary = BrotliGetDictionary();
        if self.is_last_block_emitted_ {
            return false;
        }
        if is_last {
            self.is_last_block_emitted_ = true;
        }
        if delta > self.input_block_size() as u64 {
            return false;
        }
        let mut storage_ix: usize = usize::from(self.last_bytes_bits_);
        {
            let meta_size = max(
                bytes as usize,
                self.input_pos_.wrapping_sub(self.last_flush_pos_) as usize,
            );
            self.get_brotli_storage((2usize).wrapping_mul(meta_size).wrapping_add(503 + 24));
        }
        {
            self.storage_.slice_mut()[0] = self.last_bytes_ as u8;
            self.storage_.slice_mut()[1] = (self.last_bytes_ >> 8) as u8;
        }
        let mut catable_header_size = 0;
        if let IsFirst::NothingWritten = self.is_first_mb {
            if self.params.magic_number {
                BrotliWriteMetadataMetaBlock(
                    &self.params,
                    &mut storage_ix,
                    self.storage_.slice_mut(),
                );
                self.last_bytes_ = self.storage_.slice()[(storage_ix >> 3)] as u16
                    | ((self.storage_.slice()[1 + (storage_ix >> 3)] as u16) << 8);
                self.last_bytes_bits_ = (storage_ix & 7u32 as usize) as u8;
                self.next_out_ = NextOut::DynamicStorage(0);
                catable_header_size = storage_ix >> 3;
                *out_size = catable_header_size;
                self.is_first_mb = IsFirst::HeaderWritten;
            }
        }
        if let IsFirst::BothCatableBytesWritten = self.is_first_mb {
            // nothing to do here, move along
        } else if !self.params.catable {
            self.is_first_mb = IsFirst::BothCatableBytesWritten;
        } else if bytes != 0 {
            assert!(self.last_processed_pos_ < 2 || self.custom_dictionary);
            let num_bytes_to_write_uncompressed: usize = min(2, bytes as usize);
            {
                let data =
                    &mut self.ringbuffer_.data_mo.slice_mut()[self.ringbuffer_.buffer_index..];
                store_uncompressed_meta_block(
                    &mut self.m8,
                    false,
                    data,
                    self.last_flush_pos_ as usize,
                    mask as usize,
                    &self.params,
                    num_bytes_to_write_uncompressed,
                    &mut self.recoder_state,
                    &mut storage_ix,
                    self.storage_.slice_mut(),
                    false, /* suppress meta-block logging */
                    callback,
                );
                self.last_bytes_ = self.storage_.slice()[(storage_ix >> 3)] as u16
                    | ((self.storage_.slice()[1 + (storage_ix >> 3)] as u16) << 8);
                self.last_bytes_bits_ = (storage_ix & 7u32 as usize) as u8;
                self.prev_byte2_ = self.prev_byte_;
                self.prev_byte_ = data[self.last_flush_pos_ as usize & mask as usize];
                if num_bytes_to_write_uncompressed == 2 {
                    self.prev_byte2_ = self.prev_byte_;
                    self.prev_byte_ = data[(self.last_flush_pos_ + 1) as usize & mask as usize];
                }
            }
            self.last_flush_pos_ += num_bytes_to_write_uncompressed as u64;
            bytes -= num_bytes_to_write_uncompressed as u32;
            self.last_processed_pos_ += num_bytes_to_write_uncompressed as u64;
            if num_bytes_to_write_uncompressed >= 2 {
                self.is_first_mb = IsFirst::BothCatableBytesWritten;
            } else if num_bytes_to_write_uncompressed == 1 {
                if let IsFirst::FirstCatableByteWritten = self.is_first_mb {
                    self.is_first_mb = IsFirst::BothCatableBytesWritten;
                } else {
                    self.is_first_mb = IsFirst::FirstCatableByteWritten;
                }
            }
            catable_header_size = storage_ix >> 3;
            self.next_out_ = NextOut::DynamicStorage(0);
            *out_size = catable_header_size;
            delta = self.unprocessed_input_size();
        }
        let mut wrapped_last_processed_pos: u32 = WrapPosition(self.last_processed_pos_);
        if self.params.quality == 1i32 && self.command_buf_.slice().is_empty() {
            let new_buf = allocate::<u32, _>(&mut self.m8, kCompressFragmentTwoPassBlockSize);
            self.command_buf_ = new_buf;
            let new_buf8 = allocate::<u8, _>(&mut self.m8, kCompressFragmentTwoPassBlockSize);
            self.literal_buf_ = new_buf8;
        }

        if self.params.quality == 0i32 || self.params.quality == 1i32 {
            let mut table_size: usize = 0;
            {
                if delta == 0 && !is_last {
                    *out_size = catable_header_size;
                    return true;
                }
                let data =
                    &mut self.ringbuffer_.data_mo.slice_mut()[self.ringbuffer_.buffer_index..];

                //s.storage_.slice_mut()[0] = (*s).last_bytes_ as u8;
                //        s.storage_.slice_mut()[1] = ((*s).last_bytes_ >> 8) as u8;

                let table: &mut [i32] =
                    GetHashTable!(self, self.params.quality, bytes as usize, &mut table_size);

                if self.params.quality == 0i32 {
                    compress_fragment_fast(
                        &mut self.m8,
                        &mut data[((wrapped_last_processed_pos & mask) as usize)..],
                        bytes as usize,
                        is_last,
                        table,
                        table_size,
                        &mut self.cmd_depths_[..],
                        &mut self.cmd_bits_[..],
                        &mut self.cmd_code_numbits_,
                        &mut self.cmd_code_[..],
                        &mut storage_ix,
                        self.storage_.slice_mut(),
                    );
                } else {
                    compress_fragment_two_pass(
                        &mut self.m8,
                        &mut data[((wrapped_last_processed_pos & mask) as usize)..],
                        bytes as usize,
                        is_last,
                        self.command_buf_.slice_mut(),
                        self.literal_buf_.slice_mut(),
                        table,
                        table_size,
                        &mut storage_ix,
                        self.storage_.slice_mut(),
                    );
                }
                self.last_bytes_ = self.storage_.slice()[(storage_ix >> 3)] as u16
                    | ((self.storage_.slice()[(storage_ix >> 3) + 1] as u16) << 8);
                self.last_bytes_bits_ = (storage_ix & 7u32 as usize) as u8;
            }
            self.update_last_processed_pos();
            // *output = &mut s.storage_.slice_mut();
            self.next_out_ = NextOut::DynamicStorage(0); // this always returns that
            *out_size = storage_ix >> 3;
            return true;
        }
        {
            let mut newsize: usize = self
                .num_commands_
                .wrapping_add(bytes.wrapping_div(2) as usize)
                .wrapping_add(1);
            if newsize > self.cmd_alloc_size_ {
                newsize = newsize.wrapping_add(bytes.wrapping_div(4).wrapping_add(16) as usize);
                self.cmd_alloc_size_ = newsize;
                let mut new_commands = allocate::<Command, _>(&mut self.m8, newsize);
                if !self.commands_.slice().is_empty() {
                    new_commands.slice_mut()[..self.num_commands_]
                        .clone_from_slice(&self.commands_.slice()[..self.num_commands_]);
                    <Alloc as Allocator<Command>>::free_cell(
                        &mut self.m8,
                        core::mem::take(&mut self.commands_),
                    );
                }
                self.commands_ = new_commands;
            }
        }
        InitOrStitchToPreviousBlock(
            &mut self.m8,
            &mut self.hasher_,
            &mut self.ringbuffer_.data_mo.slice_mut()[self.ringbuffer_.buffer_index..],
            mask as usize,
            self.custom_dictionary_size,
            &mut self.params,
            wrapped_last_processed_pos as usize,
            bytes as usize,
            is_last,
        );
        let literal_context_mode = ChooseContextMode(
            &self.params,
            self.ringbuffer_.data_mo.slice(),
            WrapPosition(self.last_flush_pos_) as usize,
            mask as usize,
            (self.input_pos_.wrapping_sub(self.last_flush_pos_)) as usize,
        );
        if self.num_commands_ != 0 && self.last_insert_len_ == 0 {
            self.extend_last_command(&mut bytes, &mut wrapped_last_processed_pos);
        }
        BrotliCreateBackwardReferences(
            &mut self.m8,
            dictionary,
            bytes as usize,
            wrapped_last_processed_pos as usize,
            &mut self.ringbuffer_.data_mo.slice_mut()[self.ringbuffer_.buffer_index..],
            mask as usize,
            self.custom_dictionary_size,
            &mut self.params,
            &mut self.hasher_,
            &mut self.dist_cache_,
            &mut self.last_insert_len_,
            &mut self.commands_.slice_mut()[self.num_commands_..],
            &mut self.num_commands_,
            &mut self.num_literals_,
        );
        {
            let max_length: usize = MaxMetablockSize(&mut self.params);
            let max_literals: usize = max_length.wrapping_div(8);
            let max_commands: usize = max_length.wrapping_div(8);
            let processed_bytes: usize =
                self.input_pos_.wrapping_sub(self.last_flush_pos_) as usize;
            let next_input_fits_metablock =
                processed_bytes.wrapping_add(self.input_block_size()) <= max_length;
            let should_flush = self.params.quality < 4
                && self.num_literals_.wrapping_add(self.num_commands_) >= 0x2fff;
            if !is_last
                && !force_flush
                && !should_flush
                && next_input_fits_metablock
                && self.num_literals_ < max_literals
                && self.num_commands_ < max_commands
            {
                if self.update_last_processed_pos() {
                    HasherReset(&mut self.hasher_);
                }
                *out_size = catable_header_size;
                return true;
            }
        }
        if self.last_insert_len_ > 0usize {
            self.commands_.slice_mut()[self.num_commands_].init_insert(self.last_insert_len_);
            self.num_commands_ = self.num_commands_.wrapping_add(1);
            self.num_literals_ = self.num_literals_.wrapping_add(self.last_insert_len_);
            self.last_insert_len_ = 0usize;
        }
        if !is_last && self.input_pos_ == self.last_flush_pos_ {
            *out_size = catable_header_size;
            return true;
        }
        {
            let metablock_size: u32 = self.input_pos_.wrapping_sub(self.last_flush_pos_) as u32;
            //let mut storage_ix: usize = s.last_bytes_bits_ as usize;
            //s.storage_.slice_mut()[0] = (*s).last_bytes_ as u8;
            //s.storage_.slice_mut()[1] = ((*s).last_bytes_ >> 8) as u8;

            WriteMetaBlockInternal(
                &mut self.m8,
                &mut self.ringbuffer_.data_mo.slice_mut()[self.ringbuffer_.buffer_index..],
                mask as usize,
                self.last_flush_pos_,
                metablock_size as usize,
                is_last,
                literal_context_mode,
                &mut self.params,
                &mut self.literal_scratch_space,
                &mut self.command_scratch_space,
                &mut self.distance_scratch_space,
                self.prev_byte_,
                self.prev_byte2_,
                self.num_literals_,
                self.num_commands_,
                self.commands_.slice_mut(),
                &mut self.saved_dist_cache_,
                &mut self.dist_cache_,
                &mut self.recoder_state,
                &mut storage_ix,
                self.storage_.slice_mut(),
                callback,
            );

            self.last_bytes_ = self.storage_.slice()[(storage_ix >> 3)] as u16
                | ((self.storage_.slice()[1 + (storage_ix >> 3)] as u16) << 8);
            self.last_bytes_bits_ = (storage_ix & 7u32 as usize) as u8;
            self.last_flush_pos_ = self.input_pos_;
            if self.update_last_processed_pos() {
                HasherReset(&mut self.hasher_);
            }
            let data = &self.ringbuffer_.data_mo.slice()[self.ringbuffer_.buffer_index..];
            if self.last_flush_pos_ > 0 {
                self.prev_byte_ =
                    data[(((self.last_flush_pos_ as u32).wrapping_sub(1) & mask) as usize)];
            }
            if self.last_flush_pos_ > 1 {
                self.prev_byte2_ =
                    data[((self.last_flush_pos_.wrapping_sub(2) as u32 & mask) as usize)];
            }
            self.num_commands_ = 0usize;
            self.num_literals_ = 0usize;
            self.saved_dist_cache_
                .clone_from_slice(self.dist_cache_.split_at(4).0);
            self.next_out_ = NextOut::DynamicStorage(0); // this always returns that
            *out_size = storage_ix >> 3;
            true
        }
    }

    fn write_metadata_header(&mut self) -> usize {
        let block_size = self.remaining_metadata_bytes_ as usize;
        let header = GetNextOut!(*self);
        let mut storage_ix: usize;
        storage_ix = self.last_bytes_bits_ as usize;
        header[0] = self.last_bytes_ as u8;
        header[1] = (self.last_bytes_ >> 8) as u8;
        self.last_bytes_ = 0;
        self.last_bytes_bits_ = 0;
        BrotliWriteBits(1, 0, &mut storage_ix, header);
        BrotliWriteBits(2usize, 3, &mut storage_ix, header);
        BrotliWriteBits(1, 0, &mut storage_ix, header);
        if block_size == 0usize {
            BrotliWriteBits(2usize, 0, &mut storage_ix, header);
        } else {
            let nbits: u32 = if block_size == 1 {
                0u32
            } else {
                Log2FloorNonZero((block_size as u32).wrapping_sub(1) as (u64)).wrapping_add(1)
            };
            let nbytes: u32 = nbits.wrapping_add(7).wrapping_div(8);
            BrotliWriteBits(2usize, nbytes as (u64), &mut storage_ix, header);
            BrotliWriteBits(
                (8u32).wrapping_mul(nbytes) as usize,
                block_size.wrapping_sub(1) as u64,
                &mut storage_ix,
                header,
            );
        }
        storage_ix.wrapping_add(7u32 as usize) >> 3
    }
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn process_metadata<
        MetaBlockCallback: FnMut(
            &mut interface::PredictionModeContextMap<InputReferenceMut>,
            &mut [interface::StaticCommand],
            interface::InputPair,
            &mut Alloc,
        ),
    >(
        &mut self,
        available_in: &mut usize,
        next_in_array: &[u8],
        next_in_offset: &mut usize,
        available_out: &mut usize,
        next_out_array: &mut [u8],
        next_out_offset: &mut usize,
        total_out: &mut Option<usize>,
        metablock_callback: &mut MetaBlockCallback,
    ) -> bool {
        if *available_in > (1u32 << 24) as usize {
            return false;
        }
        if self.stream_state_ as i32 == BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING as i32 {
            self.remaining_metadata_bytes_ = *available_in as u32;
            self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_METADATA_HEAD;
        }
        if self.stream_state_ as i32 != BrotliEncoderStreamState::BROTLI_STREAM_METADATA_HEAD as i32
            && (self.stream_state_ as i32
                != BrotliEncoderStreamState::BROTLI_STREAM_METADATA_BODY as i32)
        {
            return false;
        }
        loop {
            if self.inject_flush_or_push_output(
                available_out,
                next_out_array,
                next_out_offset,
                total_out,
            ) {
                continue;
            }
            if self.available_out_ != 0usize {
                break;
            }
            if self.input_pos_ != self.last_flush_pos_ {
                let mut avail_out: usize = self.available_out_;
                let result = self.encode_data(false, true, &mut avail_out, metablock_callback);
                self.available_out_ = avail_out;
                if !result {
                    return false;
                }
                continue;
            }
            if self.stream_state_ as i32
                == BrotliEncoderStreamState::BROTLI_STREAM_METADATA_HEAD as i32
            {
                self.next_out_ = NextOut::TinyBuf(0);
                self.available_out_ = self.write_metadata_header();
                self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_METADATA_BODY;
                {
                    continue;
                }
            } else {
                if self.remaining_metadata_bytes_ == 0u32 {
                    self.remaining_metadata_bytes_ = u32::MAX;
                    self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING;
                    {
                        break;
                    }
                }
                if *available_out != 0 {
                    let copy: u32 =
                        min(self.remaining_metadata_bytes_ as usize, *available_out) as u32;
                    next_out_array[*next_out_offset..(*next_out_offset + copy as usize)]
                        .clone_from_slice(
                            &next_in_array[*next_in_offset..(*next_in_offset + copy as usize)],
                        );
                    //memcpy(*next_out, *next_in, copy as usize);
                    // *next_in = next_in.offset(copy as isize);
                    *next_in_offset += copy as usize;
                    *available_in = available_in.wrapping_sub(copy as usize);
                    self.remaining_metadata_bytes_ =
                        self.remaining_metadata_bytes_.wrapping_sub(copy);
                    *next_out_offset += copy as usize;
                    // *next_out = next_out.offset(copy as isize);
                    *available_out = available_out.wrapping_sub(copy as usize);
                } else {
                    let copy: u32 = min(self.remaining_metadata_bytes_, 16u32);
                    self.next_out_ = NextOut::TinyBuf(0);
                    GetNextOut!(self)[..(copy as usize)].clone_from_slice(
                        &next_in_array[*next_in_offset..(*next_in_offset + copy as usize)],
                    );
                    //memcpy(s.next_out_, *next_in, copy as usize);
                    // *next_in = next_in.offset(copy as isize);
                    *next_in_offset += copy as usize;
                    *available_in = available_in.wrapping_sub(copy as usize);
                    self.remaining_metadata_bytes_ =
                        self.remaining_metadata_bytes_.wrapping_sub(copy);
                    self.available_out_ = copy as usize;
                }
                {
                    continue;
                }
            }
        }
        true
    }
}
fn CheckFlushCompleteInner(
    stream_state: &mut BrotliEncoderStreamState,
    available_out: usize,
    next_out: &mut NextOut,
) {
    if *stream_state == BrotliEncoderStreamState::BROTLI_STREAM_FLUSH_REQUESTED
        && (available_out == 0)
    {
        *stream_state = BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING;
        *next_out = NextOut::None;
    }
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    fn check_flush_complete(&mut self) {
        CheckFlushCompleteInner(
            &mut self.stream_state_,
            self.available_out_,
            &mut self.next_out_,
        );
    }

    fn compress_stream_fast(
        &mut self,
        op: BrotliEncoderOperation,
        available_in: &mut usize,
        next_in_array: &[u8],
        next_in_offset: &mut usize,
        available_out: &mut usize,
        next_out_array: &mut [u8],
        next_out_offset: &mut usize,
        total_out: &mut Option<usize>,
    ) -> bool {
        let block_size_limit: usize = 1 << self.params.lgwin;
        let buf_size: usize = min(
            kCompressFragmentTwoPassBlockSize,
            min(*available_in, block_size_limit),
        );
        let mut command_buf = alloc_default::<u32, Alloc>();
        let mut literal_buf = alloc_default::<u8, Alloc>();
        if self.params.quality != 0i32 && (self.params.quality != 1i32) {
            return false;
        }
        if self.params.quality == 1i32 {
            if self.command_buf_.slice().is_empty()
                && (buf_size == kCompressFragmentTwoPassBlockSize)
            {
                self.command_buf_ =
                    allocate::<u32, _>(&mut self.m8, kCompressFragmentTwoPassBlockSize);
                self.literal_buf_ =
                    allocate::<u8, _>(&mut self.m8, kCompressFragmentTwoPassBlockSize);
            }
            if !self.command_buf_.slice().is_empty() {
                command_buf = core::mem::take(&mut self.command_buf_);
                literal_buf = core::mem::take(&mut self.literal_buf_);
            } else {
                command_buf = allocate::<u32, _>(&mut self.m8, buf_size);
                literal_buf = allocate::<u8, _>(&mut self.m8, buf_size);
            }
        }
        loop {
            if self.inject_flush_or_push_output(
                available_out,
                next_out_array,
                next_out_offset,
                total_out,
            ) {
                continue;
            }
            if self.available_out_ == 0usize
                && (self.stream_state_ as i32
                    == BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING as i32)
                && (*available_in != 0usize
                    || op as i32 != BrotliEncoderOperation::BROTLI_OPERATION_PROCESS as i32)
            {
                let block_size: usize = min(block_size_limit, *available_in);
                let is_last = *available_in == block_size
                    && op == BrotliEncoderOperation::BROTLI_OPERATION_FINISH;
                let force_flush = *available_in == block_size
                    && op == BrotliEncoderOperation::BROTLI_OPERATION_FLUSH;
                let max_out_size: usize = (2usize).wrapping_mul(block_size).wrapping_add(503);
                let mut inplace: i32 = 1i32;
                let storage: &mut [u8];
                let mut storage_ix: usize = self.last_bytes_bits_ as usize;
                let mut table_size: usize = 0;

                if force_flush && block_size == 0 {
                    self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_FLUSH_REQUESTED;
                    {
                        continue;
                    }
                }
                if max_out_size <= *available_out {
                    storage = &mut next_out_array[*next_out_offset..]; //GetNextOut!(s);
                } else {
                    inplace = 0i32;
                    self.get_brotli_storage(max_out_size);
                    storage = self.storage_.slice_mut();
                }
                storage[0] = self.last_bytes_ as u8;
                storage[1] = (self.last_bytes_ >> 8) as u8;
                let table: &mut [i32] =
                    GetHashTable!(self, self.params.quality, block_size, &mut table_size);
                if self.params.quality == 0i32 {
                    compress_fragment_fast(
                        &mut self.m8,
                        &(next_in_array)[*next_in_offset..],
                        block_size,
                        is_last,
                        table,
                        table_size,
                        &mut self.cmd_depths_[..],
                        &mut self.cmd_bits_[..],
                        &mut self.cmd_code_numbits_,
                        &mut self.cmd_code_[..],
                        &mut storage_ix,
                        storage,
                    );
                } else {
                    compress_fragment_two_pass(
                        &mut self.m8,
                        &(next_in_array)[*next_in_offset..],
                        block_size,
                        is_last,
                        command_buf.slice_mut(),
                        literal_buf.slice_mut(),
                        table,
                        table_size,
                        &mut storage_ix,
                        storage,
                    );
                }
                *next_in_offset += block_size;
                *available_in = available_in.wrapping_sub(block_size);
                if inplace != 0 {
                    let out_bytes: usize = storage_ix >> 3;
                    *next_out_offset += out_bytes;
                    *available_out = available_out.wrapping_sub(out_bytes);
                    self.total_out_ = self.total_out_.wrapping_add(out_bytes as u64);
                    if let &mut Some(ref mut total_out_inner) = total_out {
                        *total_out_inner = self.total_out_ as usize;
                    }
                } else {
                    let out_bytes: usize = storage_ix >> 3;
                    self.next_out_ = NextOut::DynamicStorage(0);
                    self.available_out_ = out_bytes;
                }
                self.last_bytes_ = storage[(storage_ix >> 3)] as u16
                    | ((storage[1 + (storage_ix >> 3)] as u16) << 8);
                self.last_bytes_bits_ = (storage_ix & 7u32 as usize) as u8;
                if force_flush {
                    self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_FLUSH_REQUESTED;
                }
                if is_last {
                    self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_FINISHED;
                }
                {
                    continue;
                }
            }
            {
                break;
            }
        }
        if command_buf.slice().len() == kCompressFragmentTwoPassBlockSize
            && self.command_buf_.slice().is_empty()
        {
            // undo temporary aliasing of command_buf and literal_buf
            self.command_buf_ = core::mem::take(&mut command_buf);
            self.literal_buf_ = core::mem::take(&mut literal_buf);
        } else {
            <Alloc as Allocator<u32>>::free_cell(&mut self.m8, command_buf);
            <Alloc as Allocator<u8>>::free_cell(&mut self.m8, literal_buf);
        }
        self.check_flush_complete();
        true
    }

    fn remaining_input_block_size(&mut self) -> usize {
        let delta: u64 = self.unprocessed_input_size();
        let block_size = self.input_block_size();
        if delta >= block_size as u64 {
            return 0usize;
        }
        (block_size as u64).wrapping_sub(delta) as usize
    }

    pub fn compress_stream<
        MetablockCallback: FnMut(
            &mut interface::PredictionModeContextMap<InputReferenceMut>,
            &mut [interface::StaticCommand],
            interface::InputPair,
            &mut Alloc,
        ),
    >(
        &mut self,
        op: BrotliEncoderOperation,
        available_in: &mut usize,
        next_in_array: &[u8],
        next_in_offset: &mut usize,
        available_out: &mut usize,
        next_out_array: &mut [u8],
        next_out_offset: &mut usize,
        total_out: &mut Option<usize>,
        metablock_callback: &mut MetablockCallback,
    ) -> bool {
        if !self.ensure_initialized() {
            return false;
        }
        if self.remaining_metadata_bytes_ != u32::MAX {
            if *available_in != self.remaining_metadata_bytes_ as usize {
                return false;
            }
            if op as i32 != BrotliEncoderOperation::BROTLI_OPERATION_EMIT_METADATA as i32 {
                return false;
            }
        }
        if op as i32 == BrotliEncoderOperation::BROTLI_OPERATION_EMIT_METADATA as i32 {
            self.update_size_hint(0);
            return self.process_metadata(
                available_in,
                next_in_array,
                next_in_offset,
                available_out,
                next_out_array,
                next_out_offset,
                total_out,
                metablock_callback,
            );
        }
        if self.stream_state_ as i32 == BrotliEncoderStreamState::BROTLI_STREAM_METADATA_HEAD as i32
            || self.stream_state_ as i32
                == BrotliEncoderStreamState::BROTLI_STREAM_METADATA_BODY as i32
        {
            return false;
        }
        if self.stream_state_ as i32 != BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING as i32
            && (*available_in != 0usize)
        {
            return false;
        }
        if (self.params.quality == 0i32 || self.params.quality == 1i32) && !self.params.catable {
            // this part of the code does not support concatability
            return self.compress_stream_fast(
                op,
                available_in,
                next_in_array,
                next_in_offset,
                available_out,
                next_out_array,
                next_out_offset,
                total_out,
            );
        }
        loop {
            let remaining_block_size: usize = self.remaining_input_block_size();
            if remaining_block_size != 0usize && (*available_in != 0usize) {
                let copy_input_size: usize = min(remaining_block_size, *available_in);
                self.copy_input_to_ring_buffer(copy_input_size, &next_in_array[*next_in_offset..]);
                *next_in_offset += copy_input_size;
                *available_in = available_in.wrapping_sub(copy_input_size);
                {
                    continue;
                }
            }
            if self.inject_flush_or_push_output(
                available_out,
                next_out_array,
                next_out_offset,
                total_out,
            ) {
                continue;
            }
            if self.available_out_ == 0usize
                && (self.stream_state_ as i32
                    == BrotliEncoderStreamState::BROTLI_STREAM_PROCESSING as i32)
                && (remaining_block_size == 0usize
                    || op as i32 != BrotliEncoderOperation::BROTLI_OPERATION_PROCESS as i32)
            {
                let is_last =
                    *available_in == 0 && op == BrotliEncoderOperation::BROTLI_OPERATION_FINISH;
                let force_flush =
                    *available_in == 0 && op == BrotliEncoderOperation::BROTLI_OPERATION_FLUSH;

                self.update_size_hint(*available_in);
                let mut avail_out = self.available_out_;
                let result =
                    self.encode_data(is_last, force_flush, &mut avail_out, metablock_callback);
                self.available_out_ = avail_out;
                //this function set next_out to &storage[0]
                if !result {
                    return false;
                }
                if force_flush {
                    self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_FLUSH_REQUESTED;
                }
                if is_last {
                    self.stream_state_ = BrotliEncoderStreamState::BROTLI_STREAM_FINISHED;
                }
                {
                    continue;
                }
            }
            {
                break;
            }
        }
        self.check_flush_complete();
        true
    }

    pub fn is_finished(&self) -> bool {
        self.stream_state_ == BrotliEncoderStreamState::BROTLI_STREAM_FINISHED
            && !self.has_more_output()
    }

    pub fn has_more_output(&self) -> bool {
        self.available_out_ != 0
    }

    pub fn take_output(&mut self, size: &mut usize) -> &[u8] {
        let mut consumed_size: usize = self.available_out_;
        let mut result: &[u8] = GetNextOut!(*self);
        if *size != 0 {
            consumed_size = min(*size, self.available_out_);
        }
        if consumed_size != 0 {
            self.next_out_ = NextOutIncrement(&self.next_out_, consumed_size as i32);
            self.available_out_ = self.available_out_.wrapping_sub(consumed_size);
            self.total_out_ = self.total_out_.wrapping_add(consumed_size as u64);
            CheckFlushCompleteInner(
                &mut self.stream_state_,
                self.available_out_,
                &mut self.next_out_,
            );
            *size = consumed_size;
        } else {
            *size = 0usize;
            result = &[];
        }
        result
    }
}

pub fn BrotliEncoderVersion() -> u32 {
    0x0100_0f01
}

impl<Alloc: BrotliAlloc> BrotliEncoderStateStruct<Alloc> {
    pub fn input_block_size(&mut self) -> usize {
        if !self.ensure_initialized() {
            return 0;
        }
        1 << self.params.lgblock
    }

    pub fn write_data<
        'a,
        MetablockCallback: FnMut(
            &mut interface::PredictionModeContextMap<InputReferenceMut>,
            &mut [interface::StaticCommand],
            interface::InputPair,
            &mut Alloc,
        ),
    >(
        &'a mut self,
        // FIXME: this should be bool
        is_last: i32,
        // FIXME: this should be bool
        force_flush: i32,
        // FIXME: this should probably be removed because the slice already contains the size
        out_size: &mut usize,
        // FIXME: this should be part of the fn return value
        output: &'a mut &'a mut [u8],
        metablock_callback: &mut MetablockCallback,
    ) -> bool {
        let ret = self.encode_data(is_last != 0, force_flush != 0, out_size, metablock_callback);
        *output = self.storage_.slice_mut();
        ret
    }
}

#[cfg(feature = "std")]
mod test {
    #[cfg(test)]
    use alloc_stdlib::StandardAlloc;

    #[test]
    fn test_encoder_compress() {
        let input = include_bytes!("../../testdata/alice29.txt");
        let mut output_buffer = [0; 100000];
        let mut output_len = output_buffer.len();
        let ret = super::encoder_compress(
            StandardAlloc::default(),
            &mut StandardAlloc::default(),
            9,
            16,
            super::BrotliEncoderMode::BROTLI_MODE_GENERIC,
            input.len(),
            input,
            &mut output_len,
            &mut output_buffer,
            &mut |_, _, _, _| (),
        );
        assert!(ret);
        assert_eq!(output_len, 51737);
        let mut roundtrip = [0u8; 200000];
        let (_, s, t) = super::super::test::oneshot_decompress(
            &output_buffer[..output_len],
            &mut roundtrip[..],
        );
        assert_eq!(roundtrip[..t], input[..]);
        assert_eq!(s, output_len);
    }
}
