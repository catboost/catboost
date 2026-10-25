#![allow(non_snake_case)]
#![allow(unused_parens)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_macros)]

// #[macro_use] //<- for debugging, remove xprintln from bit_reader and replace with println
// extern crate std;
use core;
use super::alloc;
pub use alloc::{AllocatedStackMemory, Allocator, SliceWrapper, SliceWrapperMut, StackAllocator};

use core::mem;

use super::bit_reader;
use super::huffman;
use super::state;
use super::prefix;

use super::transform::{TransformDictionaryWord, kNumTransforms};
use state::{BlockTypeAndLengthState, BrotliRunningContextMapState, BrotliRunningDecodeUint8State,
            BrotliRunningHuffmanState, BrotliRunningMetablockHeaderState,
            BrotliRunningReadBlockLengthState, BrotliRunningState, BrotliRunningTreeGroupState,
            BrotliRunningUncompressedState, kLiteralContextBits,
            BrotliDecoderErrorCode,
};
use context::{kContextLookup};
use ::dictionary::{kBrotliDictionary, kBrotliDictionaryOffsetsByLength,
                   kBrotliDictionarySizeBitsByLength, kBrotliMaxDictionaryWordLength,
                   kBrotliMinDictionaryWordLength};
pub use huffman::{HuffmanCode, HuffmanTreeGroup};
#[repr(C)]
#[derive(Debug)]
pub enum BrotliResult {
  ResultSuccess = 1,
  NeedsMoreInput = 2,
  NeedsMoreOutput = 3,
  ResultFailure = 0,
}
const kBrotliWindowGap: u32 = 16;
const kBrotliLargeMinWbits: u32 = 10;
const kBrotliLargeMaxWbits: u32 = 30;
const kBrotliMaxPostfix: usize = 3;
const kBrotliMaxAllowedDistance: u32 = 0x7FFFFFFC;
const kDefaultCodeLength: u32 = 8;
const kCodeLengthRepeatCode: u32 = 16;
pub const kNumLiteralCodes: u16 = 256;
pub const kNumInsertAndCopyCodes: u16 = 704;
pub const kNumBlockLengthCodes: u32 = 26;
const kDistanceContextBits: i32 = 2;
const HUFFMAN_TABLE_BITS: u32 = 8;
const HUFFMAN_TABLE_MASK: u32 = 0xff;
const CODE_LENGTH_CODES: usize = 18;
const kCodeLengthCodeOrder: [u8; CODE_LENGTH_CODES] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10,
                                                       11, 12, 13, 14, 15];

// Static prefix code for the complex code length code lengths.
const kCodeLengthPrefixLength: [u8; 16] = [2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4];

const kCodeLengthPrefixValue: [u8; 16] = [0, 4, 3, 2, 0, 4, 3, 1, 0, 4, 3, 2, 0, 4, 3, 5];


macro_rules! BROTLI_LOG_UINT (
    ($num : expr) => {
       xprintln!("{:?} = {:?}", stringify!($num),  $num)
    };
);

macro_rules! BROTLI_LOG (
    ($str : expr, $num : expr) => {xprintln!("{:?} {:?}", $str, $num);};
    ($str : expr, $num0 : expr, $num1 : expr) => {xprintln!("{:?} {:?} {:?}", $str, $num0, $num1);};
    ($str : expr, $num0 : expr, $num1 : expr, $num2 : expr) => {
        xprintln!("{:?} {:?} {:?} {:?}", $str, $num0, $num1, $num2);
    };
    ($str : expr, $num0 : expr, $num1 : expr, $num2 : expr, $num3 : expr) => {
        xprintln!("{:?} {:?} {:?} {:?} {:?}", $str, $num0, $num1, $num2, $num3);
    };
);
fn is_fatal(e: BrotliDecoderErrorCode) -> bool {
  (e as i64) < 0
}
fn assign_error_code(output: &mut BrotliDecoderErrorCode, input: BrotliDecoderErrorCode) -> BrotliDecoderErrorCode {
  *output = input;
  input
}

#[allow(non_snake_case)]
macro_rules! SaveErrorCode {
  ($state: expr, $e:expr) => {
    match assign_error_code(&mut $state.error_code, $e) {
      BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS =>
        BrotliResult::ResultSuccess,
      BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT =>
        BrotliResult::NeedsMoreInput,
      BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_OUTPUT =>
        BrotliResult::NeedsMoreOutput,
      _ =>
        BrotliResult::ResultFailure,
    }
  }
}
macro_rules! SaveResult {
  ($state: expr, $e:expr) => {
    match ($state.error_code = match $e  {
      BrotliResult::ResultSuccess => BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS,
      BrotliResult::NeedsMoreInput => BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT,
      BrotliResult::NeedsMoreOutput => BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_OUTPUT,
      BrotliResult::ResultFailure => BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_UNREACHABLE,
    }) {
      BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS =>
        BrotliResult::ResultSuccess,
      BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT =>
        BrotliResult::NeedsMoreInput,
      BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_OUTPUT =>
        BrotliResult::NeedsMoreOutput,
      _ =>
        BrotliResult::ResultFailure,
    }
  }
}
macro_rules! BROTLI_LOG_ARRAY_INDEX (
    ($array : expr, $index : expr) => {
       xprintln!("{:?}[{:?}] = {:?}", stringify!($array), $index,  $array[$index as usize])
    };
);


const NUM_DISTANCE_SHORT_CODES: u32 = 16;
pub const BROTLI_MAX_DISTANCE_BITS:u32 = 24;

pub const BROTLI_LARGE_MAX_DISTANCE_BITS: u32 = 62;

pub fn BROTLI_DISTANCE_ALPHABET_SIZE(NPOSTFIX: u32, NDIRECT:u32, MAXNBITS: u32) -> u32 {
    NUM_DISTANCE_SHORT_CODES + (NDIRECT) +
        ((MAXNBITS) << ((NPOSTFIX) + 1))
}

// pub struct BrotliState {
// total_written : usize,
// }
//
pub use state::BrotliState;
// impl BrotliState {
// pub fn new() -> BrotliState {
// return BrotliState {total_written: 0 };
// }
// }

/* Decodes WBITS by reading 1 - 7 bits, or 0x11 for "Large Window Brotli".
   Precondition: bit-reader accumulator has at least 8 bits. */
fn DecodeWindowBits(s_large_window: &mut bool,
                    s_window_bits:&mut u32,
                    br: &mut bit_reader::BrotliBitReader) -> BrotliDecoderErrorCode {
  let mut n: u32 = 0;
  let large_window = *s_large_window;
  *s_large_window = false;
  bit_reader::BrotliTakeBits(br, 1, &mut n);
  if (n == 0) {
    *s_window_bits = 16;
    return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
  }
  bit_reader::BrotliTakeBits(br, 3, &mut n);
  if (n != 0) {
    *s_window_bits = 17 + n;
    return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
  }
  bit_reader::BrotliTakeBits(br, 3, &mut n);
  if (n == 1) {
    if (large_window) {
      bit_reader::BrotliTakeBits(br, 1, &mut n);
      if (n == 1) {
        return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_WINDOW_BITS;
      }
      *s_large_window = true;
      return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
    } else {
      return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_WINDOW_BITS;
    }
  }
  if (n != 0) {
    *s_window_bits = 8 + n;
    return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
  }
  *s_window_bits = 17;
  return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
}


#[cold]
fn mark_unlikely() {}

fn DecodeVarLenUint8(substate_decode_uint8: &mut state::BrotliRunningDecodeUint8State,
                     mut br: &mut bit_reader::BrotliBitReader,
                     value: &mut u32,
                     input: &[u8])
                     -> BrotliDecoderErrorCode {
  let mut bits: u32 = 0;
  loop {
    match *substate_decode_uint8 {
      BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_NONE => {
        if !bit_reader::BrotliSafeReadBits(&mut br, 1, &mut bits, input) {
          mark_unlikely();
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if (bits == 0) {
          *value = 0;
          return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
        }
        *substate_decode_uint8 = BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_SHORT;
        // No break, transit to the next state.
      }
      BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_SHORT => {
        if !bit_reader::BrotliSafeReadBits(&mut br, 3, &mut bits, input) {
          mark_unlikely();
          *substate_decode_uint8 = BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_SHORT;
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if (bits == 0) {
          *value = 1;
          *substate_decode_uint8 = BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_NONE;
          return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
        }
        // Use output value as a temporary storage. It MUST be persisted.
        *value = bits;
        // No break, transit to the next state.
        *substate_decode_uint8 = BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_LONG;
      }
      BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_LONG => {
        if !bit_reader::BrotliSafeReadBits(&mut br, *value, &mut bits, input) {
          mark_unlikely();
          *substate_decode_uint8 = BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_LONG;
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        *value = (1u32 << *value) + bits;
        *substate_decode_uint8 = BrotliRunningDecodeUint8State::BROTLI_STATE_DECODE_UINT8_NONE;
        return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
      }
    }
  }
}

fn DecodeMetaBlockLength<AllocU8: alloc::Allocator<u8>,
                         AllocU32: alloc::Allocator<u32>,
                         AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  let mut bits: u32 = 0;
  loop {
    match s.substate_metablock_header {
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NONE => {
        if !bit_reader::BrotliSafeReadBits(&mut s.br, 1, &mut bits, input) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        s.is_last_metablock = bits as u8;
        s.meta_block_remaining_len = 0;
        s.is_uncompressed = 0;
        s.is_metadata = 0;
        if (s.is_last_metablock == 0) {
          s.substate_metablock_header =
            BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NIBBLES;
          continue;
        }
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_EMPTY;
        // No break, transit to the next state.
      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_EMPTY => {
        if !bit_reader::BrotliSafeReadBits(&mut s.br, 1, &mut bits, input) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if bits != 0 {
          s.substate_metablock_header =
            BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NONE;
          return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
        }
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NIBBLES;
        // No break, transit to the next state.
      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NIBBLES => {
        if !bit_reader::BrotliSafeReadBits(&mut s.br, 2, &mut bits, input) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        s.size_nibbles = (bits + 4) as u8;
        s.loop_counter = 0;
        if (bits == 3) {
          s.is_metadata = 1;
          s.substate_metablock_header =
            BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_RESERVED;
          continue;
        }
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_SIZE;
        // No break, transit to the next state.

      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_SIZE => {
        let mut i = s.loop_counter;
        while i < s.size_nibbles as i32 {
          if !bit_reader::BrotliSafeReadBits(&mut s.br, 4, &mut bits, input) {
            s.loop_counter = i;
            return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
          }
          if (i + 1 == s.size_nibbles as i32 && s.size_nibbles > 4 && bits == 0) {
            return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_EXUBERANT_NIBBLE;
          }
          s.meta_block_remaining_len |= (bits << (i * 4)) as i32;
          i += 1;
        }
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_UNCOMPRESSED;
        // No break, transit to the next state.
      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_UNCOMPRESSED => {
        if (s.is_last_metablock == 0 && s.is_metadata == 0) {
          if !bit_reader::BrotliSafeReadBits(&mut s.br, 1, &mut bits, input) {
            return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
          }
          s.is_uncompressed = bits as u8;
        }
        s.meta_block_remaining_len += 1;
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NONE;
        return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_RESERVED => {
        if !bit_reader::BrotliSafeReadBits(&mut s.br, 1, &mut bits, input) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if (bits != 0) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_RESERVED;
        }
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_BYTES;
        // No break, transit to the next state.
      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_BYTES => {
        if !bit_reader::BrotliSafeReadBits(&mut s.br, 2, &mut bits, input) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if (bits == 0) {
          s.substate_metablock_header =
            BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_NONE;
          return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
        }
        s.size_nibbles = bits as u8;
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_METADATA;
        // No break, transit to the next state.
      }
      BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_METADATA => {
        let mut i = s.loop_counter;
        while i < s.size_nibbles as i32 {
          if !bit_reader::BrotliSafeReadBits(&mut s.br, 8, &mut bits, input) {
            s.loop_counter = i;
            return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
          }
          if (i + 1 == s.size_nibbles as i32 && s.size_nibbles > 1 && bits == 0) {
            return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_EXUBERANT_META_NIBBLE;
          }
          s.meta_block_remaining_len |= (bits << (i * 8)) as i32;
          i += 1;
        }
        s.substate_metablock_header =
          BrotliRunningMetablockHeaderState::BROTLI_STATE_METABLOCK_HEADER_UNCOMPRESSED;
        continue;
      }
    }
  }
}
// Decodes the Huffman code.
// This method doesn't read data from the bit reader, BUT drops the amount of
// bits that correspond to the decoded symbol.
// bits MUST contain at least 15 (BROTLI_HUFFMAN_MAX_CODE_LENGTH) valid bits.
#[inline(always)]
fn DecodeSymbol(bits: u32, table: &[HuffmanCode], br: &mut bit_reader::BrotliBitReader) -> u32 {
  let mut table_index = bits & HUFFMAN_TABLE_MASK;
  let mut table_element = fast!((table)[table_index as usize]);
  if table_element.bits > HUFFMAN_TABLE_BITS as u8 {
    let nbits = table_element.bits - HUFFMAN_TABLE_BITS as u8;
    bit_reader::BrotliDropBits(br, HUFFMAN_TABLE_BITS);
    table_index += table_element.value as u32;
    table_element = fast!((table)[(table_index
                           + ((bits >> HUFFMAN_TABLE_BITS)
                              & bit_reader::BitMask(nbits as u32))) as usize]);
  }
  bit_reader::BrotliDropBits(br, table_element.bits as u32);
  table_element.value as u32
}

// Reads and decodes the next Huffman code from bit-stream.
// This method peeks 16 bits of input and drops 0 - 15 of them.
#[inline(always)]
fn ReadSymbol(table: &[HuffmanCode], br: &mut bit_reader::BrotliBitReader, input: &[u8]) -> u32 {
  DecodeSymbol(bit_reader::BrotliGet16BitsUnmasked(br, input), table, br)
}

// Same as DecodeSymbol, but it is known that there is less than 15 bits of
// input are currently available.
fn SafeDecodeSymbol(table: &[HuffmanCode],
                    mut br: &mut bit_reader::BrotliBitReader,
                    result: &mut u32)
                    -> bool {
  let mut available_bits = bit_reader::BrotliGetAvailableBits(br);
  if (available_bits == 0) {
    if (fast!((table)[0]).bits == 0) {
      *result = fast!((table)[0]).value as u32;
      return true;
    }
    return false; /* No valid bits at all. */
  }
  let mut val = bit_reader::BrotliGetBitsUnmasked(br) as u32;
  let table_index = (val & HUFFMAN_TABLE_MASK) as usize;
  let table_element = fast!((table)[table_index]);
  if (table_element.bits <= HUFFMAN_TABLE_BITS as u8) {
    if (table_element.bits as u32 <= available_bits) {
      bit_reader::BrotliDropBits(&mut br, table_element.bits as u32);
      *result = table_element.value as u32;
      return true;
    } else {
      return false; /* Not enough bits for the first level. */
    }
  }
  if (available_bits <= HUFFMAN_TABLE_BITS) {
    return false; /* Not enough bits to move to the second level. */
  }

  // Speculatively drop HUFFMAN_TABLE_BITS.
  val = (val & bit_reader::BitMask(table_element.bits as u32)) >> HUFFMAN_TABLE_BITS;
  available_bits -= HUFFMAN_TABLE_BITS;
  let table_sub_element = fast!((table)[table_index + table_element.value as usize + val as usize]);
  if (available_bits < table_sub_element.bits as u32) {
    return false; /* Not enough bits for the second level. */
  }

  bit_reader::BrotliDropBits(&mut br, HUFFMAN_TABLE_BITS + table_sub_element.bits as u32);
  *result = table_sub_element.value as u32;
  true
}

fn SafeReadSymbol(table: &[HuffmanCode],
                  br: &mut bit_reader::BrotliBitReader,
                  result: &mut u32,
                  input: &[u8])
                  -> bool {
  let mut val: u32 = 0;
  if (bit_reader::BrotliSafeGetBits(br, 15, &mut val, input)) {
    *result = DecodeSymbol(val, table, br);
    return true;
  } else {
    mark_unlikely();
  }
  SafeDecodeSymbol(table, br, result)
}

// Makes a look-up in first level Huffman table. Peeks 8 bits.
fn PreloadSymbol(safe: bool,
                 table: &[HuffmanCode],
                 br: &mut bit_reader::BrotliBitReader,
                 bits: &mut u32,
                 value: &mut u32,
                 input: &[u8]) {
  if (safe) {
    return;
  }
  let table_element =
    fast!((table)[bit_reader::BrotliGetBits(br, HUFFMAN_TABLE_BITS, input) as usize]);
  *bits = table_element.bits as u32;
  *value = table_element.value as u32;
}

// Decodes the next Huffman code using data prepared by PreloadSymbol.
// Reads 0 - 15 bits. Also peeks 8 following bits.
fn ReadPreloadedSymbol(table: &[HuffmanCode],
                       br: &mut bit_reader::BrotliBitReader,
                       bits: &mut u32,
                       value: &mut u32,
                       input: &[u8])
                       -> u32 {
  let result = if *bits > HUFFMAN_TABLE_BITS {
    mark_unlikely();
    let val = bit_reader::BrotliGet16BitsUnmasked(br, input);
    let mut ext_index = (val & HUFFMAN_TABLE_MASK) + *value;
    let mask = bit_reader::BitMask((*bits - HUFFMAN_TABLE_BITS));
    bit_reader::BrotliDropBits(br, HUFFMAN_TABLE_BITS);
    ext_index += (val >> HUFFMAN_TABLE_BITS) & mask;
    let ext = fast!((table)[ext_index as usize]);
    bit_reader::BrotliDropBits(br, ext.bits as u32);
    ext.value as u32
  } else {
    bit_reader::BrotliDropBits(br, *bits);
    *value
  };
  PreloadSymbol(false, table, br, bits, value, input);
  result
}

fn Log2Floor(mut x: u32) -> u32 {
  let mut result: u32 = 0;
  while x != 0 {
    x >>= 1;
    result += 1;
  }
  result
}


// Reads (s->symbol + 1) symbols.
// Totally 1..4 symbols are read, 1..11 bits each.
// The list of symbols MUST NOT contain duplicates.
//
fn ReadSimpleHuffmanSymbols<AllocU8: alloc::Allocator<u8>,
                            AllocU32: alloc::Allocator<u32>,
                            AllocHC: alloc::Allocator<HuffmanCode>>
  (alphabet_size: u32, max_symbol: u32,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {

  // max_bits == 1..11; symbol == 0..3; 1..44 bits will be read.
  let max_bits = Log2Floor(alphabet_size - 1);
  let mut i = s.sub_loop_counter;
  let num_symbols = s.symbol;
  for symbols_lists_item in fast_mut!((s.symbols_lists_array)[s.sub_loop_counter as usize;
                                                  num_symbols as usize + 1])
    .iter_mut() {
    let mut v: u32 = 0;
    if !bit_reader::BrotliSafeReadBits(&mut s.br, max_bits, &mut v, input) {
      mark_unlikely();
      s.sub_loop_counter = i;
      s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_READ;
      return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
    }
    if (v >= max_symbol) {
      return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_SIMPLE_HUFFMAN_ALPHABET;
    }
    *symbols_lists_item = v as u16;
    BROTLI_LOG_UINT!(v);
    i += 1;
  }
  i = 0;
  for symbols_list_item in fast!((s.symbols_lists_array)[0; num_symbols as usize]).iter() {
    for other_item in fast!((s.symbols_lists_array)[i as usize + 1 ; num_symbols as usize+ 1])
      .iter() {
      if (*symbols_list_item == *other_item) {
        return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_SIMPLE_HUFFMAN_SAME;
      }
    }
    i += 1;
  }
  BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS
}

// Process single decoded symbol code length:
// A) reset the repeat variable
// B) remember code length (if it is not 0)
// C) extend corredponding index-chain
// D) reduce the huffman space
// E) update the histogram
//
fn ProcessSingleCodeLength(code_len: u32,
                           symbol: &mut u32,
                           repeat: &mut u32,
                           space: &mut u32,
                           prev_code_len: &mut u32,
                           symbol_lists: &mut [u16],
                           symbol_list_index_offset: usize,
                           code_length_histo: &mut [u16],
                           next_symbol: &mut [i32]) {
  *repeat = 0;
  if (code_len != 0) {
    // code_len == 1..15
    // next_symbol may be negative, hence we have to supply offset to function
    fast_mut!((symbol_lists)[(symbol_list_index_offset as i32 +
                             fast_inner!((next_symbol)[code_len as usize])) as usize]) =
      (*symbol) as u16;
    fast_mut!((next_symbol)[code_len as usize]) = (*symbol) as i32;
    *prev_code_len = code_len;
    *space = space.wrapping_sub(32768 >> code_len);
    fast_mut!((code_length_histo)[code_len as usize]) += 1;
    BROTLI_LOG!("[ReadHuffmanCode] code_length[{:}]={:} histo[]={:}\n",
                *symbol, code_len, code_length_histo[code_len as usize]);
  }
  (*symbol) += 1;
}

// Process repeated symbol code length.
// A) Check if it is the extension of previous repeat sequence; if the decoded
// value is not kCodeLengthRepeatCode, then it is a new symbol-skip
// B) Update repeat variable
// C) Check if operation is feasible (fits alphapet)
// D) For each symbol do the same operations as in ProcessSingleCodeLength
//
// PRECONDITION: code_len == kCodeLengthRepeatCode or kCodeLengthRepeatCode + 1
//
fn ProcessRepeatedCodeLength(code_len: u32,
                             mut repeat_delta: u32,
                             alphabet_size: u32,
                             symbol: &mut u32,
                             repeat: &mut u32,
                             space: &mut u32,
                             prev_code_len: &mut u32,
                             repeat_code_len: &mut u32,
                             symbol_lists: &mut [u16],
                             symbol_lists_index: usize,
                             code_length_histo: &mut [u16],
                             next_symbol: &mut [i32]) {
  let old_repeat: u32;
  let extra_bits: u32;
  let new_len: u32;
  if (code_len == kCodeLengthRepeatCode) {
    extra_bits = 2;
    new_len = *prev_code_len
  } else {
    extra_bits = 3;
    new_len = 0
  }
  if (*repeat_code_len != new_len) {
    *repeat = 0;
    *repeat_code_len = new_len;
  }
  old_repeat = *repeat;
  if (*repeat > 0) {
    *repeat -= 2;
    *repeat <<= extra_bits;
  }
  *repeat += repeat_delta + 3;
  repeat_delta = *repeat - old_repeat;
  if (*symbol + repeat_delta > alphabet_size) {
    *symbol = alphabet_size;
    *space = 0xFFFFF;
    return;
  }
  BROTLI_LOG!("[ReadHuffmanCode] code_length[{:}..{:}] = {:}\n",
              *symbol, *symbol + repeat_delta - 1, *repeat_code_len);
  if (*repeat_code_len != 0) {
    let last: u32 = *symbol + repeat_delta;
    let mut next: i32 = fast!((next_symbol)[*repeat_code_len as usize]);
    loop {
      fast_mut!((symbol_lists)[(symbol_lists_index as i32 + next) as usize]) = (*symbol) as u16;
      next = (*symbol) as i32;
      (*symbol) += 1;
      if *symbol == last {
        break;
      }
    }
    fast_mut!((next_symbol)[*repeat_code_len as usize]) = next;
    *space = space.wrapping_sub(repeat_delta << (15 - *repeat_code_len));
    fast_mut!((code_length_histo)[*repeat_code_len as usize]) =
      (fast!((code_length_histo)[*repeat_code_len as usize]) as u32 + repeat_delta) as u16;
  } else {
    *symbol += repeat_delta;
  }
}

// Reads and decodes symbol codelengths.
fn ReadSymbolCodeLengths<AllocU8: alloc::Allocator<u8>,
                         AllocU32: alloc::Allocator<u32>,
                         AllocHC: alloc::Allocator<HuffmanCode>>
  (alphabet_size: u32,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {

  let mut symbol = s.symbol;
  let mut repeat = s.repeat;
  let mut space = s.space;
  let mut prev_code_len: u32 = s.prev_code_len;
  let mut repeat_code_len: u32 = s.repeat_code_len;
  if (!bit_reader::BrotliWarmupBitReader(&mut s.br, input)) {
    return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
  }
  while (symbol < alphabet_size && space > 0) {
    let mut p_index = 0;
    let code_len: u32;
    if (!bit_reader::BrotliCheckInputAmount(&s.br, bit_reader::BROTLI_SHORT_FILL_BIT_WINDOW_READ)) {
      s.symbol = symbol;
      s.repeat = repeat;
      s.prev_code_len = prev_code_len;
      s.repeat_code_len = repeat_code_len;
      s.space = space;
      return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
    }
    bit_reader::BrotliFillBitWindow16(&mut s.br, input);
    p_index +=
      bit_reader::BrotliGetBitsUnmasked(&s.br) &
      bit_reader::BitMask(huffman::BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH as u32) as u64;
    let p = fast!((s.table)[p_index as usize]);
    bit_reader::BrotliDropBits(&mut s.br, p.bits as u32); /* Use 1..5 bits */
    code_len = p.value as u32; /* code_len == 0..17 */
    if (code_len < kCodeLengthRepeatCode) {
      ProcessSingleCodeLength(code_len,
                              &mut symbol,
                              &mut repeat,
                              &mut space,
                              &mut prev_code_len,
                              &mut s.symbols_lists_array,
                              s.symbol_lists_index as usize,
                              &mut s.code_length_histo[..],
                              &mut s.next_symbol[..]);
    } else {
      // code_len == 16..17, extra_bits == 2..3
      let extra_bits: u32 = if code_len == kCodeLengthRepeatCode {
        2
      } else {
        3
      };
      let repeat_delta: u32 = bit_reader::BrotliGetBitsUnmasked(&s.br) as u32 &
                              bit_reader::BitMask(extra_bits);
      bit_reader::BrotliDropBits(&mut s.br, extra_bits);
      ProcessRepeatedCodeLength(code_len,
                                repeat_delta,
                                alphabet_size,
                                &mut symbol,
                                &mut repeat,
                                &mut space,
                                &mut prev_code_len,
                                &mut repeat_code_len,
                                &mut s.symbols_lists_array,
                                s.symbol_lists_index as usize,
                                &mut s.code_length_histo[..],
                                &mut s.next_symbol[..]);
    }
  }
  s.space = space;
  BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS
}

fn SafeReadSymbolCodeLengths<AllocU8: alloc::Allocator<u8>,
                             AllocU32: alloc::Allocator<u32>,
                             AllocHC: alloc::Allocator<HuffmanCode>>
  (alphabet_size: u32,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  while (s.symbol < alphabet_size && s.space > 0) {
    let mut p_index = 0;
    let code_len: u32;
    let mut bits: u32 = 0;
    let available_bits: u32 = bit_reader::BrotliGetAvailableBits(&s.br);
    if (available_bits != 0) {
      bits = bit_reader::BrotliGetBitsUnmasked(&s.br) as u32;
    }
    p_index += bits &
               bit_reader::BitMask(huffman::BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH as u32);
    let p = fast!((s.table)[p_index as usize]);
    if (p.bits as u32 > available_bits) {
      // pullMoreInput;
      if (!bit_reader::BrotliPullByte(&mut s.br, input)) {
        return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
      }
      continue;
    }
    code_len = p.value as u32; /* code_len == 0..17 */
    if (code_len < kCodeLengthRepeatCode) {
      bit_reader::BrotliDropBits(&mut s.br, p.bits as u32);
      ProcessSingleCodeLength(code_len,
                              &mut s.symbol,
                              &mut s.repeat,
                              &mut s.space,
                              &mut s.prev_code_len,
                              &mut s.symbols_lists_array,
                              s.symbol_lists_index as usize,
                              &mut s.code_length_histo[..],
                              &mut s.next_symbol[..]);
    } else {
      // code_len == 16..17, extra_bits == 2..3
      let extra_bits: u32 = code_len - 14;
      let repeat_delta: u32 = (bits >> p.bits) & bit_reader::BitMask(extra_bits);
      if (available_bits < p.bits as u32 + extra_bits) {
        // pullMoreInput;
        if (!bit_reader::BrotliPullByte(&mut s.br, input)) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        continue;
      }
      bit_reader::BrotliDropBits(&mut s.br, p.bits as u32 + extra_bits);
      ProcessRepeatedCodeLength(code_len,
                                repeat_delta,
                                alphabet_size,
                                &mut s.symbol,
                                &mut s.repeat,
                                &mut s.space,
                                &mut s.prev_code_len,
                                &mut s.repeat_code_len,
                                &mut s.symbols_lists_array,
                                s.symbol_lists_index as usize,
                                &mut s.code_length_histo[..],
                                &mut s.next_symbol[..]);
    }
  }
  BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS
}

// Reads and decodes 15..18 codes using static prefix code.
// Each code is 2..4 bits long. In total 30..72 bits are used.
fn ReadCodeLengthCodeLengths<AllocU8: alloc::Allocator<u8>,
                             AllocU32: alloc::Allocator<u32>,
                             AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {

  let mut num_codes: u32 = s.repeat;
  let mut space: u32 = s.space;
  let mut i = s.sub_loop_counter;
  for code_length_code_order in
      fast!((kCodeLengthCodeOrder)[s.sub_loop_counter as usize; CODE_LENGTH_CODES]).iter() {
    let code_len_idx = *code_length_code_order;
    let mut ix: u32 = 0;

    if !bit_reader::BrotliSafeGetBits(&mut s.br, 4, &mut ix, input) {
      mark_unlikely();
      let available_bits: u32 = bit_reader::BrotliGetAvailableBits(&s.br);
      if (available_bits != 0) {
        ix = bit_reader::BrotliGetBitsUnmasked(&s.br) as u32 & 0xF;
      } else {
        ix = 0;
      }
      if (fast!((kCodeLengthPrefixLength)[ix as usize]) as u32 > available_bits) {
        s.sub_loop_counter = i;
        s.repeat = num_codes;
        s.space = space;
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_COMPLEX;
        return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
      }
    }
    BROTLI_LOG_UINT!(ix);
    let v: u32 = fast!((kCodeLengthPrefixValue)[ix as usize]) as u32;
    bit_reader::BrotliDropBits(&mut s.br,
                               fast!((kCodeLengthPrefixLength)[ix as usize]) as u32);
    fast_mut!((s.code_length_code_lengths)[code_len_idx as usize]) = v as u8;
    BROTLI_LOG_ARRAY_INDEX!(s.code_length_code_lengths, code_len_idx);
    if v != 0 {
      space = space.wrapping_sub(32 >> v);
      num_codes += 1;
      fast_mut!((s.code_length_histo)[v as usize]) += 1;
      if space.wrapping_sub(1) >= 32 {
        // space is 0 or wrapped around
        break;
      }
    }
    i += 1;
  }
  if (!(num_codes == 1 || space == 0)) {
    return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_CL_SPACE;
  }
  BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS
}


// Decodes the Huffman tables.
// There are 2 scenarios:
// A) Huffman code contains only few symbols (1..4). Those symbols are read
// directly; their code lengths are defined by the number of symbols.
// For this scenario 4 - 49 bits will be read.
//
// B) 2-phase decoding:
// B.1) Small Huffman table is decoded; it is specified with code lengths
// encoded with predefined entropy code. 32 - 74 bits are used.
// B.2) Decoded table is used to decode code lengths of symbols in resulting
// Huffman table. In worst case 3520 bits are read.
//
fn ReadHuffmanCode<AllocU8: alloc::Allocator<u8>,
                   AllocU32: alloc::Allocator<u32>,
                   AllocHC: alloc::Allocator<HuffmanCode>>
  (mut alphabet_size: u32,
   max_symbol: u32,
   table: &mut [HuffmanCode],
   offset: usize,
   opt_table_size: Option<&mut u32>,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  // Unnecessary masking, but might be good for safety.
  alphabet_size &= 0x7ff;
  // State machine
  loop {
    match s.substate_huffman {
      BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_NONE => {
        if !bit_reader::BrotliSafeReadBits(&mut s.br, 2, &mut s.sub_loop_counter, input) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }

        BROTLI_LOG_UINT!(s.sub_loop_counter);
        // The value is used as follows:
        // 1 for simple code;
        // 0 for no skipping, 2 skips 2 code lengths, 3 skips 3 code lengths
        if (s.sub_loop_counter != 1) {
          s.space = 32;
          s.repeat = 0; /* num_codes */
          let max_code_len_len = huffman::BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH as usize + 1;
          for code_length_histo in fast_mut!((s.code_length_histo)[0;max_code_len_len]).iter_mut() {
            *code_length_histo = 0; // memset
          }
          for code_length_code_length in s.code_length_code_lengths[..].iter_mut() {
            *code_length_code_length = 0;
          }
          s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_COMPLEX;
          // goto Complex;
          continue;
        }
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_SIZE;
        // No break, transit to the next state.
      }
      BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_SIZE => {
        // Read symbols, codes & code lengths directly.
        if (!bit_reader::BrotliSafeReadBits(&mut s.br, 2, &mut s.symbol, input)) {
          // num_symbols
          s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_SIZE;
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        s.sub_loop_counter = 0;
        // No break, transit to the next state.
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_READ;
      }
      BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_READ => {
        let result = ReadSimpleHuffmanSymbols(alphabet_size, max_symbol, s, input);
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
          _ => return result,
        }
        // No break, transit to the next state.
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_BUILD;
      }
      BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_BUILD => {
        let table_size: u32;
        if (s.symbol == 3) {
          let mut bits: u32 = 0;
          if (!bit_reader::BrotliSafeReadBits(&mut s.br, 1, &mut bits, input)) {
            s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_SIMPLE_BUILD;
            return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
          }
          s.symbol += bits;
        }
        BROTLI_LOG_UINT!(s.symbol);
        table_size = huffman::BrotliBuildSimpleHuffmanTable(&mut table[offset..],
                                                            HUFFMAN_TABLE_BITS as i32,
                                                            &s.symbols_lists_array[..],
                                                            s.symbol);
        if let Some(opt_table_size_ref) = opt_table_size {
          *opt_table_size_ref = table_size
        }
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_NONE;
        return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
      }

      // Decode Huffman-coded code lengths.
      BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_COMPLEX => {

        let result = ReadCodeLengthCodeLengths(s, input);
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
          _ => return result,
        }
        huffman::BrotliBuildCodeLengthsHuffmanTable(&mut s.table,
                                                    &s.code_length_code_lengths,
                                                    &s.code_length_histo);
        for code_length_histo in s.code_length_histo[..].iter_mut() {
          *code_length_histo = 0; // memset
        }

        let max_code_length = huffman::BROTLI_HUFFMAN_MAX_CODE_LENGTH as usize + 1;
        for (i, next_symbol_mut) in fast_mut!((s.next_symbol)[0; max_code_length])
          .iter_mut()
          .enumerate() {
          *next_symbol_mut = i as i32 - (max_code_length as i32);
          fast_mut!((s.symbols_lists_array)[(s.symbol_lists_index as i32
                                 + i as i32
                                 - (max_code_length as i32)) as usize]) = 0xFFFF;
        }

        s.symbol = 0;
        s.prev_code_len = kDefaultCodeLength;
        s.repeat = 0;
        s.repeat_code_len = 0;
        s.space = 32768;
        // No break, transit to the next state.
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_LENGTH_SYMBOLS;
      }
      BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_LENGTH_SYMBOLS => {
        let table_size: u32;
        let mut result = ReadSymbolCodeLengths(max_symbol, s, input);
        if let BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT = result {
          result = SafeReadSymbolCodeLengths(max_symbol, s, input)
        }
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
          _ => return result,
        }

        if (s.space != 0) {
          BROTLI_LOG!("[ReadHuffmanCode] space = %d\n", s.space);
          return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_HUFFMAN_SPACE;
        }
        table_size = huffman::BrotliBuildHuffmanTable(fast_mut!((table)[offset;]),
                                                      HUFFMAN_TABLE_BITS as i32,
                                                      &s.symbols_lists_array[..],
                                                      s.symbol_lists_index,
                                                      &mut s.code_length_histo);
        if let Some(opt_table_size_ref) = opt_table_size {
          *opt_table_size_ref = table_size
        }
        s.substate_huffman = BrotliRunningHuffmanState::BROTLI_STATE_HUFFMAN_NONE;
        return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
      }
    }
  }
}

// Decodes a block length by reading 3..39 bits.
fn ReadBlockLength(table: &[HuffmanCode],
                   br: &mut bit_reader::BrotliBitReader,
                   input: &[u8])
                   -> u32 {
  let code: u32;
  let nbits: u32;
  code = ReadSymbol(table, br, input);
  nbits = fast_ref!((prefix::kBlockLengthPrefixCode)[code as usize]).nbits as u32; /*nbits==2..24*/
  fast_ref!((prefix::kBlockLengthPrefixCode)[code as usize]).offset as u32 +
  bit_reader::BrotliReadBits(br, nbits, input)
}


// WARNING: if state is not BROTLI_STATE_READ_BLOCK_LENGTH_NONE, then
// reading can't be continued with ReadBlockLength.
fn SafeReadBlockLengthIndex(substate_read_block_length: &state::BrotliRunningReadBlockLengthState,
                            block_length_index: u32,
                            table: &[HuffmanCode],
                            mut br: &mut bit_reader::BrotliBitReader,
                            input: &[u8])
                            -> (bool, u32) {
  match *substate_read_block_length {
    state::BrotliRunningReadBlockLengthState::BROTLI_STATE_READ_BLOCK_LENGTH_NONE => {
      let mut index: u32 = 0;
      if (!SafeReadSymbol(table, &mut br, &mut index, input)) {
        return (false, 0);
      }
      (true, index)
    }
    _ => (true, block_length_index),
  }
}
fn SafeReadBlockLengthFromIndex<
    AllocHC : alloc::Allocator<HuffmanCode> >(s : &mut BlockTypeAndLengthState<AllocHC>,
                                              br : &mut bit_reader::BrotliBitReader,
                                              result : &mut u32,
                                              res_index : (bool, u32),
                                              input : &[u8]) -> bool{
  let (res, index) = res_index;
  if !res {
    return false;
  }
  let mut bits: u32 = 0;
  let nbits = fast_ref!((prefix::kBlockLengthPrefixCode)[index as usize]).nbits; /* nbits==2..24 */
  if (!bit_reader::BrotliSafeReadBits(br, nbits as u32, &mut bits, input)) {
    s.block_length_index = index;
    s.substate_read_block_length =
      state::BrotliRunningReadBlockLengthState::BROTLI_STATE_READ_BLOCK_LENGTH_SUFFIX;
    return false;
  }
  *result = fast_ref!((prefix::kBlockLengthPrefixCode)[index as usize]).offset as u32 + bits;
  s.substate_read_block_length =
    state::BrotliRunningReadBlockLengthState::BROTLI_STATE_READ_BLOCK_LENGTH_NONE;
  true
}
macro_rules! SafeReadBlockLength (
   ($state : expr, $result : expr , $table : expr) => {
       SafeReadBlockLengthFromIndex(&mut $state, &mut $result,
                                    SafeReadBlockLengthIndex($state.substate_read_block_length,
                                                             $state.block_length_index,
                                                             $table,
                                                             &mut $state.br))
   };
);

// Transform:
// 1) initialize list L with values 0, 1,... 255
// 2) For each input element X:
// 2.1) let Y = L[X]
// 2.2) remove X-th element from L
// 2.3) prepend Y to L
// 2.4) append Y to output
//
// In most cases max(Y) <= 7, so most of L remains intact.
// To reduce the cost of initialization, we reuse L, remember the upper bound
// of Y values, and reinitialize only first elements in L.
//
// Most of input values are 0 and 1. To reduce number of branches, we replace
// inner for loop with do-while.
//
fn InverseMoveToFrontTransform(v: &mut [u8],
                               v_len: u32,
                               mtf: &mut [u8;256],
                               mtf_upper_bound: &mut u32) {
  // Reinitialize elements that could have been changed.
  let mut upper_bound: u32 = *mtf_upper_bound;
  for (i, item) in fast_mut!((mtf)[0;(upper_bound as usize + 1usize)]).iter_mut().enumerate() {
    *item = i as u8;
  }

  // Transform the input.
  upper_bound = 0;
  for v_i in fast_mut!((v)[0usize ; (v_len as usize)]).iter_mut() {
    let mut index = (*v_i) as i32;
    let value = fast!((mtf)[index as usize]);
    upper_bound |= (*v_i) as u32;
    *v_i = value;
    if index <= 0 {
      fast_mut!((mtf)[0]) = 0;
    } else {
      loop {
        index -= 1;
        fast_mut!((mtf)[(index + 1) as usize]) = fast!((mtf)[index as usize]);
        if index <= 0 {
          break;
        }
      }
    }
    fast_mut!((mtf)[0]) = value;
  }
  // Remember amount of elements to be reinitialized.
  *mtf_upper_bound = upper_bound;
}
// Decodes a series of Huffman table using ReadHuffmanCode function.
fn HuffmanTreeGroupDecode<AllocU8: alloc::Allocator<u8>,
                          AllocU32: alloc::Allocator<u32>,
                          AllocHC: alloc::Allocator<HuffmanCode>>
  (group_index: i32,
   mut s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  let mut hcodes: AllocHC::AllocatedMemory;
  let mut htrees: AllocU32::AllocatedMemory;
  let alphabet_size: u16;
  let group_num_htrees: u16;
  let group_max_symbol;
  if group_index == 0 {
    hcodes = mem::replace(&mut s.literal_hgroup.codes,
                          AllocHC::AllocatedMemory::default());
    htrees = mem::replace(&mut s.literal_hgroup.htrees,
                          AllocU32::AllocatedMemory::default());
    group_num_htrees = s.literal_hgroup.num_htrees;
    alphabet_size = s.literal_hgroup.alphabet_size;
    group_max_symbol = s.literal_hgroup.max_symbol;
  } else if group_index == 1 {
    hcodes = mem::replace(&mut s.insert_copy_hgroup.codes,
                          AllocHC::AllocatedMemory::default());
    htrees = mem::replace(&mut s.insert_copy_hgroup.htrees,
                          AllocU32::AllocatedMemory::default());
    group_num_htrees = s.insert_copy_hgroup.num_htrees;
    alphabet_size = s.insert_copy_hgroup.alphabet_size;
    group_max_symbol = s.insert_copy_hgroup.max_symbol;
  } else {
    if group_index != 2 {
      let ret = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_UNREACHABLE;
      SaveErrorCode!(s, ret);
      return ret;
    }
    hcodes = mem::replace(&mut s.distance_hgroup.codes,
                          AllocHC::AllocatedMemory::default());
    htrees = mem::replace(&mut s.distance_hgroup.htrees,
                          AllocU32::AllocatedMemory::default());
    group_num_htrees = s.distance_hgroup.num_htrees;
    alphabet_size = s.distance_hgroup.alphabet_size;
    group_max_symbol = s.distance_hgroup.max_symbol;
  }
  match s.substate_tree_group {
    BrotliRunningTreeGroupState::BROTLI_STATE_TREE_GROUP_NONE => {
      s.htree_next_offset = 0;
      s.htree_index = 0;
      s.substate_tree_group = BrotliRunningTreeGroupState::BROTLI_STATE_TREE_GROUP_LOOP;
    }
    BrotliRunningTreeGroupState::BROTLI_STATE_TREE_GROUP_LOOP => {}
  }
  let mut result = BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
  for htree_iter in
      fast_mut!((htrees.slice_mut())[s.htree_index as usize ; (group_num_htrees as usize)])
    .iter_mut() {
    let mut table_size: u32 = 0;
    result = ReadHuffmanCode(u32::from(alphabet_size), u32::from(group_max_symbol),
                             hcodes.slice_mut(),
                             s.htree_next_offset as usize,
                             Some(&mut table_size),
                             &mut s,
                             input);
    match result {
      BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
      _ => break, // break and return the result code
    }
    *htree_iter = s.htree_next_offset;
    s.htree_next_offset += table_size;
    s.htree_index += 1;
  }
  if group_index == 0 {
    let _ = mem::replace(&mut s.literal_hgroup.codes,
                 mem::replace(&mut hcodes, AllocHC::AllocatedMemory::default()));
    let _ = mem::replace(&mut s.literal_hgroup.htrees,
                 mem::replace(&mut htrees, AllocU32::AllocatedMemory::default()));
  } else if group_index == 1 {
    let _ = mem::replace(&mut s.insert_copy_hgroup.codes,
                 mem::replace(&mut hcodes, AllocHC::AllocatedMemory::default()));
    let _ = mem::replace(&mut s.insert_copy_hgroup.htrees,
                 mem::replace(&mut htrees, AllocU32::AllocatedMemory::default()));
  } else {
    let _ = mem::replace(&mut s.distance_hgroup.codes,
                 mem::replace(&mut hcodes, AllocHC::AllocatedMemory::default()));
    let _ = mem::replace(&mut s.distance_hgroup.htrees,
                 mem::replace(&mut htrees, AllocU32::AllocatedMemory::default()));
  }
  if let BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS = result {
    s.substate_tree_group = BrotliRunningTreeGroupState::BROTLI_STATE_TREE_GROUP_NONE
  }
  result
}
#[allow(dead_code)]
pub fn lg_window_size(first_byte: u8, second_byte: u8) -> Result<(u8, u8), ()> {
  if first_byte & 1 == 0 {
    return Ok((16, 1));
  }
  match first_byte & 15 {
    0x3 => return Ok((18, 4)),
    0x5 => return Ok((19, 4)),
    0x7 => return Ok((20, 4)),
    0x9 => return Ok((21, 4)),
    0xb => return Ok((22, 4)),
    0xd => return Ok((23, 4)),
    0xf => return Ok((24, 4)),
    _ => match first_byte & 127 {
      0x71 => return Ok((15, 7)),
      0x61 => return Ok((14, 7)),
      0x51 => return Ok((13, 7)),
      0x41 => return Ok((12, 7)),
      0x31 => return Ok((11, 7)),
      0x21 => return Ok((10, 7)),
      0x1 => return Ok((17, 7)),
      _ => {},
    }
  }
  if (first_byte & 0x80) != 0 {
    return Err(());
  }
  let ret  = second_byte & 0x3f;
  if ret < 10 || ret > 30 {
    return Err(());
  }
  Ok((ret, 14))

}


fn bzero(data: &mut [u8]) {
  for iter in data.iter_mut() {
    *iter = 0;
  }
}


// Decodes a context map.
// Decoding is done in 4 phases:
// 1) Read auxiliary information (6..16 bits) and allocate memory.
// In case of trivial context map, decoding is finished at this phase.
// 2) Decode Huffman table using ReadHuffmanCode function.
// This table will be used for reading context map items.
// 3) Read context map items; "0" values could be run-length encoded.
// 4) Optionally, apply InverseMoveToFront transform to the resulting map.
//
fn DecodeContextMapInner<AllocU8: alloc::Allocator<u8>,
                         AllocU32: alloc::Allocator<u32>,
                         AllocHC: alloc::Allocator<HuffmanCode>>
  (context_map_size: u32,
   num_htrees: &mut u32,
   context_map_arg: &mut AllocU8::AllocatedMemory,
   mut s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {

  let mut result;
  loop {
    match s.substate_context_map {
      BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_NONE => {
        result = DecodeVarLenUint8(&mut s.substate_decode_uint8, &mut s.br, num_htrees, input);
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
          _ => return result,
        }
        (*num_htrees) += 1;
        s.context_index = 0;
        BROTLI_LOG_UINT!(context_map_size);
        BROTLI_LOG_UINT!(*num_htrees);
        *context_map_arg = s.alloc_u8.alloc_cell(context_map_size as usize);
        if (context_map_arg.slice().len() < context_map_size as usize) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_ALLOC_CONTEXT_MAP;
        }
        if (*num_htrees <= 1) {
          // This happens automatically but we do it to retain C++ similarity:
          bzero(context_map_arg.slice_mut()); // necessary if we compiler with unsafe feature flag
          return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
        }
        s.substate_context_map = BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_READ_PREFIX;
        // No break, continue to next state.
      }
      BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_READ_PREFIX => {
        let mut bits: u32 = 0;
        // In next stage ReadHuffmanCode uses at least 4 bits, so it is safe
        // to peek 4 bits ahead.
        if (!bit_reader::BrotliSafeGetBits(&mut s.br, 5, &mut bits, input)) {
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if ((bits & 1) != 0) {
          // Use RLE for zeroes.
          s.max_run_length_prefix = (bits >> 1) + 1;
          bit_reader::BrotliDropBits(&mut s.br, 5);
        } else {
          s.max_run_length_prefix = 0;
          bit_reader::BrotliDropBits(&mut s.br, 1);
        }
        BROTLI_LOG_UINT!(s.max_run_length_prefix);
        s.substate_context_map = BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_HUFFMAN;
        // No break, continue to next state.
      }
      BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_HUFFMAN => {

        let mut local_context_map_table = mem::replace(&mut s.context_map_table,
                                                       AllocHC::AllocatedMemory::default());
        let alphabet_size = *num_htrees + s.max_run_length_prefix;
        result = ReadHuffmanCode(alphabet_size, alphabet_size,
                                 &mut local_context_map_table.slice_mut(),
                                 0,
                                 None,
                                 &mut s,
                                 input);
        let _ = mem::replace(&mut s.context_map_table,
                     mem::replace(&mut local_context_map_table,
                                  AllocHC::AllocatedMemory::default()));
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
          _ => return result,
        }
        s.code = 0xFFFF;
        s.substate_context_map = BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_DECODE;
        // No break, continue to next state.
      }
      BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_DECODE => {
        let mut context_index: u32 = s.context_index;
        let max_run_length_prefix: u32 = s.max_run_length_prefix;
        let context_map = &mut context_map_arg.slice_mut();
        let mut code: u32 = s.code;
        let mut rleCodeGoto = (code != 0xFFFF);
        while (rleCodeGoto || context_index < context_map_size) {
          if !rleCodeGoto {
            if (!SafeReadSymbol(s.context_map_table.slice(), &mut s.br, &mut code, input)) {
              s.code = 0xFFFF;
              s.context_index = context_index;
              return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            }
            BROTLI_LOG_UINT!(code);

            if code == 0 {
              fast_mut!((context_map)[context_index as usize]) = 0;
              BROTLI_LOG_ARRAY_INDEX!(context_map, context_index as usize);
              context_index += 1;
              continue;
            }
            if code > max_run_length_prefix {
              fast_mut!((context_map)[context_index as usize]) =
                (code - max_run_length_prefix) as u8;
              BROTLI_LOG_ARRAY_INDEX!(context_map, context_index as usize);
              context_index += 1;
              continue;
            }
          }
          rleCodeGoto = false; // <- this was a goto beforehand... now we have reached label
          // we are treated like everyday citizens from this point forth
          {
            let mut reps: u32 = 0;
            if (!bit_reader::BrotliSafeReadBits(&mut s.br, code, &mut reps, input)) {
              s.code = code;
              s.context_index = context_index;
              return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            }
            reps += 1u32 << code;
            BROTLI_LOG_UINT!(reps);
            if (context_index + reps > context_map_size) {
              return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_CONTEXT_MAP_REPEAT;
            }
            loop {
              fast_mut!((context_map)[context_index as usize]) = 0;
              BROTLI_LOG_ARRAY_INDEX!(context_map, context_index as usize);
              context_index += 1;
              reps -= 1;
              if reps == 0 {
                break;
              }
            }
          }
        }
        // No break, continue to next state.
        s.substate_context_map = BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_TRANSFORM;
      }
      BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_TRANSFORM => {
        let mut bits: u32 = 0;
        if (!bit_reader::BrotliSafeReadBits(&mut s.br, 1, &mut bits, input)) {
          s.substate_context_map = BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_TRANSFORM;
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        if (bits != 0) {
          if let Ok(ref mut mtf) = s.mtf_or_error_string {
            InverseMoveToFrontTransform(context_map_arg.slice_mut(),
                                        context_map_size,
                                        mtf,
                                        &mut s.mtf_upper_bound);
          } else {
            // the error state is stored here--we can't make it this deep with an active error
            return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_UNREACHABLE;
          }
        }
        s.substate_context_map = BrotliRunningContextMapState::BROTLI_STATE_CONTEXT_MAP_NONE;
        return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
      }
    }
  }
  // unreachable!(); (compiler will error if it's reachable due to the unit return type)
}

fn DecodeContextMap<AllocU8: alloc::Allocator<u8>,
                    AllocU32: alloc::Allocator<u32>,
                    AllocHC: alloc::Allocator<HuffmanCode>>
  (context_map_size: usize,
   is_dist_context_map: bool,
   mut s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {

  match s.state {
    BrotliRunningState::BROTLI_STATE_CONTEXT_MAP_1 => assert_eq!(is_dist_context_map, false),
    BrotliRunningState::BROTLI_STATE_CONTEXT_MAP_2 => assert_eq!(is_dist_context_map, true),
    _ => unreachable!(),
  }
  let (mut num_htrees, mut context_map_arg) = if is_dist_context_map {
    (s.num_dist_htrees, mem::replace(&mut s.dist_context_map, AllocU8::AllocatedMemory::default()))
  } else {
    (s.num_literal_htrees, mem::replace(&mut s.context_map, AllocU8::AllocatedMemory::default()))
  };

  let retval = DecodeContextMapInner(context_map_size as u32,
                                     &mut num_htrees,
                                     &mut context_map_arg,
                                     &mut s,
                                     input);
  if is_dist_context_map {
    s.num_dist_htrees = num_htrees;
    let _ = mem::replace(&mut s.dist_context_map,
                 mem::replace(&mut context_map_arg, AllocU8::AllocatedMemory::default()));
  } else {
    s.num_literal_htrees = num_htrees;
    let _ = mem::replace(&mut s.context_map,
                 mem::replace(&mut context_map_arg, AllocU8::AllocatedMemory::default()));
  }
  retval
}

// Decodes a command or literal and updates block type ringbuffer.
// Reads 3..54 bits.
fn DecodeBlockTypeAndLength<
  AllocHC : alloc::Allocator<HuffmanCode>> (safe : bool,
                                            s : &mut BlockTypeAndLengthState<AllocHC>,
                                            br : &mut bit_reader::BrotliBitReader,
                                            tree_type : i32,
                                            input : &[u8]) -> bool {
  let max_block_type = fast!((s.num_block_types)[tree_type as usize]);
  let tree_offset = tree_type as usize * huffman::BROTLI_HUFFMAN_MAX_TABLE_SIZE as usize;

  let mut block_type: u32 = 0;
  if max_block_type <= 1 {
    return false;
  }
  // Read 0..15 + 3..39 bits
  if (!safe) {
    block_type = ReadSymbol(fast_slice!((s.block_type_trees)[tree_offset;]), br, input);
    fast_mut!((s.block_length)[tree_type as usize]) =
      ReadBlockLength(fast_slice!((s.block_len_trees)[tree_offset;]), br, input);
  } else {
    let memento = bit_reader::BrotliBitReaderSaveState(br);
    if (!SafeReadSymbol(fast_slice!((s.block_type_trees)[tree_offset;]),
                        br,
                        &mut block_type,
                        input)) {
      return false;
    }
    let mut block_length_out: u32 = 0;

    let index_ret = SafeReadBlockLengthIndex(&s.substate_read_block_length,
                                             s.block_length_index,
                                             fast_slice!((s.block_len_trees)[tree_offset;]),
                                             br,
                                             input);
    if !SafeReadBlockLengthFromIndex(s, br, &mut block_length_out, index_ret, input) {
      s.substate_read_block_length =
        BrotliRunningReadBlockLengthState::BROTLI_STATE_READ_BLOCK_LENGTH_NONE;
      bit_reader::BrotliBitReaderRestoreState(br, &memento);
      return false;
    }
    fast_mut!((s.block_length)[tree_type as usize]) = block_length_out;
  }
  let ringbuffer: &mut [u32] = &mut fast_mut!((s.block_type_rb)[tree_type as usize * 2;]);
  if (block_type == 1) {
    block_type = fast!((ringbuffer)[1]) + 1;
  } else if (block_type == 0) {
    block_type = fast!((ringbuffer)[0]);
  } else {
    block_type -= 2;
  }
  if (block_type >= max_block_type) {
    block_type -= max_block_type;
  }
  fast_mut!((ringbuffer)[0]) = fast!((ringbuffer)[1]);
  fast_mut!((ringbuffer)[1]) = block_type;
  true
}
fn DetectTrivialLiteralBlockTypes<AllocU8: alloc::Allocator<u8>,
                                  AllocU32: alloc::Allocator<u32>,
                                  AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>) {
  for iter in s.trivial_literal_contexts.iter_mut() {
    *iter = 0;
  }
  let mut i: usize = 0;
  while i < fast!((s.block_type_length_state.num_block_types)[0]) as usize {
    let offset = (i as usize) << kLiteralContextBits;
    let mut error = 0usize;
    let sample: usize = fast_slice!((s.context_map)[offset]) as usize;
    let mut j = 0usize;
    while j < ((1 as usize) << kLiteralContextBits) {
      error |= fast_slice!((s.context_map)[offset + j]) as usize ^ sample;
      j += 1;
      error |= fast_slice!((s.context_map)[offset + j]) as usize ^ sample;
      j += 1;
      error |= fast_slice!((s.context_map)[offset + j]) as usize ^ sample;
      j += 1;
      error |= fast_slice!((s.context_map)[offset + j]) as usize ^ sample;
      j += 1
    }
    if error == 0 {
      s.trivial_literal_contexts[i >> 5] |= ((1 as u32) << (i & 31));
    }
    i += 1
  }
}
fn PrepareLiteralDecoding<AllocU8: alloc::Allocator<u8>,
                          AllocU32: alloc::Allocator<u32>,
                          AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>) {

  let context_offset: u32;
  let block_type = fast!((s.block_type_length_state.block_type_rb)[1]) as usize;
  context_offset = (block_type << kLiteralContextBits) as u32;
  s.context_map_slice_index = context_offset as usize;
  let trivial = fast!((s.trivial_literal_contexts)[block_type >> 5]);
  s.trivial_literal_context = ((trivial >> (block_type & 31)) & 1) as i32;

  s.literal_htree_index = fast_slice!((s.context_map)[s.context_map_slice_index]);
  // s.literal_htree = fast!((s.literal_hgroup.htrees)[s.literal_htree_index]); // redundant
  let context_mode_index = fast!((s.context_modes.slice())[block_type]) & 3;
  s.context_lookup = &kContextLookup[context_mode_index as usize];
}

// Decodes the block ty
// pe and updates the state for literal context.
// Reads 3..54 bits.
fn DecodeLiteralBlockSwitchInternal<AllocU8: alloc::Allocator<u8>,
                                    AllocU32: alloc::Allocator<u32>,
                                    AllocHC: alloc::Allocator<HuffmanCode>>
  (safe: bool,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> bool {

  if !DecodeBlockTypeAndLength(safe, &mut s.block_type_length_state, &mut s.br, 0, input) {
    return false;
  }
  PrepareLiteralDecoding(s);
  true
}
// fn DecodeLiteralBlockSwitch<
// 'a,
// AllocU8 : alloc::Allocator<u8>,
// AllocU32 : alloc::Allocator<u32>,
// AllocHC : alloc::Allocator<HuffmanCode>> (safe : bool,
// mut s : &mut BrotliState<AllocU8, AllocU32, AllocHC>) {
// DecodeLiteralBlockSwitchInternal(false, s);
// }
//
// fn SafeDecodeLiteralBlockSwitch<
// 'a,
// AllocU8 : alloc::Allocator<u8>,
// AllocU32 : alloc::Allocator<u32>,
// AllocHC : alloc::Allocator<HuffmanCode>> (safe : bool,
// mut s : &mut BrotliState<AllocU8, AllocU32, AllocHC>) -> bool {
// return DecodeLiteralBlockSwitchInternal(true, s);
// }
//
// Block switch for insert/copy length.
// Reads 3..54 bits.
fn DecodeCommandBlockSwitchInternal<AllocU8: alloc::Allocator<u8>,
                                    AllocU32: alloc::Allocator<u32>,
                                    AllocHC: alloc::Allocator<HuffmanCode>>
  (safe: bool,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> bool {
  if (!DecodeBlockTypeAndLength(safe, &mut s.block_type_length_state, &mut s.br, 1, input)) {
    return false;
  }
  s.htree_command_index = fast!((s.block_type_length_state.block_type_rb)[3]) as u16;
  true
}

#[allow(dead_code)]
fn DecodeCommandBlockSwitch<AllocU8: alloc::Allocator<u8>,
                            AllocU32: alloc::Allocator<u32>,
                            AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8]) {
  DecodeCommandBlockSwitchInternal(false, s, input);
}
#[allow(dead_code)]
fn SafeDecodeCommandBlockSwitch<AllocU8: alloc::Allocator<u8>,
                                AllocU32: alloc::Allocator<u32>,
                                AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> bool {
  DecodeCommandBlockSwitchInternal(true, s, input)
}

// Block switch for distance codes.
// Reads 3..54 bits.
fn DecodeDistanceBlockSwitchInternal<AllocU8: alloc::Allocator<u8>,
                                     AllocU32: alloc::Allocator<u32>,
                                     AllocHC: alloc::Allocator<HuffmanCode>>
  (safe: bool,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> bool {
  if (!DecodeBlockTypeAndLength(safe, &mut s.block_type_length_state, &mut s.br, 2, input)) {
    return false;
  }
  s.dist_context_map_slice_index =
    (fast!((s.block_type_length_state.block_type_rb)[5]) << kDistanceContextBits) as usize;
  s.dist_htree_index = fast_slice!((s.dist_context_map)[s.dist_context_map_slice_index
                                                  + s.distance_context as usize]);
  true
}

#[allow(dead_code)]
fn DecodeDistanceBlockSwitch<AllocU8: alloc::Allocator<u8>,
                             AllocU32: alloc::Allocator<u32>,
                             AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8]) {
  DecodeDistanceBlockSwitchInternal(false, s, input);
}

#[allow(dead_code)]
fn SafeDecodeDistanceBlockSwitch<AllocU8: alloc::Allocator<u8>,
                                 AllocU32: alloc::Allocator<u32>,
                                 AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> bool {
  DecodeDistanceBlockSwitchInternal(true, s, input)
}

fn UnwrittenBytes<AllocU8: alloc::Allocator<u8>,
                  AllocU32: alloc::Allocator<u32>,
                  AllocHC: alloc::Allocator<HuffmanCode>> (
  s: &BrotliState<AllocU8, AllocU32, AllocHC>,
  wrap: bool,
)  -> usize {
  let pos = if wrap && s.pos > s.ringbuffer_size {
    s.ringbuffer_size as usize
  } else {
    s.pos as usize
  };
  let partial_pos_rb = (s.rb_roundtrips as usize * s.ringbuffer_size as usize) + pos as usize;
  (partial_pos_rb - s.partial_pos_out) as usize
}
fn WriteRingBuffer<'a,
                   AllocU8: alloc::Allocator<u8>,
                   AllocU32: alloc::Allocator<u32>,
                   AllocHC: alloc::Allocator<HuffmanCode>>(
  available_out: &mut usize,
  opt_output: Option<&mut [u8]>,
  output_offset: &mut usize,
  total_out: &mut usize,
  force: bool,
  s: &'a mut BrotliState<AllocU8, AllocU32, AllocHC>,
) -> (BrotliDecoderErrorCode, &'a [u8]) {
  let to_write = UnwrittenBytes(s, true);
  let mut num_written = *available_out as usize;
  if (num_written > to_write) {
    num_written = to_write;
  }
  if (s.meta_block_remaining_len < 0) {
    return (BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_BLOCK_LENGTH_1, &[]);
  }
  let start_index = (s.partial_pos_out & s.ringbuffer_mask as usize) as usize;
  let start = fast_slice!((s.ringbuffer)[start_index ; start_index + num_written as usize]);
  if let Some(output) = opt_output {
    fast_mut!((output)[*output_offset ; *output_offset + num_written as usize])
      .clone_from_slice(start);
  }
  *output_offset += num_written;
  *available_out -= num_written;
  BROTLI_LOG_UINT!(to_write);
  BROTLI_LOG_UINT!(num_written);
  s.partial_pos_out += num_written as usize;
  *total_out = s.partial_pos_out;
  if (num_written < to_write) {
    if s.ringbuffer_size == (1 << s.window_bits) || force {
      return (BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_OUTPUT, &[]);
    } else {
      return (BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS, start);
    }
  }
  if (s.ringbuffer_size == (1 << s.window_bits) &&
      s.pos >= s.ringbuffer_size) {
    s.pos -= s.ringbuffer_size;
    s.rb_roundtrips += 1;
    s.should_wrap_ringbuffer = s.pos != 0;
  }
  (BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS, start)
 }

fn WrapRingBuffer<AllocU8: alloc::Allocator<u8>,
                   AllocU32: alloc::Allocator<u32>,
                   AllocHC: alloc::Allocator<HuffmanCode>>(
  s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
) {
  if s.should_wrap_ringbuffer {
    let (ring_buffer_start, ring_buffer_end) = s.ringbuffer.slice_mut().split_at_mut(s.ringbuffer_size as usize);
    let pos = s.pos as usize;
    ring_buffer_start.split_at_mut(pos).0.clone_from_slice(ring_buffer_end.split_at(pos).0);
    s.should_wrap_ringbuffer = false;
  }

}

fn CopyUncompressedBlockToOutput<AllocU8: alloc::Allocator<u8>,
                                 AllocU32: alloc::Allocator<u32>,
                                 AllocHC: alloc::Allocator<HuffmanCode>>
  (mut available_out: &mut usize,
   mut output: &mut [u8],
   mut output_offset: &mut usize,
   mut total_out: &mut usize,
   mut s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  // State machine
  loop {
    match s.substate_uncompressed {
      BrotliRunningUncompressedState::BROTLI_STATE_UNCOMPRESSED_NONE => {
        let mut nbytes = bit_reader::BrotliGetRemainingBytes(&s.br) as i32;
        if (nbytes > s.meta_block_remaining_len) {
          nbytes = s.meta_block_remaining_len;
        }
        if (s.pos + nbytes > s.ringbuffer_size) {
          nbytes = s.ringbuffer_size - s.pos;
        }
        // Copy remaining bytes from s.br.buf_ to ringbuffer.
        bit_reader::BrotliCopyBytes(fast_mut!((s.ringbuffer.slice_mut())[s.pos as usize;]),
                                    &mut s.br,
                                    nbytes as u32,
                                    input);
        s.pos += nbytes;
        s.meta_block_remaining_len -= nbytes;
        if s.pos < (1 << s.window_bits) {
          if (s.meta_block_remaining_len == 0) {
            return BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
          }
          return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
        }
        s.substate_uncompressed = BrotliRunningUncompressedState::BROTLI_STATE_UNCOMPRESSED_WRITE;
        // s.partial_pos_rb += (size_t)s.ringbuffer_size;
        // No break, continue to next state by going aroudn the loop
      }
      BrotliRunningUncompressedState::BROTLI_STATE_UNCOMPRESSED_WRITE => {
        let (result, _) = WriteRingBuffer(&mut available_out,
                                          Some(&mut output),
                                          &mut output_offset,
                                          &mut total_out,
                                          false,
                                          &mut s);
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
          _ => return result,
        }
        if s.ringbuffer_size == 1 << s.window_bits {
          s.max_distance = s.max_backward_distance;
        }
        s.substate_uncompressed = BrotliRunningUncompressedState::BROTLI_STATE_UNCOMPRESSED_NONE;
      }
    }
  }
}

fn BrotliAllocateRingBuffer<AllocU8: alloc::Allocator<u8>,
                            AllocU32: alloc::Allocator<u32>,
                            AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> bool {
  // We need the slack region for the following reasons:
  // - doing up to two 16-byte copies for fast backward copying
  // - inserting transformed dictionary word (5 prefix + 24 base + 8 suffix)
  const kRingBufferWriteAheadSlack: i32 = 42;
  let mut is_last = s.is_last_metablock;
  s.ringbuffer_size = 1 << s.window_bits;

  if (s.is_uncompressed != 0) {
    let next_block_header =
      bit_reader::BrotliPeekByte(&mut s.br, s.meta_block_remaining_len as u32, input);
    if (next_block_header != -1) &&
        // Peek succeeded
        ((next_block_header & 3) == 3) {
      // ISLAST and ISEMPTY
      is_last = 1;
    }
  }
  let max_dict_size = s.ringbuffer_size as usize - 16;
  {
    let custom_dict = if s.custom_dict_size as usize > max_dict_size {
      let cd = fast_slice!((s.custom_dict)[(s.custom_dict_size as usize - max_dict_size); s.custom_dict_size as usize]);
      s.custom_dict_size = max_dict_size as isize;
      cd
    } else {
      fast_slice!((s.custom_dict)[0; s.custom_dict_size as usize])
    };

    // We need at least 2 bytes of ring buffer size to get the last two
    // bytes for context from there
    if (is_last != 0) {
      while (s.ringbuffer_size as isize >= (s.custom_dict_size + s.meta_block_remaining_len as isize + 16) * 2 && s.ringbuffer_size as isize > 32) {
        s.ringbuffer_size >>= 1;
      }
    }
    if s.ringbuffer_size > (1 << s.window_bits) {
      s.ringbuffer_size = (1 << s.window_bits);
    }

    s.ringbuffer_mask = s.ringbuffer_size - 1;
    s.ringbuffer = s.alloc_u8
      .alloc_cell((s.ringbuffer_size as usize + kRingBufferWriteAheadSlack as usize +
                   kBrotliMaxDictionaryWordLength as usize));
    if (s.ringbuffer.slice().len() == 0) {
      return false;
    }
    fast_mut!((s.ringbuffer.slice_mut())[s.ringbuffer_size as usize - 1]) = 0;
    fast_mut!((s.ringbuffer.slice_mut())[s.ringbuffer_size as usize - 2]) = 0;
    if custom_dict.len() != 0 {
      let offset = ((-s.custom_dict_size) & s.ringbuffer_mask as isize) as usize;
      fast_mut!((s.ringbuffer.slice_mut())[offset ; offset + s.custom_dict_size as usize]).clone_from_slice(custom_dict);
    }
  }
  if s.custom_dict.slice().len() != 0 {
    s.alloc_u8.free_cell(core::mem::replace(&mut s.custom_dict,
                         AllocU8::AllocatedMemory::default()));
  }
  true
}

// Reads 1..256 2-bit context modes.
pub fn ReadContextModes<AllocU8: alloc::Allocator<u8>,
                        AllocU32: alloc::Allocator<u32>,
                        AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {

  let mut i: i32 = s.loop_counter;

  for context_mode_iter in fast_mut!((s.context_modes.slice_mut())[i as usize ;
                                                       (s.block_type_length_state.num_block_types[0]
                                                        as usize)])
    .iter_mut() {
    let mut bits: u32 = 0;
    if (!bit_reader::BrotliSafeReadBits(&mut s.br, 2, &mut bits, input)) {
      mark_unlikely();
      s.loop_counter = i;
      return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
    }
    *context_mode_iter = bits as u8;
    BROTLI_LOG_UINT!(i);
    i += 1;
  }
  BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS
}

pub fn TakeDistanceFromRingBuffer<AllocU8: alloc::Allocator<u8>,
                                  AllocU32: alloc::Allocator<u32>,
                                  AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>) {
  if (s.distance_code == 0) {
    s.dist_rb_idx -= 1;
    s.distance_code = fast!((s.dist_rb)[(s.dist_rb_idx & 3) as usize]);
    s.distance_context = 1;
  } else {
    let distance_code = s.distance_code << 1;
    // kDistanceShortCodeIndexOffset has 2-bit values from LSB:
    // 3, 2, 1, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2
    const kDistanceShortCodeIndexOffset: u32 = 0xaaafff1b;
    // kDistanceShortCodeValueOffset has 2-bit values from LSB:
    // -0, 0,-0, 0,-1, 1,-2, 2,-3, 3,-1, 1,-2, 2,-3, 3
    const kDistanceShortCodeValueOffset: u32 = 0xfa5fa500;
    let mut v = (s.dist_rb_idx as i32 +
                 (kDistanceShortCodeIndexOffset as i32 >>
                  distance_code as i32)) as i32 & 0x3;
    s.distance_code = fast!((s.dist_rb)[v as usize]);
    v = (kDistanceShortCodeValueOffset >> distance_code) as i32 & 0x3;
    if ((distance_code & 0x3) != 0) {
      s.distance_code += v;
    } else {
      s.distance_code -= v;
      if (s.distance_code <= 0) {
        // A huge distance will cause a BROTLI_FAILURE() soon.
        // This is a little faster than failing here.
        s.distance_code = 0x7fffffff;
      }
    }
  }
}

pub fn SafeReadBits(br: &mut bit_reader::BrotliBitReader,
                    n_bits: u32,
                    val: &mut u32,
                    input: &[u8])
                    -> bool {
  if (n_bits != 0) {
    bit_reader::BrotliSafeReadBits(br, n_bits, val, input)
  } else {
    *val = 0;
    true
  }
}

// Precondition: s.distance_code < 0
pub fn ReadDistanceInternal<AllocU8: alloc::Allocator<u8>,
                            AllocU32: alloc::Allocator<u32>,
                            AllocHC: alloc::Allocator<HuffmanCode>>
  (safe: bool,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8],
   distance_hgroup: &[&[HuffmanCode]; 256])
   -> bool {
  let mut distval: i32;
  let mut memento = bit_reader::BrotliBitReaderState::default();
  if (!safe) {
    s.distance_code = ReadSymbol(fast!((distance_hgroup)[s.dist_htree_index as usize]),
                                 &mut s.br,
                                 input) as i32;
  } else {
    let mut code: u32 = 0;
    memento = bit_reader::BrotliBitReaderSaveState(&s.br);
    if !SafeReadSymbol(fast!((distance_hgroup)[s.dist_htree_index as usize]),
                       &mut s.br,
                       &mut code,
                       input) {
      return false;
    }
    s.distance_code = code as i32;
  }
  // Convert the distance code to the actual distance by possibly
  // looking up past distances from the s.ringbuffer.
  s.distance_context = 0;
  if ((s.distance_code as u64 & 0xfffffffffffffff0) == 0) {
    TakeDistanceFromRingBuffer(s);
    fast_mut!((s.block_type_length_state.block_length)[2]) -= 1;
    return true;
  }
  distval = s.distance_code - s.num_direct_distance_codes as i32;
  if (distval >= 0) {
    let nbits: u32;
    let postfix: i32;
    let offset: i32;
    if (!safe && (s.distance_postfix_bits == 0)) {
      nbits = (distval as u32 >> 1) + 1;
      offset = ((2 + (distval & 1)) << nbits) - 4;
      s.distance_code = (s.num_direct_distance_codes as i64 + offset as i64 +
                        bit_reader::BrotliReadBits(&mut s.br, nbits, input) as i64) as i32;
    } else {
      // This branch also works well when s.distance_postfix_bits == 0
      let mut bits: u32 = 0;
      postfix = distval & s.distance_postfix_mask;
      distval >>= s.distance_postfix_bits;
      nbits = (distval as u32 >> 1) + 1;
      if (safe) {
        if (!SafeReadBits(&mut s.br, nbits, &mut bits, input)) {
          s.distance_code = -1; /* Restore precondition. */
          bit_reader::BrotliBitReaderRestoreState(&mut s.br, &memento);
          return false;
        }
      } else {
        bits = bit_reader::BrotliReadBits(&mut s.br, nbits, input);
      }
      offset = (((distval & 1).wrapping_add(2)) << nbits).wrapping_sub(4);
      s.distance_code = ((i64::from(offset) + i64::from(bits)) << s.distance_postfix_bits).wrapping_add(i64::from(postfix)).wrapping_add(i64::from(s.num_direct_distance_codes)) as i32;
    }
  }
  s.distance_code = s.distance_code.wrapping_sub(NUM_DISTANCE_SHORT_CODES as i32).wrapping_add(1);
  fast_mut!((s.block_type_length_state.block_length)[2]) -= 1;
  true
}


pub fn ReadCommandInternal<AllocU8: alloc::Allocator<u8>,
                           AllocU32: alloc::Allocator<u32>,
                           AllocHC: alloc::Allocator<HuffmanCode>>
  (safe: bool,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   insert_length: &mut i32,
   input: &[u8],
   insert_copy_hgroup: &[&[HuffmanCode]; 256])
   -> bool {
  let mut cmd_code: u32 = 0;
  let mut insert_len_extra: u32 = 0;
  let mut copy_length: u32 = 0;
  let v: prefix::CmdLutElement;
  let mut memento = bit_reader::BrotliBitReaderState::default();
  if (!safe) {
    cmd_code = ReadSymbol(fast!((insert_copy_hgroup)[s.htree_command_index as usize]),
                          &mut s.br,
                          input);
  } else {
    memento = bit_reader::BrotliBitReaderSaveState(&s.br);
    if (!SafeReadSymbol(fast!((insert_copy_hgroup)[s.htree_command_index as usize]),
                        &mut s.br,
                        &mut cmd_code,
                        input)) {
      return false;
    }
  }
  v = fast!((prefix::kCmdLut)[cmd_code as usize]);
  s.distance_code = v.distance_code as i32;
  s.distance_context = v.context as i32;
  s.dist_htree_index = fast_slice!((s.dist_context_map)[s.dist_context_map_slice_index
                                                  + s.distance_context as usize]);
  *insert_length = v.insert_len_offset as i32;
  if (!safe) {
    if v.insert_len_extra_bits != 0 {
      mark_unlikely();
      insert_len_extra =
        bit_reader::BrotliReadBits(&mut s.br, v.insert_len_extra_bits as u32, input);
    }
    copy_length = bit_reader::BrotliReadBits(&mut s.br, v.copy_len_extra_bits as u32, input);
  } else if (!SafeReadBits(&mut s.br,
                    v.insert_len_extra_bits as u32,
                    &mut insert_len_extra,
                    input)) ||
     (!SafeReadBits(&mut s.br,
                    v.copy_len_extra_bits as u32,
                    &mut copy_length,
                    input)) {
    bit_reader::BrotliBitReaderRestoreState(&mut s.br, &memento);
    return false;
  }
  s.copy_length = copy_length as i32 + v.copy_len_offset as i32;
  fast_mut!((s.block_type_length_state.block_length)[1]) -= 1;
  *insert_length += insert_len_extra as i32;
  true
}


fn WarmupBitReader(safe: bool, br: &mut bit_reader::BrotliBitReader, input: &[u8]) -> bool {
  safe || bit_reader::BrotliWarmupBitReader(br, input)
}

fn CheckInputAmount(safe: bool, br: &bit_reader::BrotliBitReader, num: u32) -> bool {
  safe || bit_reader::BrotliCheckInputAmount(br, num)
}

#[inline(always)]
fn memmove16(data: &mut [u8], u32off_dst: u32, u32off_src: u32) {
  let off_dst = u32off_dst as usize;
  let off_src = u32off_src as usize;
  // data[off_dst + 15] = data[off_src + 15];
  // data[off_dst + 14] = data[off_src + 14];
  // data[off_dst + 13] = data[off_src + 13];
  // data[off_dst + 12] = data[off_src + 12];
  //
  // data[off_dst + 11] = data[off_src + 11];
  // data[off_dst + 10] = data[off_src + 10];
  // data[off_dst + 9] = data[off_src + 9];
  // data[off_dst + 8] = data[off_src + 8];
  //
  // data[off_dst + 7] = data[off_src + 7];
  // data[off_dst + 6] = data[off_src + 6];
  // data[off_dst + 5] = data[off_src + 5];
  // data[off_dst + 4] = data[off_src + 4];
  //
  // data[off_dst + 3] = data[off_src + 3];
  // data[off_dst + 2] = data[off_src + 2];
  // data[off_dst + 1] = data[off_src + 1];
  //
  let mut local_array: [u8; 16] = fast_uninitialized!(16);
  local_array.clone_from_slice(fast!((data)[off_src as usize ; off_src as usize + 16]));
  fast_mut!((data)[off_dst as usize ; off_dst as usize + 16]).clone_from_slice(&local_array);


}


#[cfg(not(feature="unsafe"))]
fn memcpy_within_slice(data: &mut [u8], off_dst: usize, off_src: usize, size: usize) {
  if off_dst > off_src {
    let (src, dst) = data.split_at_mut(off_dst);
    let src_slice = fast!((src)[off_src ; off_src + size]);
    fast_mut!((dst)[0;size]).clone_from_slice(src_slice);
  } else {
    let (dst, src) = data.split_at_mut(off_src);
    let src_slice = fast!((src)[0;size]);
    fast_mut!((dst)[off_dst;off_dst + size]).clone_from_slice(src_slice);
  }
}

#[cfg(feature="unsafe")]
fn memcpy_within_slice(data: &mut [u8], off_dst: usize, off_src: usize, size: usize) {
  let ptr = data.as_mut_ptr();
  unsafe {
    let dst = ptr.offset(off_dst as isize);
    let src = ptr.offset(off_src as isize);
    core::ptr::copy_nonoverlapping(src, dst, size);
  }
}

pub fn BrotliDecoderHasMoreOutput<AllocU8: alloc::Allocator<u8>,
                           AllocU32: alloc::Allocator<u32>,
                           AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &BrotliState<AllocU8, AllocU32, AllocHC>) -> bool {
  /* After unrecoverable error remaining output is considered nonsensical. */
  if is_fatal(s.error_code) {
    return false;
  }
  s.ringbuffer.len() != 0 && UnwrittenBytes(s, false) != 0
}
pub fn BrotliDecoderTakeOutput<'a,
                               AllocU8: alloc::Allocator<u8>,
                               AllocU32: alloc::Allocator<u32>,
                               AllocHC: alloc::Allocator<HuffmanCode>>(
  s: &'a mut BrotliState<AllocU8, AllocU32, AllocHC>,
  size: &mut usize,
) -> &'a [u8] {
  let one:usize = 1;
  let mut available_out = if *size != 0 { *size } else { one << 24 };
  let requested_out = available_out;
  if (s.ringbuffer.len() == 0) || is_fatal(s.error_code) {
    *size = 0;
    return &[];
  }
  WrapRingBuffer(s);
  let mut ign = 0usize;
  let mut ign2 = 0usize;
  let (status, result) = WriteRingBuffer(&mut available_out, None, &mut ign,&mut ign2, true, s);
  // Either WriteRingBuffer returns those "success" codes...
  match status {
    BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS |  BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_OUTPUT => {
      *size = requested_out - available_out;
    },
    _ => {
      // ... or stream is broken. Normally this should be caught by
      //   BrotliDecoderDecompressStream, this is just a safeguard.
      if is_fatal(status) {
        // SaveErrorCode!(s, status); //borrow checker doesn't like this
        // but since it's a safeguard--ignore
      }
      *size = 0;
      return &[];
    }
  }
  return result;
}

#[cfg(feature="ffi-api")]
pub fn BrotliDecoderIsUsed<AllocU8: alloc::Allocator<u8>,
                           AllocU32: alloc::Allocator<u32>,
                           AllocHC: alloc::Allocator<HuffmanCode>>(
  s: &BrotliState<AllocU8, AllocU32, AllocHC>) -> bool {
  if let BrotliRunningState::BROTLI_STATE_UNINITED = s.state {
    false
  } else {
    bit_reader::BrotliGetAvailableBits(&s.br) != 0
  }
}

pub fn BrotliDecoderIsFinished<AllocU8: alloc::Allocator<u8>,
                               AllocU32: alloc::Allocator<u32>,
                               AllocHC: alloc::Allocator<HuffmanCode>>(
  s: &BrotliState<AllocU8, AllocU32, AllocHC>) -> bool {
  if let BrotliRunningState::BROTLI_STATE_DONE = s.state {
    !BrotliDecoderHasMoreOutput(s)
  } else {
    false
  }
}

pub fn BrotliDecoderGetErrorCode<AllocU8: alloc::Allocator<u8>,
                               AllocU32: alloc::Allocator<u32>,
                               AllocHC: alloc::Allocator<HuffmanCode>>(
  s: &BrotliState<AllocU8, AllocU32, AllocHC>) -> BrotliDecoderErrorCode {
  s.error_code
}

fn ProcessCommandsInternal<AllocU8: alloc::Allocator<u8>,
                           AllocU32: alloc::Allocator<u32>,
                           AllocHC: alloc::Allocator<HuffmanCode>>
  (safe: bool,
   s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  if (!CheckInputAmount(safe, &s.br, 28)) || (!WarmupBitReader(safe, &mut s.br, input)) {
    mark_unlikely();
    return BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
  }
  let mut pos = s.pos;
  let mut i: i32 = s.loop_counter; // important that this is signed
  let mut result = BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
  let mut saved_literal_hgroup =
    core::mem::replace(&mut s.literal_hgroup,
                       HuffmanTreeGroup::<AllocU32, AllocHC>::default());
  let mut saved_distance_hgroup =
    core::mem::replace(&mut s.distance_hgroup,
                       HuffmanTreeGroup::<AllocU32, AllocHC>::default());
  let mut saved_insert_copy_hgroup =
    core::mem::replace(&mut s.insert_copy_hgroup,
                       HuffmanTreeGroup::<AllocU32, AllocHC>::default());
  {

    let literal_hgroup = saved_literal_hgroup.build_hgroup_cache();
    let distance_hgroup = saved_distance_hgroup.build_hgroup_cache();
    let insert_copy_hgroup = saved_insert_copy_hgroup.build_hgroup_cache();

    loop {
      match s.state {
        BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN => {
          if (!CheckInputAmount(safe, &s.br, 28)) {
            // 156 bits + 7 bytes
            mark_unlikely();
            result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            break; // return
          }
          if (fast_mut!((s.block_type_length_state.block_length)[1]) == 0) {
            mark_unlikely();
            if !DecodeCommandBlockSwitchInternal(safe, s, input) {
              result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
              break; // return
            }
            s.state = BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN;
            continue; // goto CommandBegin;
          }
          // Read the insert/copy length in the command
          if (!ReadCommandInternal(safe, s, &mut i, input, &insert_copy_hgroup)) && safe {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            break; // return
          }
          BROTLI_LOG!("[ProcessCommandsInternal] pos = %d insert = %d copy = %d distance = %d\n",
              pos, i, s.copy_length, s.distance_code);
          if (i == 0) {
            s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_DECODE_LITERALS;
            continue; // goto CommandPostDecodeLiterals;
          }
          s.meta_block_remaining_len -= i;
          s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER;
        }
        BrotliRunningState::BROTLI_STATE_COMMAND_INNER => {
          // Read the literals in the command
          if (s.trivial_literal_context != 0) {
            let mut bits: u32 = 0;
            let mut value: u32 = 0;
            let mut literal_htree = &fast!((literal_hgroup)[s.literal_htree_index as usize]);
            PreloadSymbol(safe, literal_htree, &mut s.br, &mut bits, &mut value, input);
            let mut inner_return: bool = false;
            let mut inner_continue: bool = false;
            loop {
              if (!CheckInputAmount(safe, &s.br, 28)) {
                // 162 bits + 7 bytes
                result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                inner_return = true;
                break;
              }
              if (fast!((s.block_type_length_state.block_length)[0]) == 0) {
                mark_unlikely();
                if (!DecodeLiteralBlockSwitchInternal(safe, s, input)) && safe {
                  result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                  inner_return = true;
                  break;
                }
                literal_htree = fast_ref!((literal_hgroup)[s.literal_htree_index as usize]);
                PreloadSymbol(safe, literal_htree, &mut s.br, &mut bits, &mut value, input);
                if (s.trivial_literal_context == 0) {
                  s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER;
                  inner_continue = true;
                  break; // goto StateCommandInner
                }
              }
              if (!safe) {
                fast_mut!((s.ringbuffer.slice_mut())[pos as usize]) =
                  ReadPreloadedSymbol(literal_htree, &mut s.br, &mut bits, &mut value, input) as u8;
              } else {
                let mut literal: u32 = 0;
                if (!SafeReadSymbol(literal_htree, &mut s.br, &mut literal, input)) {
                  result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                  inner_return = true;
                  break;
                }
                fast_mut!((s.ringbuffer.slice_mut())[pos as usize]) = literal as u8;
              }
              if (s.block_type_length_state.block_length)[0] == 0 {
                  mark_unlikely();
                  result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_WINDOW_BITS;
                  inner_return = true;
                  break;
              }
              fast_mut!((s.block_type_length_state.block_length)[0]) -= 1;
              BROTLI_LOG_UINT!(s.literal_htree_index);
              BROTLI_LOG_ARRAY_INDEX!(s.ringbuffer.slice(), pos);
              pos += 1;
              if (pos == s.ringbuffer_size) {
                mark_unlikely();
                s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER_WRITE;
                i -= 1;
                inner_return = true;
                break;
              }
              i -= 1;
              if i == 0 {
                break;
              }
            }
            if inner_return {
              break; // return
            }
            if inner_continue {
              mark_unlikely();
              continue;
            }
          } else {
            let mut p1 = fast_slice!((s.ringbuffer)[((pos - 1) & s.ringbuffer_mask) as usize]);
            let mut p2 = fast_slice!((s.ringbuffer)[((pos - 2) & s.ringbuffer_mask) as usize]);
            if s.custom_dict_avoid_context_seed && pos < 2 {
                mark_unlikely();
                p2 = 0;
                p1 = 0;
            }
            if pos > 1
            {
                // have already set both seed bytes and can now move on to using
                // the ringbuffer.
                s.custom_dict_avoid_context_seed = false;
            }
            let mut inner_return: bool = false;
            let mut inner_continue: bool = false;
            loop {
              if (!CheckInputAmount(safe, &s.br, 28)) {
                // 162 bits + 7 bytes
                s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER;
                result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                inner_return = true;
                break;
              }
              if (fast!((s.block_type_length_state.block_length)[0]) == 0) {
                mark_unlikely();
                if (!DecodeLiteralBlockSwitchInternal(safe, s, input)) && safe {
                  result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                  inner_return = true;
                  break;
                }
                if s.trivial_literal_context != 0 {
                  s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER;
                  inner_continue = true;
                  break;
                }
              }
              let context = s.context_lookup[p1 as usize] | s.context_lookup[p2 as usize |256];
              BROTLI_LOG_UINT!(p1);
              BROTLI_LOG_UINT!(p2);
              BROTLI_LOG_UINT!(context);
              let hc: &[HuffmanCode];
              {
                let i = fast_slice!((s.context_map)[s.context_map_slice_index + context as usize]);
                hc = fast!((literal_hgroup)[i as usize]);
              }
              p2 = p1;
              if (!safe) {
                p1 = ReadSymbol(hc, &mut s.br, input) as u8;
              } else {
                let mut literal: u32 = 0;
                if (!SafeReadSymbol(hc, &mut s.br, &mut literal, input)) {
                  result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                  inner_return = true;
                  break;
                }
                p1 = literal as u8;
              }
              fast_slice_mut!((s.ringbuffer)[pos as usize]) = p1;
              if (s.block_type_length_state.block_length)[0] == 0 {
                  result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_WINDOW_BITS;
                  inner_return = true;
                  break;
              }
              fast_mut!((s.block_type_length_state.block_length)[0]) -= 1;
              BROTLI_LOG_UINT!(s.context_map.slice()[s.context_map_slice_index as usize +
                                                     context as usize]);
              BROTLI_LOG_ARRAY_INDEX!(s.ringbuffer.slice(), pos & s.ringbuffer_mask);
              pos += 1;
              if (pos == s.ringbuffer_size) {
                mark_unlikely();
                s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER_WRITE;
                i -= 1;
                inner_return = true;
                break;
              }
              i -= 1;
              if i == 0 {
                break;
              }
            }
            if inner_return {
              break; // return
            }
            if inner_continue {
              mark_unlikely();
              continue;
            }
          }
          if (s.meta_block_remaining_len <= 0) {
            mark_unlikely();
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
            break; // return
          }
          s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_DECODE_LITERALS;
        }
        BrotliRunningState::BROTLI_STATE_COMMAND_POST_DECODE_LITERALS => {
          if s.distance_code >= 0 {
            let not_distance_code = if s.distance_code != 0 { 0 } else { 1 };
            s.distance_context = not_distance_code;
            s.dist_rb_idx -= 1;
            s.distance_code = fast!((s.dist_rb)[(s.dist_rb_idx & 3) as usize]);
            // goto postReadDistance
          } else {
            if fast!((s.block_type_length_state.block_length)[2]) == 0 {
              mark_unlikely();
              if (!DecodeDistanceBlockSwitchInternal(safe, s, input)) && safe {
                result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
                break; // return
              }
            }
            if (!ReadDistanceInternal(safe, s, input, &distance_hgroup)) && safe {
              result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
              break; // return
            }
          }
          // postReadDistance:
          BROTLI_LOG!("[ProcessCommandsInternal] pos = %d distance = %d\n",
                      pos, s.distance_code);

          if (s.max_distance != s.max_backward_distance) {
            if (pos < s.max_backward_distance_minus_custom_dict_size) {
              s.max_distance = pos + s.custom_dict_size as i32;
            } else {
              s.max_distance = s.max_backward_distance;
            }
          }
          i = s.copy_length;
          // Apply copy of LZ77 back-reference, or static dictionary reference if
          // the distance is larger than the max LZ77 distance
          if (s.distance_code > s.max_distance) {
            if s.distance_code > kBrotliMaxAllowedDistance as i32 {
              return BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_DISTANCE;
            }
            if (i >= kBrotliMinDictionaryWordLength as i32 &&
                i <= kBrotliMaxDictionaryWordLength as i32) {
              let mut offset = fast!((kBrotliDictionaryOffsetsByLength)[i as usize]) as i32;
              let word_id = s.distance_code - s.max_distance - 1;
              let shift = fast!((kBrotliDictionarySizeBitsByLength)[i as usize]);
              let mask = bit_reader::BitMask(shift as u32) as i32;
              let word_idx = word_id & mask;
              let transform_idx = word_id >> shift;
              s.dist_rb_idx += s.distance_context;
              offset += word_idx * i;
              if (transform_idx < kNumTransforms) {
                let mut len = i;
                let word = fast!((kBrotliDictionary)[offset as usize ; (offset + len) as usize]);
                if (transform_idx == 0) {
                  fast_slice_mut!((s.ringbuffer)[pos as usize ; ((pos + len) as usize)])
                    .clone_from_slice(word);
                } else {
                  len = TransformDictionaryWord(fast_slice_mut!((s.ringbuffer)[pos as usize;]),
                                                word,
                                                len,
                                                transform_idx);
                }
                pos += len;
                s.meta_block_remaining_len -= len;
                if (pos >= s.ringbuffer_size) {
                  // s.partial_pos_rb += (size_t)s.ringbuffer_size;
                  s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRITE_1;
                  break; // return return
                }
              } else {
                BROTLI_LOG!(
                  "Invalid backward reference. pos: %d distance: %d len: %d bytes left: %d\n",
                  pos, s.distance_code, i,
                  s.meta_block_remaining_len);
                result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_TRANSFORM;
                break; // return
              }
            } else {
              BROTLI_LOG!(
                "Invalid backward reference. pos:%d distance:%d len:%d bytes left:%d\n",
                pos, s.distance_code, i, s.meta_block_remaining_len);
              result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_DICTIONARY;
              break; // return
            }
          } else {
            // update the recent distances cache
            fast_mut!((s.dist_rb)[(s.dist_rb_idx & 3) as usize]) = s.distance_code;
            s.dist_rb_idx += 1;
            s.meta_block_remaining_len -= i;
            // There is 128+ bytes of slack in the ringbuffer allocation.
            // Also, we have 16 short codes, that make these 16 bytes irrelevant
            // in the ringbuffer. Let's copy over them as a first guess.
            //
            let src_start = ((pos - s.distance_code) & s.ringbuffer_mask) as u32;
            let dst_start = pos as u32;
            let dst_end = pos as u32 + i as u32;
            let src_end = src_start + i as u32;
            memmove16(&mut s.ringbuffer.slice_mut(), dst_start, src_start);
            // Now check if the copy extends over the ringbuffer end,
            // or if the copy overlaps with itself, if yes, do wrap-copy.
            if (src_end > pos as u32 && dst_end > src_start) {
              s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRAP_COPY;
              continue; //goto CommandPostWrapCopy;
            }
            if (dst_end >= s.ringbuffer_size as u32 || src_end >= s.ringbuffer_size as u32) {
              s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRAP_COPY;
              continue; //goto CommandPostWrapCopy;
            }
            pos += i;
            if (i > 16) {
              if (i > 32) {
                memcpy_within_slice(s.ringbuffer.slice_mut(),
                                    dst_start as usize + 16,
                                    src_start as usize + 16,
                                    (i - 16) as usize);
              } else {
                // This branch covers about 45% cases.
                // Fixed size short copy allows more compiler optimizations.
                memmove16(&mut s.ringbuffer.slice_mut(),
                          dst_start + 16,
                          src_start + 16);
              }
            }
          }
          if (s.meta_block_remaining_len <= 0) {
            // Next metablock, if any
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
            break; // return
          } else {
            s.state = BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN;
            continue; // goto CommandBegin
          }
        }
        BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRAP_COPY => {
          let mut wrap_guard = s.ringbuffer_size - pos;
          let mut inner_return: bool = false;
          while i > 0 {
            i -= 1;
            fast_slice_mut!((s.ringbuffer)[pos as usize]) =
              fast_slice!((s.ringbuffer)[((pos - s.distance_code) & s.ringbuffer_mask) as usize]);
            pos += 1;
            wrap_guard -= 1;
            if (wrap_guard == 0) {
              mark_unlikely();
              // s.partial_pos_rb += (size_t)s.ringbuffer_size;
              s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRITE_2;
              inner_return = true;
              break; //return
            }
          }
          if inner_return {
            mark_unlikely();
            break;
          }
          i -= 1;
          if (s.meta_block_remaining_len <= 0) {
            // Next metablock, if any
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
            break; // return
          } else {
            s.state = BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN;
            continue;
          }
        }
        _ => {
          result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_UNREACHABLE;
          break; // return
        }
      }
    }
  }
  s.pos = pos;
  s.loop_counter = i;

  let _ = core::mem::replace(&mut s.literal_hgroup,
                     core::mem::replace(&mut saved_literal_hgroup,
                                        HuffmanTreeGroup::<AllocU32, AllocHC>::default()));

  let _ = core::mem::replace(&mut s.distance_hgroup,
                     core::mem::replace(&mut saved_distance_hgroup,
                                        HuffmanTreeGroup::<AllocU32, AllocHC>::default()));

  let _ = core::mem::replace(&mut s.insert_copy_hgroup,
                     core::mem::replace(&mut saved_insert_copy_hgroup,
                                        HuffmanTreeGroup::<AllocU32, AllocHC>::default()));

  result
}

fn ProcessCommands<AllocU8: alloc::Allocator<u8>,
                   AllocU32: alloc::Allocator<u32>,
                   AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  ProcessCommandsInternal(false, s, input)
}

fn SafeProcessCommands<AllocU8: alloc::Allocator<u8>,
                       AllocU32: alloc::Allocator<u32>,
                       AllocHC: alloc::Allocator<HuffmanCode>>
  (s: &mut BrotliState<AllocU8, AllocU32, AllocHC>,
   input: &[u8])
   -> BrotliDecoderErrorCode {
  ProcessCommandsInternal(true, s, input)
}

/* Returns the maximum number of distance symbols which can only represent
   distances not exceeding BROTLI_MAX_ALLOWED_DISTANCE. */
pub fn BrotliMaxDistanceSymbol(ndirect: u32, npostfix: u32) -> u32{
  let bound:[u32;kBrotliMaxPostfix + 1] = [0, 4, 12, 28];
  let diff:[u32;kBrotliMaxPostfix + 1] = [73, 126, 228, 424];
  let postfix = 1 << npostfix;
  if (ndirect < bound[npostfix as usize ]) {
    return ndirect + diff[npostfix as usize] + postfix;
  } else if (ndirect > bound[npostfix as usize] + postfix) {
    return ndirect + diff[npostfix as usize];
  } else {
    return bound[npostfix as usize] + diff[npostfix as usize] + postfix;
  }
}

pub fn BrotliDecompressStream<AllocU8: alloc::Allocator<u8>,
                              AllocU32: alloc::Allocator<u32>,
                              AllocHC: alloc::Allocator<HuffmanCode>>
  (available_in: &mut usize,
   input_offset: &mut usize,
   xinput: &[u8],
   mut available_out: &mut usize,
   mut output_offset: &mut usize,
   mut output: &mut [u8],
   mut total_out: &mut usize,
   mut s: &mut BrotliState<AllocU8, AllocU32, AllocHC>)
   -> BrotliResult {

  let mut result = BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;

  let mut saved_buffer: [u8; 8] = s.buffer;
  let mut local_input: &[u8];
  if is_fatal(s.error_code) {
    return BrotliResult::ResultFailure;
  }
  if *available_in as u64 >= (1u64 << 32) {
    return SaveErrorCode!(s, BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_INVALID_ARGUMENTS);
  }
  if *input_offset as u64 >= (1u64 << 32) {
    return SaveErrorCode!(s, BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_INVALID_ARGUMENTS);
  }
  if *input_offset + *available_in > xinput.len() {
    return SaveErrorCode!(s, BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_INVALID_ARGUMENTS);
  }
  if *output_offset + *available_out > output.len() {
    return SaveErrorCode!(s, BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_INVALID_ARGUMENTS);
  }
  if s.buffer_length == 0 {
    local_input = xinput;
    s.br.avail_in = *available_in as u32;
    s.br.next_in = *input_offset as u32;
  } else {
    result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
    let copy_len = core::cmp::min(saved_buffer.len() - s.buffer_length as usize, *available_in);
    if copy_len > 0 {
      fast_mut!((saved_buffer)[s.buffer_length as usize ; (s.buffer_length as usize + copy_len)])
        .clone_from_slice(fast!((xinput)[*input_offset ; copy_len + *input_offset]));
      fast_mut!((s.buffer)[s.buffer_length as usize ; (s.buffer_length as usize + copy_len)])
        .clone_from_slice(fast!((xinput)[*input_offset ; copy_len + *input_offset]));
    }
    local_input = &saved_buffer[..];
    s.br.next_in = 0;
  }
  loop {
    match result {
      BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
      _ => {
        match result {
          BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT => {
            if s.ringbuffer.slice().len() != 0 {
              let (intermediate_result, _) = WriteRingBuffer(available_out,
                                                             Some(&mut output),
                                                             &mut output_offset,
                                                             &mut total_out,
                                                             true,
                                                             &mut s);
              if is_fatal(intermediate_result) {
                result = intermediate_result;
                break;
              }
            }
            if s.buffer_length != 0 {
              // Used with internal buffer.
              if s.br.avail_in == 0 {
                // Successfully finished read transaction.
                // Accamulator contains less than 8 bits, because internal buffer
                // is expanded byte-by-byte until it is enough to complete read.
                s.buffer_length = 0;
                // Switch to input stream and restart.
                result = BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
                local_input = xinput;
                s.br.avail_in = *available_in as u32;
                s.br.next_in = *input_offset as u32;
                continue;
              } else if *available_in != 0 {
                // Not enough data in buffer, but can take one more byte from
                // input stream.
                result = BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS;
                let new_byte = fast!((xinput)[*input_offset]);
                fast_mut!((s.buffer)[s.buffer_length as usize]) = new_byte;
                // we did the following copy upfront, so we wouldn't have to do it here
                // since saved_buffer[s.buffer_length as usize] = new_byte violates borrow rules
                assert_eq!(fast!((saved_buffer)[s.buffer_length as usize]), new_byte);
                s.buffer_length += 1;
                s.br.avail_in = s.buffer_length;
                (*input_offset) += 1;
                (*available_in) -= 1;
                // Retry with more data in buffer.
                // we can't re-borrow the saved buffer...so we have to do this recursively
                continue;
              }
              // Can't finish reading and no more input.

              // FIXME :: NOT SURE WHAT THIS MEANT
              // saved_buffer = core::mem::replace(
              //  &mut s.br.input_,
              //  &mut[]); // clear input
              break;
            } else {
              // Input stream doesn't contain enough input.
              // Copy tail to internal buffer and return.
              *input_offset = s.br.next_in as usize;
              *available_in = s.br.avail_in as usize;
              while *available_in != 0 {
                fast_mut!((s.buffer)[s.buffer_length as usize]) = fast!((xinput)[*input_offset]);
                s.buffer_length += 1;
                (*input_offset) += 1;
                (*available_in) -= 1;
              }
              break;
            }
            // unreachable!(); <- dead code
          }
          _ => {
            // Fail or needs more output.
            if s.buffer_length != 0 {
              // Just consumed the buffered input and produced some output. Otherwise
              // it would result in "needs more input". Reset internal buffer.
              s.buffer_length = 0;
            } else {
              // Using input stream in last iteration. When decoder switches to input
              // stream it has less than 8 bits in accamulator, so it is safe to
              // return unused accamulator bits there.
              bit_reader::BrotliBitReaderUnload(&mut s.br);
              *available_in = s.br.avail_in as usize;
              *input_offset = s.br.next_in as usize;
            }
          }
        }
        break;
      }
    }
    loop {
      // this emulates fallthrough behavior
      match s.state {
        BrotliRunningState::BROTLI_STATE_UNINITED => {
          // Prepare to the first read.
          if (!bit_reader::BrotliWarmupBitReader(&mut s.br, local_input)) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            break;
          }
          // Decode window size.
          /* Reads 1..8 bits. */
          result = DecodeWindowBits(&mut s.large_window, &mut s.window_bits, &mut s.br);
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          if s.large_window {
              s.state = BrotliRunningState::BROTLI_STATE_LARGE_WINDOW_BITS;
          } else {
              s.state = BrotliRunningState::BROTLI_STATE_INITIALIZE;
          }
        }
        BrotliRunningState::BROTLI_STATE_LARGE_WINDOW_BITS => {
          if (!bit_reader::BrotliSafeReadBits(&mut s.br, 6, &mut s.window_bits, local_input)) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            break;
          }
          if (s.window_bits < kBrotliLargeMinWbits ||
              s.window_bits > kBrotliLargeMaxWbits) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_WINDOW_BITS;
            break;
          }
          s.state = BrotliRunningState::BROTLI_STATE_INITIALIZE;
        }
        BrotliRunningState::BROTLI_STATE_INITIALIZE => {
          s.max_backward_distance = (1 << s.window_bits) - kBrotliWindowGap as i32;
          s.max_backward_distance_minus_custom_dict_size = (s.max_backward_distance as isize -
                                                           s.custom_dict_size) as i32;

          // (formerly) Allocate memory for both block_type_trees and block_len_trees.
          s.block_type_length_state.block_type_trees = s.alloc_hc
            .alloc_cell(3 * huffman::BROTLI_HUFFMAN_MAX_TABLE_SIZE as usize);
          if (s.block_type_length_state.block_type_trees.slice().len() == 0) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_ALLOC_BLOCK_TYPE_TREES;
            break;
          }
          s.block_type_length_state.block_len_trees = s.alloc_hc
            .alloc_cell(3 * huffman::BROTLI_HUFFMAN_MAX_TABLE_SIZE as usize);

          s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_BEGIN;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_METABLOCK_BEGIN => {
          s.BrotliStateMetablockBegin();
          BROTLI_LOG_UINT!(s.pos);
          s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_HEADER;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_METABLOCK_HEADER => {
          result = DecodeMetaBlockLength(&mut s, local_input); // Reads 2 - 31 bits.
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          BROTLI_LOG_UINT!(s.is_last_metablock);
          BROTLI_LOG_UINT!(s.meta_block_remaining_len);
          BROTLI_LOG_UINT!(s.is_metadata);
          BROTLI_LOG_UINT!(s.is_uncompressed);
          if (s.is_metadata != 0 || s.is_uncompressed != 0) &&
             !bit_reader::BrotliJumpToByteBoundary(&mut s.br) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_PADDING_2;
            break;
          }
          if s.is_metadata != 0 {
            s.state = BrotliRunningState::BROTLI_STATE_METADATA;
            break;
          }
          if s.meta_block_remaining_len == 0 {
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
            break;
          }
          if s.ringbuffer.slice().len() == 0 && !BrotliAllocateRingBuffer(&mut s, local_input) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_ALLOC_RING_BUFFER_2;
            break;
          }
          if s.is_uncompressed != 0 {
            s.state = BrotliRunningState::BROTLI_STATE_UNCOMPRESSED;
            break;
          }
          s.loop_counter = 0;
          s.state = BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_0;
          break;
        }
        BrotliRunningState::BROTLI_STATE_UNCOMPRESSED => {
          let mut _bytes_copied = s.meta_block_remaining_len;
          result = CopyUncompressedBlockToOutput(&mut available_out,
                                                 &mut output,
                                                 &mut output_offset,
                                                 &mut total_out,
                                                 &mut s,
                                                 local_input);
          _bytes_copied -= s.meta_block_remaining_len;
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
          break;
        }
        BrotliRunningState::BROTLI_STATE_METADATA => {
          while s.meta_block_remaining_len > 0 {
            let mut bits = 0u32;
            // Read one byte and ignore it.
            if !bit_reader::BrotliSafeReadBits(&mut s.br, 8, &mut bits, local_input) {
              result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
              break;
            }
            s.meta_block_remaining_len -= 1;
          }
          if let BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS = result {
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE
          }
          break;
        }
        BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_0 => {
          if s.loop_counter >= 3 {
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_HEADER_2;
            break;
          }
          // Reads 1..11 bits.
          {
            let index = s.loop_counter as usize;
            result =
              DecodeVarLenUint8(&mut s.substate_decode_uint8,
                                &mut s.br,
                                &mut fast_mut!((s.block_type_length_state.num_block_types)[index]),
                                local_input);
          }
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          fast_mut!((s.block_type_length_state.num_block_types)[s.loop_counter as usize]) += 1;
          BROTLI_LOG_UINT!(s.block_type_length_state.num_block_types[s.loop_counter as usize]);
          if fast!((s.block_type_length_state.num_block_types)[s.loop_counter as usize]) < 2 {
            s.loop_counter += 1;
            break;
          }
          s.state = BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_1;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_1 => {
          let tree_offset = s.loop_counter as u32 * huffman::BROTLI_HUFFMAN_MAX_TABLE_SIZE as u32;
          let mut new_huffman_table = mem::replace(&mut s.block_type_length_state.block_type_trees,
                                                   AllocHC::AllocatedMemory::default());
          let loop_counter = s.loop_counter as usize;
          let alphabet_size = fast!((s.block_type_length_state.num_block_types)[loop_counter]) + 2;
          result =
            ReadHuffmanCode(alphabet_size, alphabet_size,
                            new_huffman_table.slice_mut(),
                            tree_offset as usize,
                            None,
                            &mut s,
                            local_input);
          let _ = mem::replace(&mut s.block_type_length_state.block_type_trees,
                       new_huffman_table);
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          s.state = BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_2;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_2 => {
          let tree_offset = s.loop_counter * huffman::BROTLI_HUFFMAN_MAX_TABLE_SIZE as i32;
          let mut new_huffman_table = mem::replace(&mut s.block_type_length_state.block_len_trees,
                                                   AllocHC::AllocatedMemory::default());
          result = ReadHuffmanCode(kNumBlockLengthCodes, kNumBlockLengthCodes,
                                   new_huffman_table.slice_mut(),
                                   tree_offset as usize,
                                   None,
                                   &mut s,
                                   local_input);
          let _ = mem::replace(&mut s.block_type_length_state.block_len_trees,
                       new_huffman_table);
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          s.state = BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_3;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_3 => {
          let tree_offset = s.loop_counter * huffman::BROTLI_HUFFMAN_MAX_TABLE_SIZE as i32;

          let mut block_length_out: u32 = 0;
          let ind_ret: (bool, u32);
          
          ind_ret = SafeReadBlockLengthIndex(&s.block_type_length_state.substate_read_block_length,
                                             s.block_type_length_state.block_length_index,
                                             fast_slice!((s.block_type_length_state.block_len_trees)
                                                           [tree_offset as usize;]),
                                             &mut s.br, local_input);

          if !SafeReadBlockLengthFromIndex(&mut s.block_type_length_state,
                                           &mut s.br,
                                           &mut block_length_out,
                                           ind_ret,
                                           local_input) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            break;
          }
          fast_mut!((s.block_type_length_state.block_length)[s.loop_counter as usize]) =
            block_length_out;
          BROTLI_LOG_UINT!(s.block_type_length_state.block_length[s.loop_counter as usize]);
          s.loop_counter += 1;
          s.state = BrotliRunningState::BROTLI_STATE_HUFFMAN_CODE_0;
          break;
        }
        BrotliRunningState::BROTLI_STATE_METABLOCK_HEADER_2 => {
          let mut bits: u32 = 0;
          if (!bit_reader::BrotliSafeReadBits(&mut s.br, 6, &mut bits, local_input)) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT;
            break;
          }
          s.distance_postfix_bits = bits & bit_reader::BitMask(2);
          bits >>= 2;
          s.num_direct_distance_codes = NUM_DISTANCE_SHORT_CODES +
                                        (bits << s.distance_postfix_bits);
          BROTLI_LOG_UINT!(s.num_direct_distance_codes);
          BROTLI_LOG_UINT!(s.distance_postfix_bits);
          s.distance_postfix_mask = bit_reader::BitMask(s.distance_postfix_bits) as i32;
          s.context_modes = s.alloc_u8
            .alloc_cell(fast!((s.block_type_length_state.num_block_types)[0]) as usize);
          if (s.context_modes.slice().len() == 0) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_ALLOC_CONTEXT_MODES;
            break;
          }
          s.loop_counter = 0;
          s.state = BrotliRunningState::BROTLI_STATE_CONTEXT_MODES;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_CONTEXT_MODES => {
          result = ReadContextModes(&mut s, local_input);
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          s.state = BrotliRunningState::BROTLI_STATE_CONTEXT_MAP_1;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_CONTEXT_MAP_1 => {
          result =
            DecodeContextMap((fast!((s.block_type_length_state.num_block_types)[0]) as usize) <<
                             kLiteralContextBits as usize,
                             false,
                             &mut s,
                             local_input);
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          DetectTrivialLiteralBlockTypes(s);
          s.state = BrotliRunningState::BROTLI_STATE_CONTEXT_MAP_2;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_CONTEXT_MAP_2 => {
            let num_direct_codes =
              s.num_direct_distance_codes - NUM_DISTANCE_SHORT_CODES;
            let num_distance_codes = BROTLI_DISTANCE_ALPHABET_SIZE(
              s.distance_postfix_bits, num_direct_codes,
                (if s.large_window { BROTLI_LARGE_MAX_DISTANCE_BITS } else {
                    BROTLI_MAX_DISTANCE_BITS}));
            let max_distance_symbol = if s.large_window {
                BrotliMaxDistanceSymbol(
                    num_direct_codes, s.distance_postfix_bits)
            } else {
                num_distance_codes
            };
            result =
              DecodeContextMap((fast!((s.block_type_length_state.num_block_types)[2]) as usize) <<
                               kDistanceContextBits as usize,
                               true,
                               s,
                               local_input);
            match result {
              BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
              _ => break,
            }
            s.literal_hgroup.init(&mut s.alloc_u32,
                                  &mut s.alloc_hc,
                                  kNumLiteralCodes,
                                  kNumLiteralCodes,
                                  s.num_literal_htrees as u16);
            s.insert_copy_hgroup.init(&mut s.alloc_u32,
                                      &mut s.alloc_hc,
                                      kNumInsertAndCopyCodes,
                                      kNumInsertAndCopyCodes,
                                      fast!((s.block_type_length_state.num_block_types)[1]) as u16);
            s.distance_hgroup.init(&mut s.alloc_u32,
                                   &mut s.alloc_hc,
                                   num_distance_codes as u16,
                                   max_distance_symbol as u16,
                                   s.num_dist_htrees as u16);
            if (s.literal_hgroup.codes.slice().len() == 0 ||
                s.insert_copy_hgroup.codes.slice().len() == 0 ||
                s.distance_hgroup.codes.slice().len() == 0) {
              return SaveErrorCode!(s, BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_UNREACHABLE);
            }

          /*{
            let num_distance_codes: u32 = s.num_direct_distance_codes +
                                          (48u32 << s.distance_postfix_bits);
            result =
              DecodeContextMap((fast!((s.block_type_length_state.num_block_types)[2]) as usize) <<
                               kDistanceContextBits as usize,
                               true,
                               s,
                               local_input);
            match result {
              BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
              _ => break,
            }
            s.literal_hgroup.init(&mut s.alloc_u32,
                                  &mut s.alloc_hc,
                                  kNumLiteralCodes,
                                  s.num_literal_htrees as u16);
            s.insert_copy_hgroup.init(&mut s.alloc_u32,
                                      &mut s.alloc_hc,
                                      kNumInsertAndCopyCodes,
                                      fast!((s.block_type_length_state.num_block_types)[1]) as u16);
            s.distance_hgroup.init(&mut s.alloc_u32,
                                   &mut s.alloc_hc,
                                   num_distance_codes as u16,
                                   s.num_dist_htrees as u16);
            if (s.literal_hgroup.codes.slice().len() == 0 ||
                s.insert_copy_hgroup.codes.slice().len() == 0 ||
                s.distance_hgroup.codes.slice().len() == 0) {
              return SaveErrorCode!(s, BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_ALLOC_TREE_GROUPS);
            }
          }*/
          s.loop_counter = 0;
          s.state = BrotliRunningState::BROTLI_STATE_TREE_GROUP;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_TREE_GROUP => {
          result = HuffmanTreeGroupDecode(s.loop_counter, &mut s, local_input);
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          s.loop_counter += 1;
          if (s.loop_counter >= 3) {
            PrepareLiteralDecoding(s);
            s.dist_context_map_slice_index = 0;
              /*
            s.context_map_slice_index = 0;
            let context_mode_index = fast!((s.block_type_length_state.block_type_rb)[1]);
            let context_mode = fast_slice!((s.context_modes)[context_mode_index as usize]);
            s.context_lookup = &kContextLookup[context_mode as usize & 3];
               */
            s.htree_command_index = 0;
            // look it up each time s.literal_htree=s.literal_hgroup.htrees[s.literal_htree_index];
            s.state = BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN;
          }
          break;
        }
        BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN |
        BrotliRunningState::BROTLI_STATE_COMMAND_INNER |
        BrotliRunningState::BROTLI_STATE_COMMAND_POST_DECODE_LITERALS |
        BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRAP_COPY => {
          result = ProcessCommands(s, local_input);
          if let BrotliDecoderErrorCode::BROTLI_DECODER_NEEDS_MORE_INPUT = result {
            result = SafeProcessCommands(s, local_input)
          }
          break;
        }
        BrotliRunningState::BROTLI_STATE_COMMAND_INNER_WRITE |
        BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRITE_1 |
        BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRITE_2 => {
          let (xresult, _) = WriteRingBuffer(&mut available_out,
                                             Some(&mut output),
                                             &mut output_offset,
                                             &mut total_out,
                                             false,
                                             &mut s);
          result = xresult;
          match result {
            BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
            _ => break,
          }
          WrapRingBuffer(s);
          if s.ringbuffer_size == 1 << s.window_bits {
            s.max_distance = s.max_backward_distance;
          }
          match s.state {
            BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRITE_1 => {
              if (s.meta_block_remaining_len <= 0) {
                // Next metablock, if any
                s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
              } else {
                s.state = BrotliRunningState::BROTLI_STATE_COMMAND_BEGIN;
              }
              break;
            }
            BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRITE_2 => {
              s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_WRAP_COPY;
            }
            _ => {
              // BROTLI_STATE_COMMAND_INNER_WRITE
              if (s.loop_counter == 0) {
                if (s.meta_block_remaining_len <= 0) {
                  s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_DONE;
                } else {
                  s.state = BrotliRunningState::BROTLI_STATE_COMMAND_POST_DECODE_LITERALS;
                }
                break;
              }
              s.state = BrotliRunningState::BROTLI_STATE_COMMAND_INNER;
            }
          }
          break;
        }
        BrotliRunningState::BROTLI_STATE_METABLOCK_DONE => {
          s.BrotliStateCleanupAfterMetablock();
          if (s.is_last_metablock == 0) {
            s.state = BrotliRunningState::BROTLI_STATE_METABLOCK_BEGIN;
            break;
          }
          if (!bit_reader::BrotliJumpToByteBoundary(&mut s.br)) {
            result = BrotliDecoderErrorCode::BROTLI_DECODER_ERROR_FORMAT_PADDING_2;
          }
          if (s.buffer_length == 0) {
            bit_reader::BrotliBitReaderUnload(&mut s.br);
            *available_in = s.br.avail_in as usize;
            *input_offset = s.br.next_in as usize;
          }
          s.state = BrotliRunningState::BROTLI_STATE_DONE;
          // No break, continue to next state
        }
        BrotliRunningState::BROTLI_STATE_DONE => {
          if (s.ringbuffer.slice().len() != 0) {
            let (xresult, _) = WriteRingBuffer(&mut available_out,
                                               Some(&mut output),
                                               &mut output_offset,
                                               &mut total_out,
                                               true,
                                               &mut s);
            result = xresult;
            match result {
              BrotliDecoderErrorCode::BROTLI_DECODER_SUCCESS => {}
              _ => break,
            }
          }
          return SaveErrorCode!(s, result);
        }
      }
    }
  }

  SaveErrorCode!(s, result)
}

