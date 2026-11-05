#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
mod tests;
use ::core;
use alloc;
use alloc::Allocator;
use alloc::SliceWrapper;
use alloc::SliceWrapperMut;
use core::default::Default;
pub const BROTLI_HUFFMAN_MAX_CODE_LENGTH: usize = 15;

// For current format this constant equals to kNumInsertAndCopyCodes
pub const BROTLI_HUFFMAN_MAX_CODE_LENGTHS_SIZE: usize = 704;

// Maximum possible Huffman table size for an alphabet size of (index * 32),
// max code length 15 and root table bits 8.
// pub const kMaxHuffmanTableSize : [u16;23] = [
// 256, 402, 436, 468, 500, 534, 566, 598, 630, 662, 694, 726, 758, 790, 822,
// 854, 886, 920, 952, 984, 1016, 1048, 1080, 1112, 1144,1176,1208,1240,272,
// 1304, 1336, 1368, 1400, 1432, 1464, 1496, 1528];
// pub const BROTLI_HUFFMAN_MAX_SIZE_26 : u32 = 396;
// pub const BROTLI_HUFFMAN_MAX_SIZE_258 : u32 = 632;
// pub const BROTLI_HUFFMAN_MAX_SIZE_272 : u32 = 646;
//
pub const BROTLI_HUFFMAN_MAX_TABLE_SIZE: u32 = 1080;
pub const BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH: u32 = 5;

#[repr(C)]
#[derive(PartialEq, Copy, Clone, Debug)]
pub struct HuffmanCode {
  pub value: u16, // symbol value or table offset
  pub bits: u8, // number of bits used for this symbol
}

impl HuffmanCode {
  pub fn eq(&self, other: &Self) -> bool {
    self.value == other.value && self.bits == other.bits
  }
}

impl Default for HuffmanCode {
  fn default() -> Self {
    HuffmanCode {
      value: 0,
      bits: 0,
    }
  }
}

// Contains a collection of Huffman trees with the same alphabet size.
pub struct HuffmanTreeGroup<Alloc32: Allocator<u32>, AllocHC: Allocator<HuffmanCode>> {
  pub htrees: Alloc32::AllocatedMemory,
  pub codes: AllocHC::AllocatedMemory,
  pub alphabet_size: u16,
  pub max_symbol: u16,
  pub num_htrees: u16,
}

impl<AllocU32 : alloc::Allocator<u32>,
     AllocHC : alloc::Allocator<HuffmanCode> > HuffmanTreeGroup<AllocU32, AllocHC> {
    pub fn init(self : &mut Self, mut alloc_u32 : &mut AllocU32, mut alloc_hc : &mut AllocHC,
                alphabet_size : u16, max_symbol: u16, ntrees : u16) {
        self.reset(&mut alloc_u32, &mut alloc_hc);
        self.alphabet_size = alphabet_size;
        self.max_symbol = max_symbol;
        self.num_htrees = ntrees;
        let nt = ntrees as usize;
        let _ = core::mem::replace(&mut self.htrees,
                           alloc_u32.alloc_cell(nt));
        let _ = core::mem::replace(&mut self.codes,
                           alloc_hc.alloc_cell(nt * BROTLI_HUFFMAN_MAX_TABLE_SIZE as usize));
    }

//  pub fn get_tree_mut<'a>(self :&'a mut Self, index : u32, mut tree_out : &'a mut [HuffmanCode]) {
//        let start : usize = fast!((self.htrees)[index as usize]) as usize;
//        let _ = core::mem::replace(&mut tree_out, fast_mut!((self.codes.slice_mut())[start;]));
//    }
//    pub fn get_tree<'a>(self :&'a Self, index : u32, mut tree_out : &'a [HuffmanCode]) {
//        let start : usize = fast!((self.htrees)[index as usize]) as usize;
//        let _ = core::mem::replace(&mut tree_out, fast_slice!((self.codes)[start;]));
//    }
    #[allow(dead_code)]
    pub fn get_tree_mut(&mut self, index : u32) -> &mut [HuffmanCode] {
        let start : usize = fast_slice!((self.htrees)[index as usize]) as usize;
        fast_mut!((self.codes.slice_mut())[start;])
    }
    #[allow(dead_code)]
    pub fn get_tree(&self, index : u32) -> &[HuffmanCode] {
        let start : usize = fast_slice!((self.htrees)[index as usize]) as usize;
        fast_slice!((self.codes)[start;])
    }
    pub fn reset(self : &mut Self, alloc_u32 : &mut AllocU32, alloc_hc : &mut AllocHC) {
        alloc_u32.free_cell(core::mem::replace(&mut self.htrees,
                                               AllocU32::AllocatedMemory::default()));
        alloc_hc.free_cell(core::mem::replace(&mut self.codes,
                                              AllocHC::AllocatedMemory::default()));

// for mut iter in self.htrees[0..self.num_htrees as usize].iter_mut() {
//    if iter.slice().len() > 0 {
//        alloc_hc.free_cell(core::mem::replace(&mut iter,
//                                              AllocHC::AllocatedMemory::default()));
//    }
// }

    }
    pub fn build_hgroup_cache(&self) -> [&[HuffmanCode]; 256] {
      let mut ret : [&[HuffmanCode]; 256] = [&[]; 256];
      let mut index : usize = 0;
      for htree in self.htrees.slice() {
          ret[index] = fast_slice!((&self.codes)[*htree as usize ; ]);
          index += 1;
      }
      ret
    }
}

impl<AllocU32 : alloc::Allocator<u32>,
     AllocHC : alloc::Allocator<HuffmanCode> > Default for HuffmanTreeGroup<AllocU32, AllocHC> {
    fn default() -> Self {
        HuffmanTreeGroup::<AllocU32, AllocHC> {
          htrees : AllocU32::AllocatedMemory::default(),
          codes : AllocHC::AllocatedMemory::default(),
          max_symbol: 0,
          alphabet_size : 0,
          num_htrees : 0,
        }
    }
}



const BROTLI_REVERSE_BITS_MAX: usize = 8;

const BROTLI_REVERSE_BITS_BASE: u8 = 0;
const kReverseBits: [u8; (1 << BROTLI_REVERSE_BITS_MAX)] =
  [0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
   0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
   0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
   0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
   0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
   0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
   0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
   0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
   0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
   0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
   0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
   0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
   0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
   0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
   0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
   0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF];

const BROTLI_REVERSE_BITS_LOWEST: u32 =
  (1u32 << (BROTLI_REVERSE_BITS_MAX as u32 - 1 + BROTLI_REVERSE_BITS_BASE as u32));

// Returns reverse(num >> BROTLI_REVERSE_BITS_BASE, BROTLI_REVERSE_BITS_MAX),
// where reverse(value, len) is the bit-wise reversal of the len least
// significant bits of value.
fn BrotliReverseBits(num: u32) -> u32 {
  fast!((kReverseBits)[num as usize]) as u32
}

// Stores code in table[0], table[step], table[2*step], ..., table[end]
// Assumes that end is an integer multiple of step
fn ReplicateValue(table: &mut [HuffmanCode],
                  offset: u32,
                  step: i32,
                  mut end: i32,
                  code: HuffmanCode) {
  loop {
    end -= step;
    fast_mut!((table)[offset as usize + end as usize]) = code;
    if end <= 0 {
      break;
    }
  }
}

// Returns the table width of the next 2nd level table. count is the histogram
// of bit lengths for the remaining symbols, len is the code length of the next
// processed symbol
fn NextTableBitSize(count: &[u16], mut len: i32, root_bits: i32) -> i32 {
  let mut left: i32 = 1 << (len - root_bits);
  while len < BROTLI_HUFFMAN_MAX_CODE_LENGTH as i32 {
    left -= fast!((count)[len as usize]) as i32;
    if left <= 0 {
      break;
    }
    len += 1;
    left <<= 1;
  }
  len - root_bits
}


pub fn BrotliBuildCodeLengthsHuffmanTable(mut table: &mut [HuffmanCode],
                                          code_lengths: &[u8],
                                          count: &[u16]) {
  let mut sorted: [i32; 18] = fast_uninitialized![18];     /* symbols sorted by code length */
  // offsets in sorted table for each length
  let mut offset: [i32; (BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH + 1) as usize] =
    fast_uninitialized![(BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH + 1) as usize];
  assert!(BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH as usize <= BROTLI_REVERSE_BITS_MAX as usize);

  // generate offsets into sorted symbol table by code length
  let mut symbol: i32 = -1;         /* symbol index in original or sorted table */
  let mut bits: i32 = 1;
  for _ in 0..BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH {
    symbol += fast!((count)[bits as usize]) as i32;
    fast_mut!((offset)[bits as usize]) = symbol;
    bits += 1;
  }
  // Symbols with code length 0 are placed after all other symbols.
  fast_mut!((offset)[0]) = 17;

  // sort symbols by length, by symbol order within each length
  symbol = 18;
  loop {
    for _ in 0..6 {
      symbol -= 1;
      let index = fast!((offset)[fast_inner!((code_lengths)[symbol as usize]) as usize]);
      fast_mut!((offset)[fast_inner!((code_lengths)[symbol as usize]) as usize]) -= 1;
      fast_mut!((sorted)[index as usize]) = symbol;
    }
    if symbol == 0 {
      break;
    }
  }

  const table_size: i32 = 1 << BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH;

  // Special case: all symbols but one have 0 code length.
  if fast!((offset)[0]) == 0 {
    let code: HuffmanCode = HuffmanCode {
      bits: 0,
      value: fast!((sorted)[0]) as u16,
    };
    for val in fast_mut!((table)[0 ; table_size as usize]).iter_mut() {
      *val = code;
    }
    return;
  }

  // fill in table
  let mut key: u32 = 0; /* prefix code */
  let mut key_step: u32 = BROTLI_REVERSE_BITS_LOWEST; /* prefix code addend */
  symbol = 0;
  bits = 1;
  let mut step: i32 = 2;
  loop {
    let mut code: HuffmanCode = HuffmanCode {
      bits: (bits as u8),
      value: 0,
    };
    let mut bits_count: i32 = fast!((count)[bits as usize]) as i32;

    while bits_count != 0 {
      code.value = fast!((sorted)[symbol as usize]) as u16;
      symbol += 1;
      ReplicateValue(&mut table, BrotliReverseBits(key), step, table_size, code);
      key += key_step;
      bits_count -= 1;
    }
    step <<= 1;
    key_step >>= 1;
    bits += 1;
    if !(bits <= BROTLI_HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH as i32) {
      break;
    }
  }
}

pub fn BrotliBuildHuffmanTable(mut root_table: &mut [HuffmanCode],
                               root_bits: i32,
                               symbol_lists: &[u16],
                               symbol_lists_offset: usize, /* need negative-index to symbol_lists */
                               count: &mut [u16])
                               -> u32 {
  let mut code: HuffmanCode = HuffmanCode {
    bits: 0,
    value: 0,
  };       /* current table entry */
  let mut max_length: i32 = -1;

  assert!(root_bits as isize <= BROTLI_REVERSE_BITS_MAX as isize);
  assert!(BROTLI_HUFFMAN_MAX_CODE_LENGTH as isize - root_bits as isize <=
          BROTLI_REVERSE_BITS_MAX as isize);

  while fast!((symbol_lists)[((symbol_lists_offset as isize) + max_length as isize) as usize]) ==
        0xFFFF {
    max_length -= 1;
  }
  max_length += BROTLI_HUFFMAN_MAX_CODE_LENGTH as i32 + 1;

  let mut table_free_offset: u32 = 0;
  let mut table_bits: i32 = root_bits;      /* key length of current table */
  let mut table_size: i32 = 1 << table_bits;/* size of current table */
  let mut total_size: i32 = table_size;     /* sum of root table size and 2nd level table sizes */

  // fill in root table
  // let's reduce the table size to a smaller size if possible, and
  // create the repetitions by memcpy if possible in the coming loop
  if table_bits > max_length {
    table_bits = max_length;
    table_size = 1 << table_bits;
  }
  let mut key: u32 = 0; /* prefix code */
  let mut key_step: u32 = BROTLI_REVERSE_BITS_LOWEST; /* prefix code addend */
  let mut bits: i32 = 1;
  let mut step: i32 = 2; /* step size to replicate values in current table */
  loop {
    code.bits = bits as u8;
    let mut symbol: i32 = bits - (BROTLI_HUFFMAN_MAX_CODE_LENGTH as i32 + 1);
    let mut bits_count: i32 = fast!((count)[bits as usize]) as i32;
    while bits_count != 0 {
      symbol =
        fast!((symbol_lists)[(symbol_lists_offset as isize + symbol as isize) as usize]) as i32;
      code.value = symbol as u16;
      ReplicateValue(&mut root_table,
                     table_free_offset + BrotliReverseBits(key),
                     step,
                     table_size,
                     code);
      key += key_step;
      bits_count -= 1;
    }
    step <<= 1;
    key_step >>= 1;
    bits += 1;
    if !(bits <= table_bits) {
      break;
    }
  }

  // if root_bits != table_bits we only created one fraction of the
  // table, and we need to replicate it now.
  while total_size != table_size {
    for index in 0..table_size {
      // FIXME: did I get this right?
      fast_mut!((root_table)[table_free_offset as usize + table_size as usize + index as usize]) =
        fast!((root_table)[table_free_offset as usize + index as usize]);
    }
    table_size <<= 1;
  }

  // fill in 2nd level tables and add pointers to root table
  key_step = BROTLI_REVERSE_BITS_LOWEST >> (root_bits - 1);
  let mut sub_key: u32 = BROTLI_REVERSE_BITS_LOWEST << 1;       /* 2nd level table prefix code */
  let mut sub_key_step: u32 = BROTLI_REVERSE_BITS_LOWEST;   /* 2nd level table prefix code addend */

  step = 2;

  let mut len: i32 = root_bits + 1; /* current code length */
  while len <= max_length {
    let mut symbol: i32 = len - (BROTLI_HUFFMAN_MAX_CODE_LENGTH as i32 + 1);
    while fast!((count)[len as usize]) != 0 {
      if sub_key == (BROTLI_REVERSE_BITS_LOWEST << 1u32) {
        table_free_offset += table_size as u32;
        table_bits = NextTableBitSize(count, len, root_bits);
        table_size = 1 << table_bits;
        total_size += table_size;
        sub_key = BrotliReverseBits(key);
        key += key_step;
        fast_mut!((root_table)[sub_key as usize]).bits = (table_bits + root_bits) as u8;
        fast_mut!((root_table)[sub_key as usize]).value =
          ((table_free_offset as usize) - sub_key as usize) as u16;
        sub_key = 0;
      }
      code.bits = (len - root_bits) as u8;
      symbol =
        fast!((symbol_lists)[(symbol_lists_offset as isize + symbol as isize) as usize]) as i32;
      code.value = symbol as u16;
      ReplicateValue(&mut root_table,
                     table_free_offset + BrotliReverseBits(sub_key),
                     step,
                     table_size,
                     code);
      sub_key += sub_key_step;
      fast_mut!((count)[len as usize]) -= 1;
    }
    step <<= 1;
    sub_key_step >>= 1;
    len += 1
  }
  total_size as u32
}



pub fn BrotliBuildSimpleHuffmanTable(table: &mut [HuffmanCode],
                                     root_bits: i32,
                                     val: &[u16],
                                     num_symbols: u32)
                                     -> u32 {
  let mut table_size: u32 = 1;
  let goal_size: u32 = 1u32 << root_bits;
  assert!(num_symbols <= 4);
  if num_symbols == 0 {
    fast_mut!((table)[0]).bits = 0;
    fast_mut!((table)[0]).value = fast!((val)[0]);
  } else if num_symbols == 1 {
    fast_mut!((table)[0]).bits = 1;
    fast_mut!((table)[1]).bits = 1;
    if fast!((val)[1]) > fast!((val)[0]) {
      fast_mut!((table)[0]).value = fast!((val)[0]);
      fast_mut!((table)[1]).value = fast!((val)[1]);
    } else {
      fast_mut!((table)[0]).value = fast!((val)[1]);
      fast_mut!((table)[1]).value = fast!((val)[0]);
    }
    table_size = 2;
  } else if num_symbols == 2 {
    fast_mut!((table)[0]).bits = 1;
    fast_mut!((table)[0]).value = fast!((val)[0]);
    fast_mut!((table)[2]).bits = 1;
    fast_mut!((table)[2]).value = fast!((val)[0]);
    if fast!((val)[2]) > fast!((val)[1]) {
      fast_mut!((table)[1]).value = fast!((val)[1]);
      fast_mut!((table)[3]).value = fast!((val)[2]);
    } else {
      fast_mut!((table)[1]).value = fast!((val)[2]);
      fast_mut!((table)[3]).value = fast!((val)[1]);
    }
    fast_mut!((table)[1]).bits = 2;
    fast_mut!((table)[3]).bits = 2;
    table_size = 4;
  } else if num_symbols == 3 {
    let last: u16 = if val.len() > 3 { fast!((val)[3]) } else { 65535 };
    let mut mval: [u16; 4] = [fast!((val)[0]), fast!((val)[1]), fast!((val)[2]), last];
    for i in 0..3 {
      for k in i + 1..4 {
        if mval[k] < mval[i] {
          mval.swap(k, i);
        }
      }
    }
    for i in 0..4 {
      fast_mut!((table)[i]).bits = 2;
    }
    fast_mut!((table)[0]).value = mval[0];
    fast_mut!((table)[2]).value = mval[1];
    fast_mut!((table)[1]).value = mval[2];
    fast_mut!((table)[3]).value = mval[3];
    table_size = 4;
  } else if num_symbols == 4 {
    let mut mval: [u16; 4] = [fast!((val)[0]), fast!((val)[1]), fast!((val)[2]), fast!((val)[3])];
    if mval[3] < mval[2] {
      mval.swap(3, 2)
    }
    for i in 0..7 {
      fast_mut!((table)[i]).value = mval[0];
      fast_mut!((table)[i]).bits = (1 + (i & 1)) as u8;
    }
    fast_mut!((table)[1]).value = mval[1];
    fast_mut!((table)[3]).value = mval[2];
    fast_mut!((table)[5]).value = mval[1];
    fast_mut!((table)[7]).value = mval[3];
    fast_mut!((table)[3]).bits = 3;
    fast_mut!((table)[7]).bits = 3;
    table_size = 8;
  } else {
    assert!(false);
  }
  while table_size != goal_size {
    for index in 0..table_size {
      fast_mut!((table)[(table_size + index) as usize]) = fast!((table)[index as usize]);
    }
    table_size <<= 1;
  }
  goal_size
}
