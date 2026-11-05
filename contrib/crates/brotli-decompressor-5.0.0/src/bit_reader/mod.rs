#![allow(non_snake_case)]
use core::default::Default;
macro_rules! xprintln (
  ($a : expr) => ();
  ($a : expr, $b : expr) => ();
  ($a : expr, $b : expr, $c : expr) => ();
  ($a : expr, $b : expr, $c : expr, $d : expr) => ();
  ($a : expr, $b : expr, $c : expr, $d : expr, $e : expr) => ();
  ($a : expr, $b : expr, $c : expr, $d : expr, $e : expr, $f : expr) => ();
);

pub const BROTLI_SHORT_FILL_BIT_WINDOW_READ: u32 = 4;

#[allow(non_camel_case_types)]
pub type reg_t = u64;

#[allow(non_upper_case_globals)]
const kBitMask: [u32; 33] =
  [0x0000, 0x00000001, 0x00000003, 0x00000007, 0x0000000F, 0x0000001F, 0x0000003F, 0x0000007F,
   0x000000FF, 0x000001FF, 0x000003FF, 0x000007FF, 0x00000FFF, 0x00001FFF, 0x00003FFF, 0x00007FFF,
   0x0000FFFF, 0x0001FFFF, 0x0003FFFF, 0x0007FFFF, 0x000FFFFF, 0x001FFFFF, 0x003FFFFF, 0x007FFFFF,
   0x00FFFFFF, 0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF, 0x0FFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF,
   0xFFFFFFFF];

#[inline]
pub fn BitMask(n: u32) -> u32 {
  if false {
    // Masking with this expression turns to a single
    // "Unsigned Bit Field Extract" UBFX instruction on ARM.
    !((0xffffffffu32) << n)
  } else {
    fast!((kBitMask)[n as usize])
  }
}

pub struct BrotliBitReader {
  pub val_: reg_t, // pre-fetched bits
  pub bit_pos_: u32, // current bit-reading position in val_
  pub next_in: u32, // the byte we're reading from
  pub avail_in: u32,
}

impl Default for BrotliBitReader {
  fn default() -> Self {
    BrotliBitReader {
      val_: 0,
      bit_pos_: 0,
      next_in: 0,
      avail_in: 0,
    }
  }
}

pub struct BrotliBitReaderState {
  pub val_: reg_t, // pre-fetched bits
  pub bit_pos_: u32, // current bit-reading position in val_
  pub next_in: u32, // the byte we're reading from
  pub avail_in: u32,
}
impl Default for BrotliBitReaderState {
  #[inline]
  fn default() -> Self {
    BrotliBitReaderState {
      val_: 0,
      bit_pos_: 0,
      next_in: 0,
      avail_in: 0,
    }
  }
}

pub fn BrotliBitReaderSaveState(from: &BrotliBitReader) -> BrotliBitReaderState {
  BrotliBitReaderState {
    val_: from.val_,
    bit_pos_: from.bit_pos_,
    next_in: from.next_in,
    avail_in: from.avail_in,
  }
}

pub fn BrotliBitReaderRestoreState(to: &mut BrotliBitReader, from: &BrotliBitReaderState) {
  to.val_ = from.val_;
  to.bit_pos_ = from.bit_pos_;
  to.next_in = from.next_in;
  to.avail_in = from.avail_in;
}

pub fn BrotliGetAvailableBits(br: &BrotliBitReader) -> u32 {
  ((::core::mem::size_of::<reg_t>() as u32) << 3) - br.bit_pos_
}

// Returns amount of unread bytes the bit reader still has buffered from the
// BrotliInput, including whole bytes in br->val_.
pub fn BrotliGetRemainingBytes(br: &BrotliBitReader) -> u32 {
  br.avail_in + (BrotliGetAvailableBits(br) >> 3)
}

// Checks if there is at least num bytes left in the input ringbuffer (excluding
// the bits remaining in br->val_).
pub fn BrotliCheckInputAmount(br: &BrotliBitReader, num: u32) -> bool {
  br.avail_in >= num
}


#[inline(always)]
fn BrotliLoad16LE(input: &[u8], next_in_u32: u32) -> u16 {
  let next_in: usize = next_in_u32 as usize;
  (fast!((input)[next_in]) as u16) | ((fast!((input)[next_in + 1]) as u16) << 8)
}


#[inline(always)]
fn BrotliLoad32LE(input: &[u8], next_in_u32: u32) -> u32 {
  let next_in: usize = next_in_u32 as usize;
  let mut four_byte: [u8; 4] = fast_uninitialized![4];
  four_byte.clone_from_slice(fast!((input)[next_in ; next_in + 4]));
  (four_byte[0] as u32) | ((four_byte[1] as u32) << 8) | ((four_byte[2] as u32) << 16) |
      ((four_byte[3] as u32) << 24)
}

#[inline(always)]
fn BrotliLoad64LE(input: &[u8], next_in_u32: u32) -> u64 {
  let next_in: usize = next_in_u32 as usize;
  let mut eight_byte: [u8; 8] = fast_uninitialized![8];
  eight_byte.clone_from_slice(fast!((input)[next_in ; next_in + 8]));

  (eight_byte[0] as u64) | ((eight_byte[1] as u64) << 8) | ((eight_byte[2] as u64) << 16) |
      ((eight_byte[3] as u64) << 24) |
      ((eight_byte[4] as u64) << 32) | ((eight_byte[5] as u64) << 40) |
      ((eight_byte[6] as u64) << 48) | ((eight_byte[7] as u64) << 56)
}
pub const BROTLI_ALIGNED_READ: u8 = 0;

#[inline(always)]
pub fn BrotliFillBitWindow(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) {
  if ::core::mem::size_of::<reg_t>() == 8 {
    if (n_bits <= 8 && BROTLI_ALIGNED_READ == 0 && br.bit_pos_ >= 56) {
      br.val_ >>= 56;
      br.bit_pos_ ^= 56;  // here same as -= 56 because of the if condition
      br.val_ |= BrotliLoad64LE(input, br.next_in) << 8;
      br.avail_in -= 7;
      br.next_in += 7;
    } else if (BROTLI_ALIGNED_READ == 0 && n_bits <= 16 && br.bit_pos_ >= 48) {
      br.val_ >>= 48;
      br.bit_pos_ ^= 48;  // here same as -= 48 because of the if condition
      br.val_ |= BrotliLoad64LE(input, br.next_in) << 16;
      br.avail_in -= 6;
      br.next_in += 6;
    } else if br.bit_pos_ >= 32 {
      br.val_ >>= 32;
      br.bit_pos_ ^= 32;  /* here same as -= 32 because of the if condition */
      br.val_ |= (BrotliLoad32LE(input, br.next_in) as reg_t) << 32;
      br.avail_in -= BROTLI_SHORT_FILL_BIT_WINDOW_READ;
      br.next_in += BROTLI_SHORT_FILL_BIT_WINDOW_READ;
    }
  } else if (BROTLI_ALIGNED_READ == 0 && (n_bits <= 8)) {
    if (br.bit_pos_ >= 24) {
      br.val_ >>= 24;
      br.bit_pos_ ^= 24;  // here same as -= 24 because of the if condition
      br.val_ |= (BrotliLoad32LE(input, br.next_in) << 8) as u64;
      br.avail_in -= 3;
      br.next_in += 3;
    }
  } else if br.bit_pos_ >= 16 {
    br.val_ >>= 16;
    br.bit_pos_ ^= 16;  /* here same as -= 16 because of the if condition */
    br.val_ |= (BrotliLoad16LE(input, br.next_in) as reg_t) << 16;
    br.avail_in -= BROTLI_SHORT_FILL_BIT_WINDOW_READ;
    br.next_in += BROTLI_SHORT_FILL_BIT_WINDOW_READ;
  }
}

#[inline(always)]
fn BrotliFillBitWindowCompileTimeNbits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) {
  if ::core::mem::size_of::<reg_t>() == 8 {
    if BROTLI_ALIGNED_READ == 0 && n_bits <= 8 {
      // !BROTLI_ALIGNED_READ && IS_CONSTANT(n_bits) && (n_bits <= 8)) {
      if br.bit_pos_ >= 56 {
        br.val_ >>= 56;
        br.bit_pos_ ^= 56;  /* here same as -= 56 because of the if condition */
        br.val_ |= BrotliLoad64LE(input, br.next_in) << 8;
        br.avail_in -= 7;
        br.next_in += 7;
      }
    } else if BROTLI_ALIGNED_READ == 0 && n_bits <= 16 {
      // (!BROTLI_ALIGNED_READ && IS_CONSTANT(n_bits) && (n_bits <= 16)) {
      if br.bit_pos_ >= 48 {
        br.val_ >>= 48;
        br.bit_pos_ ^= 48;  /* here same as -= 48 because of the if condition */
        br.val_ |= BrotliLoad64LE(input, br.next_in) << 16;
        br.avail_in -= 6;
        br.next_in += 6;
      }
    } else if br.bit_pos_ >= 32 {
      br.val_ >>= 32;
      br.bit_pos_ ^= 32;  /* here same as -= 32 because of the if condition */
      br.val_ |= (BrotliLoad32LE(input, br.next_in) as reg_t) << 32;
      br.avail_in -= BROTLI_SHORT_FILL_BIT_WINDOW_READ;
      br.next_in += BROTLI_SHORT_FILL_BIT_WINDOW_READ;
    }
  } else if
      // BROTLI_ALIGNED_READ == false &&
      n_bits <= 8 {
    // !BROTLI_ALIGNED_READ && IS_CONSTANT(n_bits) && (n_bits <= 8)) {
    if br.bit_pos_ >= 24 {
      br.val_ >>= 24;
      br.bit_pos_ ^= 24;  /* here same as -= 24 because of the if condition */
      br.val_ |= (BrotliLoad32LE(input, br.next_in) as reg_t) << 8;
      br.avail_in -= 3;
      br.next_in += 3;
    }
  } else if br.bit_pos_ >= 16 {
      br.val_ >>= 16;
      br.bit_pos_ ^= 16;  /* here same as -= 16 because of the if condition */
      br.val_ |= (BrotliLoad16LE(input, br.next_in) as reg_t) << 16;
      br.avail_in -= BROTLI_SHORT_FILL_BIT_WINDOW_READ;
      br.next_in += BROTLI_SHORT_FILL_BIT_WINDOW_READ;
  }
}

// Mosltly like BrotliFillBitWindow, but guarantees only 16 bits and reads no
// more than BROTLI_SHORT_FILL_BIT_WINDOW_READ bytes of input.
pub fn BrotliFillBitWindow16(br: &mut BrotliBitReader, input: &[u8]) {
  BrotliFillBitWindowCompileTimeNbits(br, 17, input);
}

// Pulls one byte of input to accumulator.
pub fn BrotliPullByte(br: &mut BrotliBitReader, input: &[u8]) -> bool {
  if br.avail_in == 0 {
    return false;
  }
  br.val_ >>= 8;
  if ::core::mem::size_of::<reg_t>() == 8 {
    br.val_ |= (fast!((input)[br.next_in as usize]) as reg_t) << 56;
  } else {
    br.val_ |= (fast!((input)[br.next_in as usize]) as reg_t) << 24;
  }
  br.bit_pos_ -= 8;
  br.avail_in -= 1;
  br.next_in += 1;
  true
}

// Returns currently available bits.
// The number of valid bits could be calclulated by BrotliGetAvailableBits.
#[inline(always)]
pub fn BrotliGetBitsUnmasked(br: &BrotliBitReader) -> reg_t {
  br.val_ >> br.bit_pos_
}

// Like BrotliGetBits, but does not mask the result.
// The result contains at least 16 valid bits.
#[inline(always)]
pub fn BrotliGet16BitsUnmasked(br: &mut BrotliBitReader, input: &[u8]) -> u32 {
  BrotliFillBitWindowCompileTimeNbits(br, 16, input);
  (BrotliGetBitsUnmasked(br) & (0xffffffffu32 as reg_t)) as u32
}

// Returns the specified number of bits from br without advancing bit pos.
pub fn BrotliGetBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  BrotliFillBitWindow(br, n_bits, input);
  (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits)
}

// Returns the specified number of bits from br without advancing bit pos.
#[allow (dead_code)]
pub fn BrotliGetConstantNBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  BrotliFillBitWindowCompileTimeNbits(br, n_bits, input);
  (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits)
}

// Tries to peek the specified amount of bits. Returns 0, if there is not
// enough input.
pub fn BrotliSafeGetBits(br: &mut BrotliBitReader,
                         n_bits: u32,
                         val: &mut u32,
                         input: &[u8])
                         -> bool {
  while BrotliGetAvailableBits(br) < n_bits {
    if !BrotliPullByte(br, input) {
      return false;
    }
  }
  *val = (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits);
  true
}

// Advances the bit pos by n_bits.
#[inline(always)]
pub fn BrotliDropBits(br: &mut BrotliBitReader, n_bits: u32) {
  br.bit_pos_ += n_bits;
}

pub fn BrotliBitReaderUnload(br: &mut BrotliBitReader) {
  let unused_bytes: u32 = BrotliGetAvailableBits(br) >> 3;
  let unused_bits: u32 = unused_bytes << 3;
  br.avail_in += unused_bytes;
  br.next_in -= unused_bytes;
  if unused_bits as usize == (::core::mem::size_of::<reg_t>() << 3) {
    br.val_ = 0;
  } else {
    br.val_ <<= unused_bits;
  }
  br.bit_pos_ += unused_bits;
}

// Reads the specified number of bits from br and advances the bit pos.
// Precondition: accumulator MUST contain at least n_bits.
#[inline(always)]
pub fn BrotliTakeBits(br: &mut BrotliBitReader, n_bits: u32, val: &mut u32) {
  *val = (BrotliGetBitsUnmasked(br) as u32) & BitMask(n_bits);
  // if true {
  xprintln!("[BrotliReadBits]  {:?} {:?} {:?} val: {:x}\n",
           br.avail_in, br.bit_pos_, n_bits, *val);
  // }
  BrotliDropBits(br, n_bits);
}

// Reads the specified number of bits from br and advances the bit pos.
// Assumes that there is enough input to perform BrotliFillBitWindow.
#[inline(always)]
pub fn BrotliReadBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  if ::core::mem::size_of::<reg_t>() == 8 || (n_bits <= 16) {
    let mut val: u32 = 0;
    BrotliFillBitWindow(br, n_bits, input);
    BrotliTakeBits(br, n_bits, &mut val);
    val
  } else {
    let mut low_val: u32 = 0;
    let mut high_val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, 16, input);
    BrotliTakeBits(br, 16, &mut low_val);
    BrotliFillBitWindowCompileTimeNbits(br, 8, input);
    BrotliTakeBits(br, n_bits - 16, &mut high_val);
    low_val | (high_val << 16)
  }
}

// Reads the specified number of bits from br and advances the bit pos.
// Assumes that there is enough input to perform BrotliFillBitWindow.
#[allow (dead_code)]
pub fn BrotliReadConstantNBits(br: &mut BrotliBitReader, n_bits: u32, input: &[u8]) -> u32 {
  if ::core::mem::size_of::<reg_t>() == 8 || (n_bits <= 16) {
    let mut val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, n_bits, input);
    BrotliTakeBits(br, n_bits, &mut val);
    val
  } else {
    let mut low_val: u32 = 0;
    let mut high_val: u32 = 0;
    BrotliFillBitWindowCompileTimeNbits(br, 16, input);
    BrotliTakeBits(br, 16, &mut low_val);
    BrotliFillBitWindowCompileTimeNbits(br, 8, input);
    BrotliTakeBits(br, n_bits - 16, &mut high_val);
    low_val | (high_val << 16)
  }
}

// Tries to read the specified amount of bits. Returns 0, if there is not
// enough input. n_bits MUST be positive.
pub fn BrotliSafeReadBits(br: &mut BrotliBitReader,
                          n_bits: u32,
                          val: &mut u32,
                          input: &[u8])
                          -> bool {
  while BrotliGetAvailableBits(br) < n_bits {
    if !BrotliPullByte(br, input) {
      return false;
    }
  }
  BrotliTakeBits(br, n_bits, val);
  true
}

// Advances the bit reader position to the next byte boundary and verifies
// that any skipped bits are set to zero.
pub fn BrotliJumpToByteBoundary(br: &mut BrotliBitReader) -> bool {
  let pad_bits_count: u32 = BrotliGetAvailableBits(br) & 0x7;
  let mut pad_bits: u32 = 0;
  if pad_bits_count != 0 {
    BrotliTakeBits(br, pad_bits_count, &mut pad_bits);
  }
  pad_bits == 0
}

// Peeks a byte at specified offset.
// Precondition: bit reader is parked to a byte boundary.
// Returns -1 if operation is not feasible.
#[allow(dead_code)]
pub fn BrotliPeekByte(br: &mut BrotliBitReader, mut offset: u32, input: &[u8]) -> i32 {
  let available_bits: u32 = BrotliGetAvailableBits(br);
  let bytes_left: u32 = (available_bits >> 3);
  assert!((available_bits & 7) == 0);
  if offset < bytes_left {
    return ((BrotliGetBitsUnmasked(br) >> ((offset << 3)) as u32) & 0xFF) as i32;
  }
  offset -= bytes_left;
  if offset < br.avail_in {
    return fast!((input)[br.next_in as usize + offset as usize]) as i32;
  }
  -1
}

// Copies remaining input bytes stored in the bit reader to the output. Value
// num may not be larger than BrotliGetRemainingBytes. The bit reader must be
// warmed up again after this.
pub fn BrotliCopyBytes(dest: &mut [u8], br: &mut BrotliBitReader, mut num: u32, input: &[u8]) {
  let mut offset: u32 = 0;
  while BrotliGetAvailableBits(br) >= 8 && num > 0 {
    fast_mut!((dest)[offset as usize]) = BrotliGetBitsUnmasked(br) as u8;
    BrotliDropBits(br, 8);
    offset += 1;
    num -= 1;
  }
  for index in 0..num {
    fast_mut!((dest)[offset as usize + index as usize]) =
      fast!((input)[br.next_in as usize + index as usize]);
  }
  br.avail_in -= num;
  br.next_in += num;
}

pub fn BrotliInitBitReader(br: &mut BrotliBitReader) {
  br.val_ = 0;
  br.bit_pos_ = (::core::mem::size_of::<reg_t>() << 3) as u32;
}

pub fn BrotliWarmupBitReader(br: &mut BrotliBitReader, input: &[u8]) -> bool {
  let _aligned_read_mask: usize = (::core::mem::size_of::<reg_t>() >> 1) - 1;
  // Fixing alignment after unaligned BrotliFillWindow would result accumulator
  // overflow. If unalignment is caused by BrotliSafeReadBits, then there is
  // enough space in accumulator to fix aligment.
  if BrotliGetAvailableBits(br) == 0 && !BrotliPullByte(br, input) {
    return false;
  }
  // we can't tell about alignment yet
  // while (((size_t)br->next_in) & aligned_read_mask) != 0 {
  // if !BrotliPullByte(br) {
  // If we consumed all the input, we don't care about the alignment. */
  // return true;
  // }
  // }
  //
  true
}



#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn warmup_works() {
    let data: [u8; 37] = [0xde, 0xad, 0xbe, 0xef /* 4 bytes garbage at beginning */, 0x21,
                          0xfd, 0xff, 0x87, 0x03, 0x6c, 0x63, 0x90, 0xde, 0xe9, 0x36, 0x04, 0xf8,
                          0x89, 0xb4, 0x4d, 0x57, 0x96, 0x51, 0x0a, 0x54, 0xb7, 0x54, 0xee, 0xf4,
                          0xdc, 0xa0, 0x05, 0x9b, 0xd2, 0x2b, 0x30, 0x8f]; // 33 b goods
    let mut bit_reader = BrotliBitReader {
      val_: 0x0,
      bit_pos_: 64,
      avail_in: 33,
      next_in: 4,
    };
    let ret = BrotliWarmupBitReader(&mut bit_reader, &data[..]);
    assert_eq!(ret, true);
    assert_eq!(bit_reader.val_, 0x2100000000000000);
    assert_eq!(bit_reader.bit_pos_, 56);
    assert_eq!(bit_reader.avail_in, 32);
    assert_eq!(bit_reader.next_in, 5);
  }
  #[test]
  fn warmup_errcheck() {
    let data: [u8; 3] = [0xde, 0xad, 0xeb]; // 3 bytes garbage at beginning
    let mut bit_reader = BrotliBitReader {
      val_: 0x86e884e1ffff577b,
      bit_pos_: 64,
      avail_in: 0,
      next_in: 3,
    };
    let ret = BrotliWarmupBitReader(&mut bit_reader, &data[..]);
    assert_eq!(ret, false);
    assert_eq!(bit_reader.val_, 0x86e884e1ffff577b);
    assert_eq!(bit_reader.bit_pos_, 64);
    assert_eq!(bit_reader.avail_in, 0);
    assert_eq!(bit_reader.next_in, 3);
  }
  #[test]
  fn safe_read_tests() {
    {
      let data: [u8; 12] = [0x50, 0x3b, 0xbb, 0x5e, 0xc5, 0x96, 0x81, 0xb7, 0x52, 0x89, 0xea, 0x3d];
      let mut bit_reader = BrotliBitReader {
        val_: 0xe1e56a736e04fbf5,
        bit_pos_: 6,
        avail_in: 12,
        next_in: 0,
      };
      let mut val: u32 = 0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 7, &mut val, &data[..]);
      assert_eq!(ret, true);
      assert_eq!(val, 0x6f);
      assert_eq!(bit_reader.avail_in, 12);
      assert_eq!(bit_reader.next_in, 0);
      assert_eq!(bit_reader.bit_pos_, 13);
    }
    {
      let data: [u8; 7] = [0xba, 0xd0, 0xf0, 0x0d, 0xc8, 0xcd, 0xcc];
      let mut bit_reader = BrotliBitReader {
        val_: 0x17f115ae26916f0,
        bit_pos_: 57,
        avail_in: 3,
        next_in: 4,
      };
      let mut val: u32 = 0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 15, &mut val, &data[..]);
      assert_eq!(ret, true);
      assert_eq!(val, 0x6400);
      assert_eq!(bit_reader.avail_in, 2);
      assert_eq!(bit_reader.next_in, 5);
      assert_eq!(bit_reader.bit_pos_, 64);
    }
    {
      let data: [u8; 4] = [0xee, 0xee, 0xf0, 0xd5];
      let mut bit_reader = BrotliBitReader {
        val_: 0x5f43f252c027e447,
        bit_pos_: 53,
        avail_in: 2,
        next_in: 2,
      };
      let mut val: u32 = 0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 14, &mut val, &data[..]);
      assert_eq!(ret, true);
      assert_eq!(bit_reader.avail_in, 1);
      assert_eq!(bit_reader.next_in, 3);
      assert_eq!(bit_reader.bit_pos_, 59);
      assert_eq!(bit_reader.val_, 0xf05f43f252c027e4);
      assert_eq!(val, 0x2fa);
    }
    {
      let data: [u8; 4] = [0xee, 0xee, 0xf0, 0xd5];
      let mut bit_reader = BrotliBitReader {
        val_: 0x2f902339697460,
        bit_pos_: 57,
        avail_in: 0,
        next_in: 4,
      };
      let mut val: u32 = 0x74eca3f0;
      let ret = BrotliSafeReadBits(&mut bit_reader, 14, &mut val, &data[..]);
      assert_eq!(ret, false);
      assert_eq!(bit_reader.avail_in, 0);
      assert_eq!(bit_reader.next_in, 4);
      assert_eq!(bit_reader.bit_pos_, 57);
      assert_eq!(bit_reader.val_, 0x2f902339697460);
      assert_eq!(val, 0x74eca3f0);
    }
  }
  #[test]
  fn bit_read_tests() {
    {
      let data: [u8; 32] = [0xba, 0xaa, 0xad, 0xdd, 0x57, 0x5c, 0xd9, 0xa3, 0x3e, 0xb3, 0x77,
                            0xe7, 0xa0, 0x1e, 0x09, 0xd3, 0x12, 0xa1, 0x3f, 0xb8, 0x7e, 0x5a,
                            0x06, 0x86, 0xe5, 0x36, 0xef, 0x9c, 0x9f, 0x6d, 0x9b, 0xcc];
      let mut bit_reader = BrotliBitReader {
        val_: 0xf5917f07daaaeabb,
        bit_pos_: 33,
        avail_in: 29,
        next_in: 3,
      };
      let ret = BrotliReadBits(&mut bit_reader, 8, &data[..]);
      assert_eq!(ret, 0x83);
      assert_eq!(bit_reader.bit_pos_, 9);
      assert_eq!(bit_reader.avail_in, 25);
      assert_eq!(bit_reader.next_in, 7);
    }
    {
      let data: [u8; 28] = [0xba, 0xaa, 0xaa, 0xad, 0x74, 0x40, 0x8e, 0xee, 0xd2, 0x38, 0xf1,
                            0xf4, 0xf8, 0x1d, 0x9f, 0x24, 0x48, 0x1e, 0x82, 0xce, 0x48, 0x88,
                            0xd7, 0x25, 0x74, 0xaf, 0xe3, 0xea];
      let mut bit_reader = BrotliBitReader {
        val_: 0x27e33b2440d3feaf,
        bit_pos_: 18,
        avail_in: 24,
        next_in: 4,
      };
      let ret = BrotliReadBits(&mut bit_reader, 15, &data[..]);
      assert_eq!(ret, 0x1034);
      assert_eq!(bit_reader.bit_pos_, 33);
      assert_eq!(bit_reader.avail_in, 24);
      assert_eq!(bit_reader.next_in, 4);
    }
  }
  #[test]
  fn const_fill_bit_window_tests() {
    {
      let data: [u8; 36] = [0xff, 0x9a, 0xa0, 0xde, 0x50, 0x99, 0x67, 0x67, 0x69, 0x87, 0x0e,
                            0x69, 0xeb, 0x6a, 0xd1, 0x56, 0xc0, 0x32, 0x96, 0xed, 0x78, 0x0e,
                            0x19, 0xdd, 0x0b, 0xe8, 0xf8, 0x33, 0x9f, 0xe0, 0x69, 0x55, 0x59,
                            0x3f, 0x5d, 0xc8];
      let mut bit_reader = BrotliBitReader {
        val_: 0xb3fc441e0181dc4,
        bit_pos_: 59,
        avail_in: 33,
        next_in: 1,
      };
      let ret = BrotliReadConstantNBits(&mut bit_reader, 4, &data[..]);
      assert_eq!(ret, 0x1);
      assert_eq!(bit_reader.bit_pos_, 7);
      assert_eq!(bit_reader.avail_in, 26);
      assert_eq!(bit_reader.next_in, 8);
    }
    {
      let data: [u8; 67] = [0xf0, 0x0d, 0x7e, 0x18, 0x70, 0x1c, 0x18, 0x57, 0xbd, 0x73, 0x47,
                            0xc1, 0xb4, 0xf7, 0xe2, 0xbe, 0x17, 0x6e, 0x26, 0x01, 0xb2, 0xd5,
                            0x55, 0xd8, 0x68, 0x1b, 0xc2, 0x87, 0xb4, 0xb1, 0xd9, 0x42, 0xac,
                            0x0d, 0x67, 0xb1, 0x93, 0x54, 0x49, 0xa4, 0x69, 0xf8, 0x16, 0x0e,
                            0x61, 0xb3, 0xdb, 0x98, 0xbb, 0xeb, 0xfa, 0xcb, 0x14, 0xcd, 0x68,
                            0x77, 0xa1, 0x33, 0x6c, 0x49, 0xfa, 0x35, 0xbb, 0xeb, 0xee, 0x7b, 0xae];
      let mut bit_reader = BrotliBitReader {
        val_: 0x655b1fe0dd6f1e78,
        bit_pos_: 63,
        avail_in: 65,
        next_in: 2,
      };
      let ret = BrotliGet16BitsUnmasked(&mut bit_reader, &data[..]);
      assert_eq!(ret, 0x38e030fc);
      assert_eq!(bit_reader.bit_pos_, 15);
      assert_eq!(bit_reader.avail_in, 59);
      assert_eq!(bit_reader.next_in, 8);
    }
  }
}
