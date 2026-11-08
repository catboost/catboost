// This is mostly taken from json-rust's codegen
// as it seems to perform well and it makes sense to see
// if we can adopt the approach
//
// https://github.com/maciejhirsz/json-rust/blob/master/src/codegen.rs

macro_rules! stry {
    ($e:expr) => {
        match $e {
            ::std::result::Result::Ok(val) => val,
            ::std::result::Result::Err(err) => return ::std::result::Result::Err(err),
        }
    };
}

use std::io;
use std::io::Write;
use std::ptr;

const QU: u8 = b'"';
const BS: u8 = b'\\';
const BB: u8 = b'b';
const TT: u8 = b't';
const NN: u8 = b'n';
const FF: u8 = b'f';
const RR: u8 = b'r';
const UU: u8 = b'u';
const __: u8 = 0;

// Look up table for characters that need escaping in a product string
pub(crate) static ESCAPED: [u8; 256] = [
    // 0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    UU, UU, UU, UU, UU, UU, UU, UU, BB, TT, NN, UU, FF, RR, UU, UU, // 0
    UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, UU, // 1
    __, __, QU, __, __, __, __, __, __, __, __, __, __, __, __, __, // 2
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 3
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 4
    __, __, __, __, __, __, __, __, __, __, __, __, BS, __, __, __, // 5
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 6
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 7
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 8
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // 9
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // A
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // B
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // C
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // D
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // E
    __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, __, // F
];

// taken from https://github.com/serde-rs/json/blob/4354fc3eb2232ee0ba9a9a23acce107a980a6dc0/src/ser.rs#L1790
// This is called rarely
#[inline(never)]
fn u_encode<W>(w: &mut W, byte: u8) -> io::Result<()>
where
    W: Write,
{
    static HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";
    let bytes = [
        b'\\',
        b'u',
        b'0',
        b'0',
        HEX_DIGITS[(byte >> 4) as usize],
        HEX_DIGITS[(byte & 0xF) as usize],
    ];
    w.write_all(&bytes)
}

/// Base generator trait
pub trait BaseGenerator {
    /// The writer
    type T: Write;

    /// returns the writer
    fn get_writer(&mut self) -> &mut Self::T;

    /// Write a slice
    /// # Errors
    /// if the write fails
    #[inline]
    fn write(&mut self, slice: &[u8]) -> io::Result<()> {
        self.get_writer().write_all(slice)
    }

    /// Write a char
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_char(&mut self, ch: u8) -> io::Result<()> {
        self.get_writer().write_all(&[ch])
    }

    /// write with minimum
    /// # Errors
    /// if the write fails
    fn write_min(&mut self, slice: &[u8], min: u8) -> io::Result<()>;

    /// writes new line
    /// # Errors
    /// if the write fails
    #[inline]
    fn new_line(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// indents one step
    #[inline]
    fn indent(&mut self) {}

    /// dedents one step
    #[inline]
    fn dedent(&mut self) {}

    /// writes a string
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_string(&mut self, string: &str) -> io::Result<()> {
        stry!(self.write_char(b'"'));
        stry!(self.write_string_content(string));
        self.write_char(b'"')
    }

    /// writes a string
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_string_content(&mut self, string: &str) -> io::Result<()> {
        let mut string = string.as_bytes();
        unsafe {
            // Looking at the table above the lower 5 bits are entirely
            // quote characters that gives us a bitmask of 0x1f for that
            // region, only quote (`"`) and backslash (`\`) are not in
            // this range.
            stry!(self.write_str_simd(&mut string));
        }
        write_string_rust(self.get_writer(), &mut string)
    }

    /// writes a simple string (usually short and non escaped)
    /// This means we can skip the simd accelerated writing which is
    /// expensive on short strings.
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_simple_string(&mut self, string: &str) -> io::Result<()> {
        self.write(br#"""#)?;
        write_string_rust(self.get_writer(), &mut string.as_bytes())?;
        self.write(br#"""#)
    }
    /// writes a simple string content  (usually short and non escaped)
    /// This means we can skip the simd accelerated writing which is
    /// expensive on short strings.
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_simple_str_content(&mut self, string: &str) -> io::Result<()> {
        let mut string = string.as_bytes();
        // Legacy code to handle the remainder of the code
        write_string_rust(self.get_writer(), &mut string)
    }

    /// writes a float value
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_float(&mut self, num: f64) -> io::Result<()> {
        let mut buffer = ryu::Buffer::new();
        let s = buffer.format_finite(num);
        self.get_writer().write_all(s.as_bytes())
    }

    /// writes an integer value
    /// # Errors
    /// if the write fails
    #[inline]
    fn write_int<I: itoa::Integer>(&mut self, num: I) -> io::Result<()> {
        let mut buffer = itoa::Buffer::new();
        let s = buffer.format(num);
        self.get_writer().write_all(s.as_bytes())
    }

    /// # Safety
    /// This function is unsafe because it may use simd instructions
    /// # Errors
    ///  if the write fails
    #[cfg(all(
        feature = "runtime-detection",
        any(target_arch = "x86_64", target_arch = "x86"),
    ))]
    unsafe fn write_str_simd(&mut self, string: &mut &[u8]) -> io::Result<()> {
        write_str_simd_fastest(self.get_writer(), string)
    }
    #[cfg(all(target_feature = "avx2", not(feature = "runtime-detection")))]
    #[inline]
    /// Writes a string with simd-acceleration
    /// # Safety
    /// This function is unsafe because it uses simd instructions
    /// # Errors
    ///  if the write fails
    unsafe fn write_str_simd(&mut self, string: &mut &[u8]) -> io::Result<()> {
        write_str_simd_avx2(self.get_writer(), string)
    }

    #[cfg(all(
        target_feature = "sse2",
        not(target_feature = "avx2"),
        not(feature = "runtime-detection")
    ))]
    #[inline]
    /// Writes a string with simd-acceleration
    /// # Safety
    /// This function is unsafe because it uses simd instructions
    /// # Errors
    ///  if the write fails
    unsafe fn write_str_simd(&mut self, string: &mut &[u8]) -> io::Result<()> {
        write_str_simd_sse42(self.get_writer(), string)
    }

    #[cfg(not(any(
        all(
            feature = "runtime-detection",
            any(target_arch = "x86_64", target_arch = "x86")
        ),
        feature = "portable",
        target_feature = "avx2",
        target_feature = "sse4.2",
        target_feature = "simd128",
        target_arch = "aarch64",
    )))]
    #[inline]
    /// Writes a string with simd-acceleration (not really, as the architecture doesn't support it)
    /// # Safety
    /// This function is unsafe because it uses simd instructions
    /// # Errors
    ///  if the write fails
    unsafe fn write_str_simd(&mut self, string: &mut &[u8]) -> io::Result<()> {
        write_string_rust(self.get_writer(), string)?;
        *string = &string[string.len()..];
        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    /// Writes a string with simd-acceleration
    /// # Safety
    /// This function is unsafe because it uses simd instructions
    /// # Errors
    ///  if the write fails
    unsafe fn write_str_simd(&mut self, string: &mut &[u8]) -> io::Result<()> {
        use std::arch::aarch64::{
            uint8x16_t, vandq_u8, vceqq_u8, vdupq_n_u8, veorq_u8, vgetq_lane_u16, vld1q_u8,
            vorrq_u8, vpaddq_u8, vreinterpretq_u16_u8,
        };
        use std::mem;

        #[inline]
        unsafe fn bit_mask() -> uint8x16_t {
            unsafe {
                mem::transmute([
                    0x01_u8, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x4, 0x8, 0x10,
                    0x20, 0x40, 0x80,
                ])
            }
        }

        #[inline]
        unsafe fn neon_movemask(input: uint8x16_t) -> u16 {
            unsafe {
                let simd_input: uint8x16_t = vandq_u8(input, bit_mask());
                let tmp: uint8x16_t = vpaddq_u8(simd_input, simd_input);
                let tmp = vpaddq_u8(tmp, tmp);
                let tmp = vpaddq_u8(tmp, tmp);

                vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0)
            }
        }

        let writer = self.get_writer();
        // The case where we have a 16+ byte block
        // we repeate the same logic as above but with
        // only 16 bytes
        let mut idx = 0;
        unsafe {
            let zero = vdupq_n_u8(0);
            let lower_quote_range = vdupq_n_u8(0x1F);
            let quote = vdupq_n_u8(b'"');
            let backslash = vdupq_n_u8(b'\\');
            while string.len() - idx > 16 {
                // Load 16 bytes of data;
                let data: uint8x16_t = vld1q_u8(string.as_ptr().add(idx));
                // Test the data against being backslash and quote.
                let bs_or_quote = vorrq_u8(vceqq_u8(data, backslash), vceqq_u8(data, quote));
                // Now mask the data with the quote range (0x1F).
                let in_quote_range = vandq_u8(data, lower_quote_range);
                // then test of the data is unchanged. aka: xor it with the
                // Any field that was inside the quote range it will be zero
                // now.
                let is_unchanged = veorq_u8(data, in_quote_range);
                let in_range = vceqq_u8(is_unchanged, zero);
                let quote_bits = neon_movemask(vorrq_u8(bs_or_quote, in_range));
                if quote_bits == 0 {
                    idx += 16;
                } else {
                    let quote_dist = quote_bits.trailing_zeros() as usize;
                    stry!(writer.write_all(&string[0..idx + quote_dist]));
                    let ch = string[idx + quote_dist];
                    match ESCAPED[ch as usize] {
                        b'u' => stry!(u_encode(writer, ch)),
                        escape => stry!(writer.write_all(&[b'\\', escape])),
                    }

                    *string = &string[idx + quote_dist + 1..];
                    idx = 0;
                }
            }
        }
        stry!(writer.write_all(&string[0..idx]));
        *string = &string[idx..];
        Ok(())
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline]
    /// Writes a string with simd-acceleration
    /// # Safety
    /// This function is unsafe because it uses simd instructions
    /// # Errors
    ///  if the write fails
    unsafe fn write_str_simd(&mut self, string: &mut &[u8]) -> io::Result<()> {
        let writer = self.get_writer();
        use std::arch::wasm32::{
            u8x16_bitmask, u8x16_eq, u8x16_splat, v128, v128_and, v128_load, v128_or, v128_xor,
        };

        // The case where we have a 16+ byte block
        // we repeat the same logic as above but with
        // only 16 bytes
        let mut idx = 0;
        let zero = u8x16_splat(0);
        let lower_quote_range = u8x16_splat(0x1F);
        let quote = u8x16_splat(b'"');
        let backslash = u8x16_splat(b'\\');
        while string.len() - idx > 16 {
            // Load 16 bytes of data;
            let data = v128_load(string.as_ptr().add(idx).cast::<v128>());
            // Test the data against being backslash and quote.
            let bs_or_quote = v128_or(u8x16_eq(data, backslash), u8x16_eq(data, quote));
            // Now mask the data with the quote range (0x1F).
            let in_quote_range = v128_and(data, lower_quote_range);
            // then test of the data is unchanged. aka: xor it with the
            // Any field that was inside the quote range it will be zero
            // now.
            let is_unchanged = v128_xor(data, in_quote_range);
            let in_range = u8x16_eq(is_unchanged, zero);
            let quote_bits = u8x16_bitmask(v128_or(bs_or_quote, in_range));
            if quote_bits == 0 {
                idx += 16;
            } else {
                let quote_dist = quote_bits.trailing_zeros() as usize;
                stry!(writer.write_all(&string[0..idx + quote_dist]));
                let ch = string[idx + quote_dist];
                match ESCAPED[ch as usize] {
                    b'u' => stry!(u_encode(writer, ch)),
                    escape => stry!(writer.write_all(&[b'\\', escape])),
                }

                *string = &string[idx + quote_dist + 1..];
                idx = 0;
            }
        }
        stry!(writer.write_all(&string[0..idx]));
        *string = &string[idx..];
        Ok(())
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
unsafe fn write_str_simd_fastest<W>(writer: &mut W, string: &mut &[u8]) -> io::Result<()>
where
    W: Write,
{
    // This is not possible right now since we can't get `W` to be part of the static
    // use std::sync::atomic::{AtomicPtr, Ordering};
    // type FnRaw = *mut ();

    // type WriteStrFn<T> = for<'a, 'b, 'c> unsafe fn(&'a mut T, &'b mut &'c [u8]) -> io::Result<()>;
    // static FN: AtomicPtr<()> = AtomicPtr::new(get_fastest::<W> as FnRaw);
    // #[inline]
    // fn get_fastest_available_implementation<W>() -> WriteStrFn<W>
    // where
    //     W: Write,
    // {
    //     if std::is_x86_feature_detected!("avx2") {
    //         write_str_simd_avx2
    //     } else if std::is_x86_feature_detected!("sse4.2") {
    //         write_str_simd_sse42
    //     } else {
    //         write_str_simd_rust
    //     }
    // }

    // #[inline]
    // unsafe fn get_fastest<'invoke, 'de, W>(writer: &mut W, string: &mut &[u8]) -> io::Result<()>
    // where
    //     W: Write,
    //     'de: 'invoke,
    // {
    //     let fun = get_fastest_available_implementation();
    //     FN.store(fun as FnRaw, Ordering::Relaxed);
    //     (fun)(writer, string)
    // }
    // let fun = FN.load(Ordering::Relaxed);
    // mem::transmute::<FnRaw, WriteStrFn>(fun)(writer, string)

    if std::is_x86_feature_detected!("avx2") {
        write_str_simd_avx2(writer, string)
    } else if std::is_x86_feature_detected!("sse4.2") {
        write_str_simd_sse42(writer, string)
    } else {
        #[cfg(not(feature = "portable"))]
        return write_string_rust(writer, string);
        #[cfg(feature = "portable")]
        return write_str_simd_portable(writer, string);
    }
}
#[inline]
fn write_string_container<W>(writer: &mut W, string: &[u8], mut start: usize) -> io::Result<()>
where
    W: Write,
{
    stry!(writer.write_all(&string[..start]));

    for (index, ch) in string.iter().enumerate().skip(start) {
        let escape = ESCAPED[*ch as usize];
        if escape > 0 {
            stry!(writer.write_all(&string[start..index]));
            if escape == b'u' {
                stry!(u_encode(writer, *ch));
            } else {
                stry!(writer.write_all(&[b'\\', escape]));
            };
            start = index + 1;
        }
    }
    writer.write_all(&string[start..])
}

#[inline]
fn write_string_rust<W>(writer: &mut W, string: &mut &[u8]) -> io::Result<()>
where
    W: Write,
{
    // Legacy code to handle the remainder of the code
    for (index, ch) in string.iter().enumerate() {
        if ESCAPED[*ch as usize] > 0 {
            return write_string_container(writer, string, index);
        }
    }
    writer.write_all(string)
}

#[cfg(feature = "portable")]
#[inline]
/// Writes a string with simd-acceleration
/// # Safety
/// This function is unsafe because it uses simd instructions
/// # Errors
///  if the write fails
unsafe fn write_str_simd_portable<W>(writer: &mut W, string: &mut &[u8]) -> io::Result<()>
where
    W: Write,
{
    use std::simd::{SimdPartialEq, ToBitMask, u8x32};

    let mut idx = 0;
    let zero = u8x32::splat(0);
    let lower_quote_range = u8x32::splat(0x1F_u8);
    let quote = u8x32::splat(b'"');
    let backslash = u8x32::splat(b'\\');
    while string.len() - idx >= 32 {
        // Load 32 bytes of data;
        let data = u8x32::from_slice(&string[idx..]);
        // Test the data against being backslash and quote.
        let bs_or_quote = data.simd_eq(backslash) | data.simd_eq(quote);
        // Now mask the data with the quote range (0x1F).
        let in_quote_range = data & lower_quote_range;
        // then test of the data is unchanged. aka: xor it with the
        // Any field that was inside the quote range it will be zero
        // now.
        let is_unchanged = data ^ in_quote_range;
        let in_range = is_unchanged.simd_eq(zero);
        let quote_bits = (bs_or_quote | in_range).to_bitmask();
        if quote_bits == 0 {
            idx += 32;
        } else {
            let quote_dist = quote_bits.trailing_zeros() as usize;
            stry!(writer.write_all(string.get_unchecked(0..idx + quote_dist)));

            let ch = string[idx + quote_dist];
            match ESCAPED[ch as usize] {
                b'u' => stry!(u_encode(writer, ch)),
                escape => stry!(writer.write_all(&[b'\\', escape])),
            };

            *string = string.get_unchecked(idx + quote_dist + 1..);
            idx = 0;
        }
    }
    stry!(writer.write_all(&string[0..idx]));
    *string = string.get_unchecked(idx..);
    Ok(())
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
#[inline]
/// Writes a string with simd-acceleration
/// # Safety
/// This function is unsafe because it uses simd instructions
/// # Errors
///  if the write fails
unsafe fn write_str_simd_avx2<W>(writer: &mut W, string: &mut &[u8]) -> io::Result<()>
where
    W: Write,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256i, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_or_si256, _mm256_set1_epi8, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_or_si256, _mm256_set1_epi8, _mm256_xor_si256,
    };

    let mut idx = 0;
    let zero = _mm256_set1_epi8(0);
    let lower_quote_range = _mm256_set1_epi8(0x1F_i8);
    #[allow(clippy::cast_possible_wrap)] // it's a const, it's fine
    let quote = _mm256_set1_epi8(b'"' as i8);
    #[allow(clippy::cast_possible_wrap)] // it's a const, it's fine
    let backslash = _mm256_set1_epi8(b'\\' as i8);
    while string.len() - idx >= 32 {
        // Load 32 bytes of data; _mm256_loadu_si256 does not require alignment
        #[allow(clippy::cast_ptr_alignment)]
        let data: __m256i = _mm256_loadu_si256(string.as_ptr().add(idx).cast::<__m256i>());
        // Test the data against being backslash and quote.
        let bs_or_quote = _mm256_or_si256(
            _mm256_cmpeq_epi8(data, backslash),
            _mm256_cmpeq_epi8(data, quote),
        );
        // Now mask the data with the quote range (0x1F).
        let in_quote_range = _mm256_and_si256(data, lower_quote_range);
        // then test of the data is unchanged. aka: xor it with the
        // Any field that was inside the quote range it will be zero
        // now.
        let is_unchanged = _mm256_xor_si256(data, in_quote_range);
        let in_range = _mm256_cmpeq_epi8(is_unchanged, zero);
        let quote_bits = _mm256_movemask_epi8(_mm256_or_si256(bs_or_quote, in_range));
        if quote_bits == 0 {
            idx += 32;
        } else {
            let quote_dist = quote_bits.trailing_zeros() as usize;
            stry!(writer.write_all(string.get_unchecked(0..idx + quote_dist)));

            let ch = string[idx + quote_dist];
            match ESCAPED[ch as usize] {
                b'u' => stry!(u_encode(writer, ch)),
                escape => stry!(writer.write_all(&[b'\\', escape])),
            };

            *string = string.get_unchecked(idx + quote_dist + 1..);
            idx = 0;
        }
    }
    stry!(writer.write_all(&string[0..idx]));
    *string = string.get_unchecked(idx..);
    Ok(())
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse4.2")]
#[inline]
/// Writes a string with simd-acceleration
/// # Safety
/// This function is unsafe because it uses simd instructions
/// # Errors
///  if the write fails
unsafe fn write_str_simd_sse42<W>(writer: &mut W, string: &mut &[u8]) -> io::Result<()>
where
    W: Write,
{
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128,
        _mm_set1_epi8, _mm_xor_si128,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128,
        _mm_set1_epi8, _mm_xor_si128,
    };

    let mut idx = 0;
    let zero = _mm_set1_epi8(0);
    let lower_quote_range = _mm_set1_epi8(0x1F_i8);
    #[allow(clippy::cast_possible_wrap)] // it's a const, it's fine
    let quote = _mm_set1_epi8(b'"' as i8);
    #[allow(clippy::cast_possible_wrap)] // it's a const, it's fine
    let backslash = _mm_set1_epi8(b'\\' as i8);
    while string.len() - idx > 16 {
        // Load 16 bytes of data; _mm_loadu_si128 does not require alignment
        #[allow(clippy::cast_ptr_alignment)]
        let data: __m128i = _mm_loadu_si128(string.as_ptr().add(idx).cast::<__m128i>());
        // Test the data against being backslash and quote.
        let bs_or_quote =
            _mm_or_si128(_mm_cmpeq_epi8(data, backslash), _mm_cmpeq_epi8(data, quote));
        // Now mask the data with the quote range (0x1F).
        let in_quote_range = _mm_and_si128(data, lower_quote_range);
        // then test of the data is unchanged. aka: xor it with the
        // Any field that was inside the quote range it will be zero
        // now.
        let is_unchanged = _mm_xor_si128(data, in_quote_range);
        let in_range = _mm_cmpeq_epi8(is_unchanged, zero);
        let quote_bits = _mm_movemask_epi8(_mm_or_si128(bs_or_quote, in_range));
        if quote_bits == 0 {
            idx += 16;
        } else {
            let quote_dist = quote_bits.trailing_zeros() as usize;
            stry!(writer.write_all(&string[0..idx + quote_dist]));

            let ch = string[idx + quote_dist];
            match ESCAPED[ch as usize] {
                b'u' => stry!(u_encode(writer, ch)),
                escape => stry!(writer.write_all(&[b'\\', escape])),
            }

            *string = &string[idx + quote_dist + 1..];
            idx = 0;
        }
    }
    stry!(writer.write_all(&string[0..idx]));
    *string = &string[idx..];
    Ok(())
}

///  Simple dump Generator
pub struct DumpGenerator {
    code: Vec<u8>,
}

impl Default for DumpGenerator {
    fn default() -> Self {
        Self {
            code: Vec::with_capacity(1024),
        }
    }
}

impl DumpGenerator {
    /// Creates a new generator
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the data as a String
    #[must_use]
    pub fn consume(self) -> String {
        // Original strings were unicode, numbers are all ASCII,
        // therefore this is safe.
        unsafe { String::from_utf8_unchecked(self.code) }
    }
}

impl BaseGenerator for DumpGenerator {
    type T = Vec<u8>;

    #[inline]
    fn write(&mut self, slice: &[u8]) -> io::Result<()> {
        extend_from_slice(&mut self.code, slice);
        Ok(())
    }
    #[inline]
    fn write_char(&mut self, ch: u8) -> io::Result<()> {
        self.code.push(ch);
        Ok(())
    }

    #[inline]
    fn get_writer(&mut self) -> &mut Vec<u8> {
        &mut self.code
    }

    #[inline]
    fn write_min(&mut self, _: &[u8], min: u8) -> io::Result<()> {
        self.code.push(min);
        Ok(())
    }
}

/// Pretty Generator
pub struct PrettyGenerator {
    code: Vec<u8>,
    dent: u16,
    spaces_per_indent: u16,
}

impl PrettyGenerator {
    /// Creates a new pretty printing generator
    #[must_use]
    pub fn new(spaces: u16) -> Self {
        Self {
            code: Vec::with_capacity(1024),
            dent: 0,
            spaces_per_indent: spaces,
        }
    }

    /// Returns the data as a String
    #[must_use]
    pub fn consume(self) -> String {
        unsafe { String::from_utf8_unchecked(self.code) }
    }
}

impl BaseGenerator for PrettyGenerator {
    type T = Vec<u8>;
    #[inline]
    fn write(&mut self, slice: &[u8]) -> io::Result<()> {
        extend_from_slice(&mut self.code, slice);
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, ch: u8) -> io::Result<()> {
        self.code.push(ch);
        Ok(())
    }

    #[inline]
    fn get_writer(&mut self) -> &mut Vec<u8> {
        &mut self.code
    }

    #[inline]
    fn write_min(&mut self, slice: &[u8], _: u8) -> io::Result<()> {
        extend_from_slice(&mut self.code, slice);
        Ok(())
    }

    fn new_line(&mut self) -> io::Result<()> {
        self.code.push(b'\n');
        self.code.resize(
            self.code.len() + (self.dent * self.spaces_per_indent) as usize,
            b' ',
        );
        Ok(())
    }

    fn indent(&mut self) {
        self.dent += 1;
    }

    fn dedent(&mut self) {
        self.dent -= 1;
    }
}

/// Writer Generator
pub struct WriterGenerator<'w, W: 'w + Write> {
    writer: &'w mut W,
}

impl<'w, W> WriterGenerator<'w, W>
where
    W: 'w + Write,
{
    /// Creates a new generator
    pub fn new(writer: &'w mut W) -> Self {
        WriterGenerator { writer }
    }
}

impl<W> BaseGenerator for WriterGenerator<'_, W>
where
    W: Write,
{
    type T = W;

    #[inline]
    fn get_writer(&mut self) -> &mut W {
        self.writer
    }

    #[inline]
    fn write_min(&mut self, _: &[u8], min: u8) -> io::Result<()> {
        self.writer.write_all(&[min])
    }
}

/// Pretty Writer Generator
pub struct PrettyWriterGenerator<'w, W>
where
    W: 'w + Write,
{
    writer: &'w mut W,
    dent: u16,
    spaces_per_indent: u16,
}

impl<'w, W> PrettyWriterGenerator<'w, W>
where
    W: 'w + Write,
{
    /// Creates a new generator
    pub fn new(writer: &'w mut W, spaces_per_indent: u16) -> Self {
        PrettyWriterGenerator {
            writer,
            dent: 0,
            spaces_per_indent,
        }
    }
}

impl<W> BaseGenerator for PrettyWriterGenerator<'_, W>
where
    W: Write,
{
    type T = W;

    #[inline]
    fn get_writer(&mut self) -> &mut W {
        self.writer
    }

    #[inline]
    fn write_min(&mut self, slice: &[u8], _: u8) -> io::Result<()> {
        self.writer.write_all(slice)
    }

    fn new_line(&mut self) -> io::Result<()> {
        stry!(self.write_char(b'\n'));
        for _ in 0..(self.dent * self.spaces_per_indent) {
            stry!(self.write_char(b' '));
        }
        Ok(())
    }

    fn indent(&mut self) {
        self.dent += 1;
    }

    fn dedent(&mut self) {
        self.dent -= 1;
    }
}

// From: https://github.com/dtolnay/fastwrite/blob/master/src/lib.rs#L68
//
// LLVM is not able to lower `Vec::extend_from_slice` into a memcpy, so this
// helps eke out that last bit of performance.
#[inline]
pub(crate) fn extend_from_slice(dst: &mut Vec<u8>, src: &[u8]) {
    let dst_len = dst.len();
    let src_len = src.len();

    dst.reserve(src_len);

    unsafe {
        // We would have failed if `reserve` overflowed\
        ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr().add(dst_len), src_len);
        dst.set_len(dst_len + src_len);
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_write_string_rust() {
        let mut writer = Vec::new();
        let mut string = "Hello, World!".as_bytes();
        super::write_string_rust(&mut writer, &mut string).expect("failed to write string");
        assert_eq!(writer, "Hello, World!".as_bytes());
    }
    #[test]
    fn test_write_string_rust2() {
        let mut writer = Vec::new();
        let mut string = "Hello, \"World!\"".as_bytes();
        super::write_string_rust(&mut writer, &mut string).expect("failed to write string");
        assert_eq!(writer, "Hello, \\\"World!\\\"".as_bytes());
    }
}
