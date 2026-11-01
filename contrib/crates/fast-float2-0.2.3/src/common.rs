use core::marker::PhantomData;
use core::ptr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsciiStr<'a> {
    ptr: *const u8,
    end: *const u8,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a> AsciiStr<'a> {
    #[inline]
    pub fn new(s: &'a [u8]) -> Self {
        Self {
            ptr: s.as_ptr(),
            end: unsafe { s.as_ptr().add(s.len()) },
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> isize {
        self.end as isize - self.ptr as isize
    }

    /// # Safety
    ///
    /// Safe if `n <= self.len()`
    #[inline]
    pub unsafe fn step_by(&mut self, n: usize) -> &mut Self {
        debug_assert!(
            // FIXME: remove when we drop support for < 1.43.0
            n < isize::max_value() as usize && n as isize <= self.len(),
            "buffer overflow: stepping by greater than our buffer length."
        );
        // SAFETY: Safe if `n <= self.len()`
        unsafe { self.ptr = self.ptr.add(n) };
        self
    }

    /// # Safety
    ///
    /// Safe if `!self.is_empty()`
    #[inline]
    pub unsafe fn step(&mut self) -> &mut Self {
        debug_assert!(!self.is_empty(), "buffer overflow: buffer is empty.");
        // SAFETY: Safe if the buffer is not empty, that is, `self.len() >= 1`
        unsafe { self.step_by(1) }
    }

    #[inline]
    pub fn step_if(&mut self, c: u8) -> bool {
        let stepped = self.first_is(c);
        if stepped {
            // SAFETY: safe since we have at least 1 character in the buffer
            unsafe { self.step() };
        }
        stepped
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ptr == self.end
    }

    /// # Safety
    ///
    /// Safe if `!self.is_empty()`
    #[inline]
    pub unsafe fn first_unchecked(&self) -> u8 {
        debug_assert!(!self.is_empty(), "attempting to get first value of empty buffer.");
        unsafe { *self.ptr }
    }

    #[inline]
    pub fn first(&self) -> Option<u8> {
        if self.is_empty() {
            None
        } else {
            // SAFETY: safe since `!self.is_empty()`
            Some(unsafe { self.first_unchecked() })
        }
    }

    #[inline]
    pub fn first_is(&self, c: u8) -> bool {
        self.first() == Some(c)
    }

    #[inline]
    pub fn first_is2(&self, c1: u8, c2: u8) -> bool {
        self.first().map_or(false, |c| c == c1 || c == c2)
    }

    #[inline]
    pub fn first_is_digit(&self) -> bool {
        self.first().map_or(false, |c| c.is_ascii_digit())
    }

    #[inline]
    pub fn first_digit(&self) -> Option<u8> {
        self.first().and_then(|x| {
            if x.is_ascii_digit() {
                Some(x - b'0')
            } else {
                None
            }
        })
    }

    #[inline]
    pub fn try_read_digit(&mut self) -> Option<u8> {
        let digit = self.first_digit()?;
        // SAFETY: Safe since `first_digit` means the buffer is not empty
        unsafe { self.step() };
        Some(digit)
    }

    #[inline]
    pub fn parse_digits(&mut self, mut func: impl FnMut(u8)) {
        while let Some(digit) = self.try_read_digit() {
            func(digit);
        }
    }

    #[inline]
    pub fn try_read_u64(&self) -> Option<u64> {
        if self.len() >= 8 {
            Some(unsafe { self.read_u64_unchecked() })
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// Safe if `self.len() >= 8`
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn read_u64_unchecked(&self) -> u64 {
        debug_assert!(self.len() >= 8, "overflowing buffer: buffer is not 8 bytes long");
        let src = self.ptr as *const u64;
        // SAFETY: Safe if `self.len() >= 8`
        u64::from_le(unsafe { ptr::read_unaligned(src) })
    }

    #[inline]
    pub fn offset_from(&self, other: &Self) -> isize {
        isize::wrapping_sub(self.ptr as isize, other.ptr as isize) // assuming the same end
    }
}

// Most of these are inherently unsafe; we assume we know what we're calling and
// when.
pub trait ByteSlice: AsRef<[u8]> + AsMut<[u8]> {
    #[inline]
    fn check_first(&self, c: u8) -> bool {
        self.as_ref().first() == Some(&c)
    }

    #[inline]
    fn check_first2(&self, c1: u8, c2: u8) -> bool {
        if let Some(&c) = self.as_ref().first() {
            c == c1 || c == c2
        } else {
            false
        }
    }

    #[inline]
    fn eq_ignore_case(&self, u: &[u8]) -> bool {
        let s = self.as_ref();
        if s.len() < u.len() {
            return false;
        }
        let d = (0..u.len()).fold(0, |d, i| d | s[i] ^ u[i]);
        d == 0 || d == 32
    }

    #[inline]
    fn advance(&self, n: usize) -> &[u8] {
        &self.as_ref()[n..]
    }

    #[inline]
    fn skip_chars(&self, c: u8) -> &[u8] {
        let mut s = self.as_ref();
        while s.check_first(c) {
            s = s.advance(1);
        }
        s
    }

    /// # Safety
    ///
    /// Safe if `self.len() >= 8`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn read_u64(&self) -> u64 {
        debug_assert!(self.as_ref().len() >= 8);
        let src = self.as_ref().as_ptr() as *const u64;
        // SAFETY: safe if `self.len() >= 8`.
        u64::from_le(unsafe { ptr::read_unaligned(src) })
    }

    /// # Safety
    ///
    /// Safe if `self.len() >= 8`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    unsafe fn write_u64(&mut self, value: u64) {
        debug_assert!(self.as_ref().len() >= 8);
        let dst = self.as_mut().as_mut_ptr() as *mut u64;
        // SAFETY: safe if `self.len() >= 8`.
        unsafe { ptr::write_unaligned(dst, u64::to_le(value)) };
    }
}

impl ByteSlice for [u8] {
}

#[inline]
pub fn is_8digits(v: u64) -> bool {
    let a = v.wrapping_add(0x4646_4646_4646_4646);
    let b = v.wrapping_sub(0x3030_3030_3030_3030);
    (a | b) & 0x8080_8080_8080_8080 == 0
}

#[inline]
pub fn parse_digits(s: &mut &[u8], mut f: impl FnMut(u8)) {
    while let Some(&ch) = s.first() {
        let c = ch.wrapping_sub(b'0');
        if c < 10 {
            f(c);
            *s = s.advance(1);
        } else {
            break;
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct AdjustedMantissa {
    pub mantissa: u64,
    pub power2: i32,
}

impl AdjustedMantissa {
    #[inline]
    pub const fn zero_pow2(power2: i32) -> Self {
        Self {
            mantissa: 0,
            power2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write_u64() {
        let bytes = b"01234567";
        let string = AsciiStr::new(bytes);
        let int = string.try_read_u64();
        assert_eq!(int, Some(0x3736353433323130));

        let int = unsafe { bytes.read_u64() };
        assert_eq!(int, 0x3736353433323130);

        let mut slc = [0u8; 8];
        unsafe { slc.write_u64(0x3736353433323130) };
        assert_eq!(&slc, bytes);
    }
}
