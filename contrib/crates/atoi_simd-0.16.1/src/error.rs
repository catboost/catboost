use ::core::fmt;

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum AtoiSimdError<'a> {
    Empty,
    Size(usize, &'a [u8]),
    Overflow(&'a [u8]),
    Invalid64(u64, usize, &'a [u8]),
    Invalid128(u128, usize, &'a [u8]),
}

impl fmt::Display for AtoiSimdError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "atoi_simd string is empty"),
            Self::Size(len, input) => {
                write!(
                    f,
                    "atoi_simd wrong size: {} input: {:X?}",
                    len,
                    &input[..input.len().min(48)]
                )
            }
            Self::Overflow(input) => {
                write!(
                    f,
                    "atoi_simd overflow, input: {:X?}",
                    &input[..input.len().min(48)]
                )
            }
            Self::Invalid64(res, index, input) => {
                write!(
                    f,
                    "atoi_simd invalid at index: {} it must contain only digits, starting with: {}  input: {:X?}",
                    index, res, &input[..input.len().min(48)]
                )
            }
            Self::Invalid128(res, index, input) => {
                write!(
                    f,
                    "atoi_simd invalid at index: {} it must contain only digits, starting with: {} input: {:X?}",
                    index, res, &input[..input.len().min(48)]
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for AtoiSimdError<'_> {}
