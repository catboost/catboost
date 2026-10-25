//! The interface and implementations of LZ77 compression algorithm.
//!
//! LZ77 is a compression algorithm used in [DEFLATE](https://tools.ietf.org/html/rfc1951).
pub use libflate_lz77::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deflate::symbol::Symbol;

    #[test]
    // See: https://github.com/sile/libflate/issues/21
    fn issue21() {
        let mut enc = DefaultLz77Encoder::new();
        let mut sink = Vec::<Symbol>::new();
        enc.encode(b"aaaaa", &mut sink);
        enc.flush(&mut sink);
        assert_eq!(
            sink,
            vec![
                Symbol::Code(Code::Literal(97)),
                Symbol::Code(Code::Pointer {
                    length: 4,
                    backward_distance: 1
                })
            ]
        );
    }
}
