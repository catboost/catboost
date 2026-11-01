use alloc::string::String;
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

use self::map::lookup;
mod map;

#[derive(Clone, Copy, Debug, Default)]
pub struct Unicode<S>(pub S);

impl<S: AsRef<str>> Unicode<S> {
    fn to_folded_chars(&self) -> impl Iterator<Item = char> + '_ {
        self.0.as_ref().chars().flat_map(lookup)
    }

    pub fn to_folded_case(&self) -> String {
        self.to_folded_chars().collect()
    }
}

impl<S1: AsRef<str>, S2: AsRef<str>> PartialEq<Unicode<S2>> for Unicode<S1> {
    #[inline]
    fn eq(&self, other: &Unicode<S2>) -> bool {
        self.to_folded_chars().eq(other.to_folded_chars())
    }
}

impl<S: AsRef<str>> Eq for Unicode<S> {}

impl<T: AsRef<str>> PartialOrd for Unicode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: AsRef<str>> Ord for Unicode<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_folded_chars().cmp(other.to_folded_chars())
    }
}

impl<S: AsRef<str>> Hash for Unicode<S> {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        let mut buf = [0; 4];
        for c in self.to_folded_chars() {
            let len = c.encode_utf8(&mut buf).len();
            // we can't use `write(buf)` because the ASCII variant uses
            // `write_u8`. The docs for Hash say that's technically different.
            // ¯\_(ツ)_/¯
            for &b in &buf[..len] {
                hasher.write_u8(b);
            }
        }
        // prefix-freedom
        hasher.write_u8(0xFF);
    }
}

// internal mod so that the enum can be 'pub'
// thanks privacy-checker :___(
mod fold {
    #[derive(Clone, Copy)]
    pub enum Fold {
        Zero,
        One(char),
        Two(char, char),
        Three(char, char, char),
    }

    impl Iterator for Fold {
        type Item = char;
        #[inline]
        fn next(&mut self) -> Option<char> {
            match *self {
                Fold::Zero => None,
                Fold::One(one) => {
                    *self = Fold::Zero;
                    Some(one)
                }
                Fold::Two(one, two) => {
                    *self = Fold::One(two);
                    Some(one)
                }
                Fold::Three(one, two, three) => {
                    *self = Fold::Two(two, three);
                    Some(one)
                }
            }
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            match *self {
                Fold::Zero => (0, Some(0)),
                Fold::One(..) => (1, Some(1)),
                Fold::Two(..) => (2, Some(2)),
                Fold::Three(..) => (3, Some(3)),
            }
        }
    }
    impl From<(char,)> for Fold {
        #[inline]
        fn from((one,): (char,)) -> Fold {
            Fold::One(one)
        }
    }

    impl From<(char, char)> for Fold {
        #[inline]
        fn from((one, two): (char, char)) -> Fold {
            Fold::Two(one, two)
        }
    }

    impl From<(char, char, char)> for Fold {
        #[inline]
        fn from((one, two, three): (char, char, char)) -> Fold {
            Fold::Three(one, two, three)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Unicode;

    macro_rules! eq {
        ($left:expr, $right:expr) => {{
            assert_eq!(Unicode($left), Unicode($right));
        }};
    }

    #[test]
    fn test_ascii_folding() {
        eq!("foo bar", "FoO BAR");
    }

    #[test]
    fn test_simple_case_folding() {
        eq!("στιγμας", "στιγμασ");
    }

    #[test]
    fn test_full_case_folding() {
        eq!("ﬂour", "flour");
        eq!("Maße", "MASSE");
        eq!("ᾲ στο διάολο", "ὰι στο διάολο");
    }

    #[test]
    fn test_three_char_case_folding() {
        // Characters whose full case folding is three characters.
        eq!("ﬃ", "ffi");
        eq!("ﬄ", "FFL");
        eq!("ΐ", "ι\u{308}\u{301}");
        eq!("ᾷ", "α\u{342}ι");
        assert_eq!(Unicode("ﬃ").to_folded_case(), "ffi");
    }

    #[test]
    fn test_to_folded_case() {
        assert_eq!(Unicode("Maße").to_folded_case(), "masse");
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_ascii_folding(b: &mut ::test::Bencher) {
        b.bytes = b"foo bar".len() as u64;
        b.iter(|| eq!("foo bar", "FoO BAR"));
    }

    #[cfg(feature = "nightly")]
    #[bench]
    fn bench_simple_case_folding(b: &mut ::test::Bencher) {
        b.bytes = "στιγμας".len() as u64;
        b.iter(|| eq!("στιγμας", "στιγμασ"));
    }
}
