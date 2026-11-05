use std::convert::TryFrom;
use std::str::FromStr;
use std::{cmp, fmt, hash, str};

use bytes::Bytes;

use super::{ErrorKind, InvalidUri, MAX_LEN};
use crate::byte_str::ByteStr;

/// Represents the path component of a URI
#[derive(Clone)]
pub struct PathAndQuery {
    pub(super) data: ByteStr,
    pub(super) query: u16,
}

const NONE: u16 = u16::MAX;

impl PathAndQuery {
    // Not public while `bytes` is unstable.
    pub(super) fn from_shared(mut src: Bytes) -> Result<Self, InvalidUri> {
        let Scanned {
            query,
            fragment,
            is_maybe_not_utf8,
        } = scan_path_and_query(&src)?;

        if let Some(i) = fragment {
            src.truncate(i as usize);
        }

        let data = if is_maybe_not_utf8 {
            ByteStr::from_utf8(src).map_err(|_| ErrorKind::InvalidUriChar)?
        } else {
            unsafe { ByteStr::from_utf8_unchecked(src) }
        };

        Ok(PathAndQuery { data, query })
    }

    /// Convert a `PathAndQuery` from a static string.
    ///
    /// This function will not perform any copying, however the string is
    /// checked to ensure that it is valid.
    ///
    /// # Panics
    ///
    /// This function panics if the argument is an invalid path and query.
    ///
    /// # Examples
    ///
    /// ```
    /// # use http::uri::*;
    /// let v = PathAndQuery::from_static("/hello?world");
    ///
    /// assert_eq!(v.path(), "/hello");
    /// assert_eq!(v.query(), Some("world"));
    /// ```
    #[inline]
    pub const fn from_static(src: &'static str) -> Self {
        match scan_path_and_query(src.as_bytes()) {
            Ok(Scanned {
                query,
                fragment: None,
                is_maybe_not_utf8: false,
            }) => PathAndQuery {
                data: ByteStr::from_static(src),
                query,
            },
            // Yes, we reject fragments and non-utf8
            _ => panic!("static str is not valid path"),
        }
    }

    /// Attempt to convert a `Bytes` buffer to a `PathAndQuery`.
    ///
    /// This will try to prevent a copy if the type passed is the type used
    /// internally, and will copy the data if it is not.
    pub fn from_maybe_shared<T>(src: T) -> Result<Self, InvalidUri>
    where
        T: AsRef<[u8]> + 'static,
    {
        if_downcast_into!(T, Bytes, src, {
            return PathAndQuery::from_shared(src);
        });

        PathAndQuery::try_from(src.as_ref())
    }

    pub(super) fn empty() -> Self {
        PathAndQuery {
            data: ByteStr::new(),
            query: NONE,
        }
    }

    pub(super) fn slash() -> Self {
        PathAndQuery {
            data: ByteStr::from_static("/"),
            query: NONE,
        }
    }

    pub(super) fn star() -> Self {
        PathAndQuery {
            data: ByteStr::from_static("*"),
            query: NONE,
        }
    }

    /// Returns the path component
    ///
    /// The path component is **case sensitive**.
    ///
    /// ```notrust
    /// abc://username:password@example.com:123/path/data?key=value&key2=value2#fragid1
    ///                                        |--------|
    ///                                             |
    ///                                           path
    /// ```
    ///
    /// If the URI is `*` then the path component is equal to `*`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use http::uri::*;
    ///
    /// let path_and_query: PathAndQuery = "/hello/world".parse().unwrap();
    ///
    /// assert_eq!(path_and_query.path(), "/hello/world");
    /// ```
    #[inline]
    pub fn path(&self) -> &str {
        let ret = if self.query == NONE {
            &self.data[..]
        } else {
            &self.data[..self.query as usize]
        };

        if ret.is_empty() {
            return "/";
        }

        ret
    }

    /// Returns the query string component
    ///
    /// The query component contains non-hierarchical data that, along with data
    /// in the path component, serves to identify a resource within the scope of
    /// the URI's scheme and naming authority (if any). The query component is
    /// indicated by the first question mark ("?") character and terminated by a
    /// number sign ("#") character or by the end of the URI.
    ///
    /// ```notrust
    /// abc://username:password@example.com:123/path/data?key=value&key2=value2#fragid1
    ///                                                   |-------------------|
    ///                                                             |
    ///                                                           query
    /// ```
    ///
    /// # Examples
    ///
    /// With a query string component
    ///
    /// ```
    /// # use http::uri::*;
    /// let path_and_query: PathAndQuery = "/hello/world?key=value&foo=bar".parse().unwrap();
    ///
    /// assert_eq!(path_and_query.query(), Some("key=value&foo=bar"));
    /// ```
    ///
    /// Without a query string component
    ///
    /// ```
    /// # use http::uri::*;
    /// let path_and_query: PathAndQuery = "/hello/world".parse().unwrap();
    ///
    /// assert!(path_and_query.query().is_none());
    /// ```
    #[inline]
    pub fn query(&self) -> Option<&str> {
        if self.query == NONE {
            None
        } else {
            let i = self.query + 1;
            Some(&self.data[i as usize..])
        }
    }

    /// Returns the path and query as a string component.
    ///
    /// # Examples
    ///
    /// With a query string component
    ///
    /// ```
    /// # use http::uri::*;
    /// let path_and_query: PathAndQuery = "/hello/world?key=value&foo=bar".parse().unwrap();
    ///
    /// assert_eq!(path_and_query.as_str(), "/hello/world?key=value&foo=bar");
    /// ```
    ///
    /// Without a query string component
    ///
    /// ```
    /// # use http::uri::*;
    /// let path_and_query: PathAndQuery = "/hello/world".parse().unwrap();
    ///
    /// assert_eq!(path_and_query.as_str(), "/hello/world");
    /// ```
    #[inline]
    pub fn as_str(&self) -> &str {
        let ret = &self.data[..];
        if ret.is_empty() {
            return "/";
        }
        ret
    }
}

impl TryFrom<&[u8]> for PathAndQuery {
    type Error = InvalidUri;
    #[inline]
    fn try_from(s: &[u8]) -> Result<Self, Self::Error> {
        PathAndQuery::from_shared(Bytes::copy_from_slice(s))
    }
}

impl TryFrom<&str> for PathAndQuery {
    type Error = InvalidUri;
    #[inline]
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        TryFrom::try_from(s.as_bytes())
    }
}

impl TryFrom<Vec<u8>> for PathAndQuery {
    type Error = InvalidUri;
    #[inline]
    fn try_from(vec: Vec<u8>) -> Result<Self, Self::Error> {
        PathAndQuery::from_shared(vec.into())
    }
}

impl TryFrom<String> for PathAndQuery {
    type Error = InvalidUri;
    #[inline]
    fn try_from(s: String) -> Result<Self, Self::Error> {
        PathAndQuery::from_shared(s.into())
    }
}

impl TryFrom<&String> for PathAndQuery {
    type Error = InvalidUri;
    #[inline]
    fn try_from(s: &String) -> Result<Self, Self::Error> {
        TryFrom::try_from(s.as_bytes())
    }
}

impl FromStr for PathAndQuery {
    type Err = InvalidUri;
    #[inline]
    fn from_str(s: &str) -> Result<Self, InvalidUri> {
        TryFrom::try_from(s)
    }
}

impl fmt::Debug for PathAndQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for PathAndQuery {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.data.is_empty() {
            match self.data.as_bytes()[0] {
                b'/' | b'*' => fmt.write_str(&self.data),
                _ => {
                    fmt.write_str("/")?;
                    fmt.write_str(&self.data)
                }
            }
        } else {
            fmt.write_str("/")
        }
    }
}

impl hash::Hash for PathAndQuery {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

// ===== PartialEq / PartialOrd =====

impl PartialEq for PathAndQuery {
    #[inline]
    fn eq(&self, other: &PathAndQuery) -> bool {
        self.data == other.data
    }
}

impl Eq for PathAndQuery {}

impl PartialEq<str> for PathAndQuery {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<PathAndQuery> for &str {
    #[inline]
    fn eq(&self, other: &PathAndQuery) -> bool {
        self == &other.as_str()
    }
}

impl PartialEq<&str> for PathAndQuery {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<PathAndQuery> for str {
    #[inline]
    fn eq(&self, other: &PathAndQuery) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<String> for PathAndQuery {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<PathAndQuery> for String {
    #[inline]
    fn eq(&self, other: &PathAndQuery) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialOrd for PathAndQuery {
    #[inline]
    fn partial_cmp(&self, other: &PathAndQuery) -> Option<cmp::Ordering> {
        self.as_str().partial_cmp(other.as_str())
    }
}

impl PartialOrd<str> for PathAndQuery {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        self.as_str().partial_cmp(other)
    }
}

impl PartialOrd<PathAndQuery> for str {
    #[inline]
    fn partial_cmp(&self, other: &PathAndQuery) -> Option<cmp::Ordering> {
        self.partial_cmp(other.as_str())
    }
}

impl PartialOrd<&str> for PathAndQuery {
    #[inline]
    fn partial_cmp(&self, other: &&str) -> Option<cmp::Ordering> {
        self.as_str().partial_cmp(*other)
    }
}

impl PartialOrd<PathAndQuery> for &str {
    #[inline]
    fn partial_cmp(&self, other: &PathAndQuery) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.as_str())
    }
}

impl PartialOrd<String> for PathAndQuery {
    #[inline]
    fn partial_cmp(&self, other: &String) -> Option<cmp::Ordering> {
        self.as_str().partial_cmp(other.as_str())
    }
}

impl PartialOrd<PathAndQuery> for String {
    #[inline]
    fn partial_cmp(&self, other: &PathAndQuery) -> Option<cmp::Ordering> {
        self.as_str().partial_cmp(other.as_str())
    }
}

// Scanner implementation that is `const fn`, usable by both `from_static`
// and `from_shared`.
// =====

struct Scanned {
    query: u16,
    fragment: Option<u16>,
    is_maybe_not_utf8: bool,
}

// Per-byte character classes for the path and query scanners.
const CLASS_VALID: u8 = 0;
const CLASS_QUERY: u8 = 1;
const CLASS_FRAGMENT: u8 = 2;
const CLASS_HIGH: u8 = 3;
const CLASS_INVALID: u8 = 4;

const fn build_path_map() -> [u8; 256] {
    let mut t = [CLASS_INVALID; 256];
    let mut i = 0;
    while i < 256 {
        // See https://url.spec.whatwg.org/#path-state
        t[i] = match i as u8 {
            b'?' => CLASS_QUERY,
            b'#' => CLASS_FRAGMENT,

            // Bytes that don't need to be percent-encoded in the path.
            0x21 | 0x24..=0x3B | 0x3D | 0x40..=0x5F | 0x61..=0x7A | 0x7C | 0x7E => CLASS_VALID,

            // Potentially utf8, checked later.
            0x80..=0xFF => CLASS_HIGH,

            // Should be percent-encoded, but accepted for parity with clients
            // that send them as-is (e.g. JSON embedded in the path).
            b'"' | b'{' | b'}' => CLASS_VALID,

            _ => CLASS_INVALID,
        };
        i += 1;
    }
    t
}

const fn build_query_map() -> [u8; 256] {
    let mut t = [CLASS_INVALID; 256];
    let mut i = 0;
    while i < 256 {
        // See https://url.spec.whatwg.org/#query-state
        t[i] = match i as u8 {
            b'#' => CLASS_FRAGMENT,

            // Allowed: 0x21 / 0x24 - 0x3B / 0x3D / 0x3F - 0x7E
            0x21 | 0x24..=0x3B | 0x3D | 0x3F..=0x7E => CLASS_VALID,

            0x80..=0xFF => CLASS_HIGH,

            _ => CLASS_INVALID,
        };
        i += 1;
    }
    t
}

const PATH_MAP: [u8; 256] = build_path_map();
const QUERY_MAP: [u8; 256] = build_query_map();

const fn scan_path_and_query(bytes: &[u8]) -> Result<Scanned, ErrorKind> {
    let mut i = 0;
    let mut query = NONE;
    let mut fragment = None;

    let mut is_maybe_not_utf8 = false;

    if bytes.is_empty() {
        return Err(ErrorKind::Empty);
    }

    if bytes.len() > MAX_LEN {
        return Err(ErrorKind::TooLong);
    }

    if bytes.len() == 1 && bytes[0] == b'*' {
        return Ok(Scanned {
            query,
            fragment,
            is_maybe_not_utf8: false,
        });
    }

    if !matches!(bytes[0], b'/' | b'?' | b'#') {
        return Err(ErrorKind::PathDoesNotStartWithSlash);
    }

    while i < bytes.len() {
        match PATH_MAP[bytes[i] as usize] {
            CLASS_VALID => {}
            CLASS_QUERY => {
                debug_assert!(query == NONE);
                query = i as u16;
                i += 1;
                break;
            }
            CLASS_FRAGMENT => {
                fragment = Some(i as u16);
                break;
            }
            CLASS_HIGH => {
                is_maybe_not_utf8 = true;
            }
            _ => return Err(ErrorKind::InvalidUriChar),
        }
        i += 1;
    }

    // query ...
    if query != NONE {
        while i < bytes.len() {
            match QUERY_MAP[bytes[i] as usize] {
                CLASS_VALID => {}
                CLASS_HIGH => {
                    is_maybe_not_utf8 = true;
                }
                CLASS_FRAGMENT => {
                    fragment = Some(i as u16);
                    break;
                }
                _ => return Err(ErrorKind::InvalidUriChar),
            }
            i += 1;
        }
    }

    Ok(Scanned {
        query,
        fragment,
        is_maybe_not_utf8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_to_self_of_same_path() {
        let p1: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        let p2: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        assert_eq!(p1, p2);
        assert_eq!(p2, p1);
    }

    #[test]
    fn not_equal_to_self_of_different_path() {
        let p1: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        let p2: PathAndQuery = "/world&foo=bar".parse().unwrap();
        assert_ne!(p1, p2);
        assert_ne!(p2, p1);
    }

    #[test]
    fn equates_with_a_str() {
        let path_and_query: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        assert_eq!(&path_and_query, "/hello/world&foo=bar");
        assert_eq!("/hello/world&foo=bar", &path_and_query);
        assert_eq!(path_and_query, "/hello/world&foo=bar");
        assert_eq!("/hello/world&foo=bar", path_and_query);
    }

    #[test]
    fn not_equal_with_a_str_of_a_different_path() {
        let path_and_query: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        // as a reference
        assert_ne!(&path_and_query, "/hello&foo=bar");
        assert_ne!("/hello&foo=bar", &path_and_query);
        // without reference
        assert_ne!(path_and_query, "/hello&foo=bar");
        assert_ne!("/hello&foo=bar", path_and_query);
    }

    #[test]
    fn equates_with_a_string() {
        let path_and_query: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        assert_eq!(path_and_query, "/hello/world&foo=bar".to_string());
        assert_eq!("/hello/world&foo=bar".to_string(), path_and_query);
    }

    #[test]
    fn not_equal_with_a_string_of_a_different_path() {
        let path_and_query: PathAndQuery = "/hello/world&foo=bar".parse().unwrap();
        assert_ne!(path_and_query, "/hello&foo=bar".to_string());
        assert_ne!("/hello&foo=bar".to_string(), path_and_query);
    }

    #[test]
    fn compares_to_self() {
        let p1: PathAndQuery = "/a/world&foo=bar".parse().unwrap();
        let p2: PathAndQuery = "/b/world&foo=bar".parse().unwrap();
        assert!(p1 < p2);
        assert!(p2 > p1);
    }

    #[test]
    fn compares_with_a_str() {
        let path_and_query: PathAndQuery = "/b/world&foo=bar".parse().unwrap();
        // by ref
        assert!(&path_and_query < "/c/world&foo=bar");
        assert!("/c/world&foo=bar" > &path_and_query);
        assert!(&path_and_query > "/a/world&foo=bar");
        assert!("/a/world&foo=bar" < &path_and_query);

        // by val
        assert!(path_and_query < "/c/world&foo=bar");
        assert!("/c/world&foo=bar" > path_and_query);
        assert!(path_and_query > "/a/world&foo=bar");
        assert!("/a/world&foo=bar" < path_and_query);
    }

    #[test]
    fn compares_with_a_string() {
        let path_and_query: PathAndQuery = "/b/world&foo=bar".parse().unwrap();
        assert!(path_and_query < "/c/world&foo=bar".to_string());
        assert!("/c/world&foo=bar".to_string() > path_and_query);
        assert!(path_and_query > "/a/world&foo=bar".to_string());
        assert!("/a/world&foo=bar".to_string() < path_and_query);
    }

    #[test]
    fn ignores_valid_percent_encodings() {
        assert_eq!("/a%20b", pq("/a%20b?r=1").path());
        assert_eq!("qr=%31", pq("/a/b?qr=%31").query().unwrap());
    }

    #[test]
    fn ignores_invalid_percent_encodings() {
        assert_eq!("/a%%b", pq("/a%%b?r=1").path());
        assert_eq!("/aaa%", pq("/aaa%").path());
        assert_eq!("/aaa%", pq("/aaa%?r=1").path());
        assert_eq!("/aa%2", pq("/aa%2").path());
        assert_eq!("/aa%2", pq("/aa%2?r=1").path());
        assert_eq!("qr=%3", pq("/a/b?qr=%3").query().unwrap());
    }

    #[test]
    fn allow_utf8_in_path() {
        assert_eq!("/🍕", pq("/🍕").path());
    }

    #[test]
    fn allow_utf8_in_query() {
        assert_eq!(Some("pizza=🍕"), pq("/test?pizza=🍕").query());
    }

    #[test]
    fn rejects_invalid_utf8_in_path() {
        PathAndQuery::try_from(&[b'/', 0xFF][..]).expect_err("reject invalid utf8");
    }

    #[test]
    fn rejects_invalid_utf8_in_query() {
        PathAndQuery::try_from(&[b'/', b'a', b'?', 0xFF][..]).expect_err("reject invalid utf8");
    }

    #[test]
    fn rejects_empty_string() {
        PathAndQuery::try_from("").expect_err("reject empty str");
    }

    #[test]
    fn requires_starting_with_slash() {
        PathAndQuery::try_from("sneaky").expect_err("reject missing slash");
    }

    #[test]
    fn rejects_del_in_path() {
        PathAndQuery::try_from(&[b'/', 0x7F][..]).expect_err("reject DEL");
    }

    #[test]
    fn rejects_del_in_query() {
        PathAndQuery::try_from(&[b'/', b'a', b'?', 0x7F][..]).expect_err("reject DEL");
    }

    #[test]
    fn rejects_too_long_path_and_query() {
        let path = format!("/{}?query", "a".repeat(MAX_LEN));
        let err = PathAndQuery::try_from(path).expect_err("reject overly long path and query");
        assert_eq!(err.0, ErrorKind::TooLong);
    }

    #[test]
    fn accepts_max_length_path_and_query() {
        let path = format!("/{}?", "a".repeat(MAX_LEN - 2));
        let path_and_query = PathAndQuery::try_from(path).expect("accept maximum length");
        assert_eq!(path_and_query.as_str().len(), MAX_LEN);
        assert_eq!(path_and_query.query(), Some(""));
    }

    #[test]
    fn json_is_fine() {
        assert_eq!(
            r#"/{"bread":"baguette"}"#,
            pq(r#"/{"bread":"baguette"}"#).path()
        );
    }

    fn pq(s: &str) -> PathAndQuery {
        s.parse().expect(&format!("parsing {}", s))
    }
}
