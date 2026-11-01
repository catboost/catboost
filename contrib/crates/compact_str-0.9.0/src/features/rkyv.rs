#![cfg_attr(docsrs, doc(cfg(feature = "rkyv")))]

use rkyv::rancor::{Fallible, Source};
use rkyv::string::{ArchivedString, StringResolver};
use rkyv::{Archive, Deserialize, DeserializeUnsized, Place, Serialize, SerializeUnsized};

use crate::CompactString;

impl Archive for CompactString {
    type Archived = ArchivedString;
    type Resolver = StringResolver;

    #[inline]
    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        ArchivedString::resolve_from_str(self.as_str(), resolver, out);
    }
}

impl<S: Fallible + ?Sized> Serialize<S> for CompactString
where
    str: SerializeUnsized<S>,
    S::Error: Source,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        ArchivedString::serialize_from_str(self.as_str(), serializer)
    }
}

impl<D: Fallible + ?Sized> Deserialize<CompactString, D> for ArchivedString
where
    str: DeserializeUnsized<str, D>,
{
    #[inline]
    fn deserialize(&self, _: &mut D) -> Result<CompactString, D::Error> {
        Ok(self.as_str().into())
    }
}

impl PartialEq<CompactString> for ArchivedString {
    #[inline]
    fn eq(&self, other: &CompactString) -> bool {
        PartialEq::eq(self.as_str(), other.as_str())
    }
}

impl PartialOrd<CompactString> for ArchivedString {
    #[inline]
    fn partial_cmp(&self, other: &CompactString) -> Option<core::cmp::Ordering> {
        PartialOrd::partial_cmp(self.as_str(), other.as_str())
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use rkyv::string::ArchivedString;
    use rkyv::{rancor, Archive};
    use test_strategy::proptest;

    use crate::CompactString;

    #[cfg_attr(miri, ignore)] // https://github.com/rust-lang/unsafe-code-guidelines/issues/134
    #[test]
    fn test_roundtrip() {
        const VALUE: &str = "Hello, 🌍!";

        let bytes_compact = rkyv::to_bytes::<rancor::Error>(&CompactString::from(VALUE)).unwrap();
        let bytes_control = rkyv::to_bytes::<rancor::Error>(&String::from(VALUE)).unwrap();
        assert_eq!(&*bytes_compact, &*bytes_control);

        let archived = rkyv::access::<ArchivedString, rancor::Error>(&bytes_compact).unwrap();
        let compact = rkyv::deserialize::<CompactString, rancor::Error>(archived).unwrap();
        let control = rkyv::deserialize::<String, rancor::Error>(archived).unwrap();
        assert_eq!(archived, VALUE);
        assert_eq!(compact, VALUE);
        assert_eq!(control, VALUE);

        let archived = rkyv::access::<ArchivedString, rancor::Error>(&bytes_compact).unwrap();
        let compact = rkyv::deserialize::<CompactString, rancor::Error>(archived).unwrap();
        let control = rkyv::deserialize::<String, rancor::Error>(archived).unwrap();
        assert_eq!(archived, VALUE);
        assert_eq!(compact, VALUE);
        assert_eq!(control, VALUE);
    }

    #[cfg_attr(miri, ignore)]
    #[proptest]
    fn proptest_roundtrip(s: String) {
        let bytes_compact = rkyv::to_bytes::<rancor::Error>(&CompactString::from(&s)).unwrap();
        let bytes_control = rkyv::to_bytes::<rancor::Error>(&s).unwrap();
        assert_eq!(&*bytes_compact, &*bytes_control);

        let archived =
            rkyv::access::<<CompactString as Archive>::Archived, rancor::Error>(&bytes_compact)
                .unwrap();
        let compact = rkyv::deserialize::<CompactString, rancor::Error>(archived).unwrap();
        let control = rkyv::deserialize::<String, rancor::Error>(archived).unwrap();
        assert_eq!(archived, &s);
        assert_eq!(compact, s);
        assert_eq!(control, s);

        let archived =
            rkyv::access::<<String as Archive>::Archived, rancor::Error>(&bytes_compact).unwrap();
        let compact = rkyv::deserialize::<CompactString, rancor::Error>(archived).unwrap();
        let control = rkyv::deserialize::<String, rancor::Error>(archived).unwrap();
        assert_eq!(archived, &s);
        assert_eq!(compact, s);
        assert_eq!(control, s);
    }
}
