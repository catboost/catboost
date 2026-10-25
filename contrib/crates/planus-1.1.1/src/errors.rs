/// The main error type for Planus
#[derive(Copy, Clone, Debug)]
pub struct Error {
    /// The location of the error
    pub source_location: ErrorLocation,
    /// The kind of error
    pub error_kind: ErrorKind,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "In {}: {}", self.source_location, self.error_kind)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error_kind)
    }
}

/// The possible errors in planus when reading data from a serialized buffer.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The offset was out of bounds.
    InvalidOffset,
    /// The buffer was too short while validating a length field.
    InvalidLength,
    /// A vector of unions had different lengths for the tag and value vectors.
    UnionVectorLengthsMismatched {
        /// The length of the tags vector
        tags_len: usize,
        /// The length of the values vector
        values_len: usize,
    },
    /// An enum contained an unknown value. For forward compatibility this
    /// error should be handled appropriately.
    UnknownEnumTag {
        /// The enum value that wasn't recognized.
        source: UnknownEnumTagKind,
    },
    /// An union contained an unknown variant. For forward compatibility this
    /// error should be handled appropriately.
    UnknownUnionTag {
        /// The union tag that wasn't recognized.
        tag: u8,
    },
    /// A vtable had an invalid length (too large, too small or unaligned).
    InvalidVtableLength {
        /// The length of the vtable.
        length: u16,
    },
    /// A string contained invalid utf-8.
    InvalidUtf8 {
        /// The utf-8 error triggered by the string.
        source: core::str::Utf8Error,
    },
    /// A required field was missing.
    MissingRequired,
    /// A string null terminator was missing.
    MissingNullTerminator,
}

impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ErrorKind::InvalidOffset => write!(f, "Invalid offset"),
            ErrorKind::InvalidLength => write!(f, "Invalid length"),
            ErrorKind::UnionVectorLengthsMismatched {
                tags_len,
                values_len,
            } => {
                write!(f, "Mismatched lengths between tag and value vectors. Length of tags = {tags_len}, length of values = {values_len}")
            }
            ErrorKind::UnknownEnumTag { source } => source.fmt(f),
            ErrorKind::UnknownUnionTag { tag } => write!(f, "Unknown union (tag = {tag})"),
            ErrorKind::InvalidVtableLength { length } => {
                write!(f, "Invalid vtable length (length = {length})")
            }
            ErrorKind::InvalidUtf8 { source } => write!(f, "Invalid utf-8: {source}"),
            ErrorKind::MissingRequired => write!(f, "Missing required field"),
            ErrorKind::MissingNullTerminator => write!(f, "Missing null terminator"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ErrorKind::InvalidOffset => None,
            ErrorKind::InvalidLength => None,
            ErrorKind::UnionVectorLengthsMismatched { .. } => None,
            ErrorKind::UnknownEnumTag { source } => Some(source),
            ErrorKind::UnknownUnionTag { .. } => None,
            ErrorKind::InvalidVtableLength { .. } => None,
            ErrorKind::InvalidUtf8 { source } => Some(source),
            ErrorKind::MissingRequired => None,
            ErrorKind::MissingNullTerminator => None,
        }
    }
}

impl From<UnknownEnumTagKind> for ErrorKind {
    fn from(source: UnknownEnumTagKind) -> Self {
        ErrorKind::UnknownEnumTag { source }
    }
}

impl From<core::str::Utf8Error> for ErrorKind {
    fn from(source: core::str::Utf8Error) -> Self {
        ErrorKind::InvalidUtf8 { source }
    }
}

#[derive(Clone, Debug)]
/// Information about an unrecognized enum tag.
///
/// In order to be forward compatible [`Result`]s with this error variant should
/// be handled gracefully.
pub struct UnknownEnumTag {
    /// The location of the unknown tag.
    pub source_location: ErrorLocation,
    /// The unknown tag.
    pub error_kind: UnknownEnumTagKind,
}

impl core::fmt::Display for UnknownEnumTag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "In {}: {}", self.source_location, self.error_kind)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UnknownEnumTag {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error_kind)
    }
}

#[derive(Copy, Clone, Debug)]
/// The value of an unknown enum tag.
pub struct UnknownEnumTagKind {
    /// The unknown tag.
    pub tag: i128,
}

impl core::fmt::Display for UnknownEnumTagKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Unknown enum (tag = {})", self.tag)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UnknownEnumTagKind {}

#[derive(Copy, Clone, Debug)]
/// The location of the error in both the generated code and the binary data
/// where it was encountered.
pub struct ErrorLocation {
    /// The flatbuffers type where the error was encountered.
    pub type_: &'static str,
    /// The generated method where the error was encountered.
    pub method: &'static str,
    /// Offset into the flatbuffers buffer where the error was encountered.
    pub byte_offset: usize,
}

impl core::fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.byte_offset != usize::MAX {
            write!(
                f,
                "<{}@{:x}>::{}()",
                self.type_, self.byte_offset, self.method,
            )
        } else {
            write!(f, "<{}>::{}()", self.type_, self.method,)
        }
    }
}

impl From<UnknownEnumTag> for Error {
    fn from(error: UnknownEnumTag) -> Self {
        Self {
            source_location: error.source_location,
            error_kind: error.error_kind.into(),
        }
    }
}

impl From<core::convert::Infallible> for Error {
    fn from(value: core::convert::Infallible) -> Self {
        match value {}
    }
}

impl UnknownEnumTagKind {
    /// Helper function that adds an error location to this error.
    pub fn with_error_location(
        self,
        type_: &'static str,
        method: &'static str,
        byte_offset: usize,
    ) -> UnknownEnumTag {
        UnknownEnumTag {
            source_location: ErrorLocation {
                type_,
                method,
                byte_offset,
            },
            error_kind: self,
        }
    }
}

impl ErrorKind {
    /// Helper function that adds an error location to this error.
    pub fn with_error_location(
        self,
        type_: &'static str,
        method: &'static str,
        byte_offset: usize,
    ) -> Error {
        Error {
            source_location: ErrorLocation {
                type_,
                method,
                byte_offset,
            },
            error_kind: self,
        }
    }
}
