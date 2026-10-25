//! Contains [`Error`]

/// Error from this crate
#[derive(Debug, Clone, Copy)]
pub enum Error {
    /// Generic error when the file is out of spec
    OutOfSpec,
    /// When reading or writing with compression but the feature flag "compression" is not active.
    RequiresCompression,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<std::io::Error> for Error {
    fn from(_: std::io::Error) -> Self {
        Error::OutOfSpec
    }
}
