//! Functions to compress and write Files' metadata and blocks
mod compression;
pub use compression::compress;
mod block;
pub mod encode;
pub(crate) mod file;
pub use block::write_block;
pub use file::write_metadata;
