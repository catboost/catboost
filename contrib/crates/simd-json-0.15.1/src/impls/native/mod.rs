#![allow(unused_imports, dead_code)]
use simdutf8::basic::imp::ChunkedUtf8Validator;

mod deser;
mod stage1;

pub(crate) use deser::parse_str;
pub(crate) use stage1::SimdInput;

/// This is a hack, since there is no native implementation of the chunked validator we pre-validate the entire
/// input string in the case of a fallback and then always let the chunked validator return true.
pub(crate) struct ChunkedUtf8ValidatorImp();

impl ChunkedUtf8Validator for ChunkedUtf8ValidatorImp {
    unsafe fn new() -> Self
    where
        Self: Sized,
    {
        ChunkedUtf8ValidatorImp()
    }

    unsafe fn update_from_chunks(&mut self, _input: &[u8]) {}

    unsafe fn finalize(
        self,
        _remaining_input: core::option::Option<&[u8]>,
    ) -> core::result::Result<(), simdutf8::basic::Utf8Error> {
        Ok(())
    }
}
