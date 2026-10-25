//! Legacy version of ChaCha20 with a 64-bit nonce

use crate::{ChaChaCore, Key, R20, variants::Legacy};
use cipher::{
    IvSizeUser, KeyIvInit, KeySizeUser, StreamCipherCoreWrapper,
    array::Array,
    consts::{U8, U32},
};

/// Nonce type used by [`ChaCha20Legacy`].
pub type LegacyNonce = Array<u8, U8>;

/// The ChaCha20 stream cipher (legacy "djb" construction with 64-bit nonce).
pub type ChaCha20Legacy = StreamCipherCoreWrapper<ChaCha20LegacyCore>;

/// The ChaCha20 stream cipher (legacy "djb" construction with 64-bit nonce).
pub type ChaCha20LegacyCore = ChaChaCore<R20, Legacy>;

impl KeySizeUser for ChaCha20LegacyCore {
    type KeySize = U32;
}

impl IvSizeUser for ChaCha20LegacyCore {
    type IvSize = U8;
}

impl KeyIvInit for ChaCha20LegacyCore {
    #[inline(always)]
    fn new(key: &Key, iv: &LegacyNonce) -> Self {
        ChaChaCore::<R20, Legacy>::new_internal(key.as_ref(), iv.as_ref())
    }
}
