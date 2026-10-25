use cipher::{
    IvSizeUser, KeyIvInit, KeySizeUser, StreamCipherCoreWrapper,
    array::Array,
    consts::{U12, U32, U64},
};

use crate::{ChaChaCore, R8, R12, R20, Rounds, variants::Ietf};

/// Key type used by all ChaCha variants.
pub type Key = Array<u8, U32>;

/// Nonce type used by ChaCha variants.
pub type Nonce = Array<u8, U12>;

/// ChaCha8 stream cipher (reduced-round variant of [`ChaCha20`] with 8 rounds)
pub type ChaCha8 = StreamCipherCoreWrapper<ChaChaCore<R8, Ietf>>;

/// ChaCha12 stream cipher (reduced-round variant of [`ChaCha20`] with 12 rounds)
pub type ChaCha12 = StreamCipherCoreWrapper<ChaChaCore<R12, Ietf>>;

/// ChaCha20 stream cipher (RFC 8439 version with 96-bit nonce)
pub type ChaCha20 = StreamCipherCoreWrapper<ChaChaCore<R20, Ietf>>;

pub(crate) type Block = Array<u8, U64>;

impl<R: Rounds> KeySizeUser for ChaChaCore<R, Ietf> {
    type KeySize = U32;
}

impl<R: Rounds> IvSizeUser for ChaChaCore<R, Ietf> {
    type IvSize = U12;
}
impl<R: Rounds> KeyIvInit for ChaChaCore<R, Ietf> {
    #[inline]
    fn new(key: &Key, iv: &Nonce) -> Self {
        ChaChaCore::<R, Ietf>::new_internal(key.as_ref(), iv.as_ref())
    }
}
