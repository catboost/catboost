// Originally sourced from `futures_util::io::buf_writer`, needs to be redefined locally so that
// the `AsyncBufWrite` impl can access its internals, and changed a bit to make it more efficient
// with those methods.

use crate::generic::write::impl_buf_writer;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::io::AsyncWrite;

impl_buf_writer!(poll_shutdown);
