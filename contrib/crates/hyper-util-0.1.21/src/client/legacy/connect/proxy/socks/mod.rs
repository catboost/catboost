mod v5;
pub use v5::{SocksV5, SocksV5Error};

mod v4;
pub use v4::{SocksV4, SocksV4Error};

use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::{Buf, BytesMut};

use hyper::rt::Read;

#[derive(Debug)]
pub enum SocksError<C> {
    Inner(C),
    Io(std::io::Error),

    DnsFailure,
    MissingHost,
    MissingPort,

    V4(SocksV4Error),
    V5(SocksV5Error),

    Parsing(ParsingError),
    Serialize(SerializeError),
}

#[derive(Debug)]
pub enum ParsingError {
    Incomplete,
    WouldOverflow,
    Other,
}

#[derive(Debug)]
pub enum SerializeError {
    WouldOverflow,
}

async fn read_message<T, M, C>(mut conn: &mut T, buf: &mut BytesMut) -> Result<M, SocksError<C>>
where
    T: Read + Unpin,
    M: for<'a, 'b> TryFrom<&'a mut &'b [u8], Error = ParsingError>,
{
    let mut tmp = [0; 513];

    loop {
        let mut view = &buf[..];
        match M::try_from(&mut view) {
            Err(ParsingError::Incomplete) => {
                let n = crate::rt::read(&mut conn, &mut tmp).await?;

                if n == 0 {
                    if buf.spare_capacity_mut().is_empty() {
                        return Err(SocksError::Parsing(ParsingError::WouldOverflow));
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "unexpected eof",
                        )
                        .into());
                    }
                }

                buf.extend_from_slice(&tmp[..n]);
            }
            Err(err) => return Err(err.into()),
            Ok(res) => {
                let consumed = buf.len() - view.len();
                buf.advance(consumed);
                return Ok(res);
            }
        }
    }
}

impl<C> std::fmt::Display for SocksError<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SOCKS error: ")?;

        match self {
            Self::Inner(_) => f.write_str("failed to create underlying connection"),
            Self::Io(_) => f.write_str("io error during SOCKS handshake"),

            Self::DnsFailure => f.write_str("could not resolve to acceptable address type"),
            Self::MissingHost => f.write_str("missing destination host"),
            Self::MissingPort => f.write_str("missing destination port"),

            Self::Parsing(_) => f.write_str("failed parsing server response"),
            Self::Serialize(_) => f.write_str("failed serialize request"),

            Self::V4(e) => e.fmt(f),
            Self::V5(e) => e.fmt(f),
        }
    }
}

impl<C: std::fmt::Debug + std::fmt::Display> std::error::Error for SocksError<C> {}

impl<C> From<std::io::Error> for SocksError<C> {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl<C> From<ParsingError> for SocksError<C> {
    fn from(err: ParsingError) -> Self {
        Self::Parsing(err)
    }
}

impl<C> From<SerializeError> for SocksError<C> {
    fn from(err: SerializeError) -> Self {
        Self::Serialize(err)
    }
}

impl<C> From<SocksV4Error> for SocksError<C> {
    fn from(err: SocksV4Error) -> Self {
        Self::V4(err)
    }
}

impl<C> From<SocksV5Error> for SocksError<C> {
    fn from(err: SocksV5Error) -> Self {
        Self::V5(err)
    }
}

pin_project! {
    // Not publicly exported (so missing_docs doesn't trigger).
    //
    // We return this `Future` instead of the `Pin<Box<dyn Future>>` directly
    // so that users don't rely on it fitting in a `Pin<Box<dyn Future>>` slot
    // (and thus we can change the type in the future).
    #[must_use = "futures do nothing unless polled"]
    #[allow(missing_debug_implementations)]
    pub struct Handshaking<F, T, E> {
        #[pin]
        fut: BoxHandshaking<T, E>,
        _marker: std::marker::PhantomData<F>
    }
}

type BoxHandshaking<T, E> = Pin<Box<dyn Future<Output = Result<T, SocksError<E>>> + Send>>;

impl<F, T, E> Future for Handshaking<F, T, E>
where
    F: Future<Output = Result<T, E>>,
{
    type Output = Result<T, SocksError<E>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().fut.poll(cx)
    }
}

#[cfg(all(test, feature = "tokio"))]
mod test {
    use bytes::BytesMut;
    use tokio::io::AsyncWriteExt;

    use super::v5::messages::{AuthMethod, AuthenticationRes, NegotiationRes, ProxyRes, Status};
    use super::{SocksError, read_message};
    use crate::rt::TokioIo;

    // A SOCKS5 ProxyRes message. Successful, bound to 127.0.0.1:8080.
    const SEG1: [u8; 4] = [0x05, 0x00, 0x00, 0x01];
    const SEG2: [u8; 6] = [0x7F, 0x00, 0x00, 0x01, 0x1F, 0x90];

    // A SOCKS5 NegotiationRes message: username/password method selected.
    const NEG_RES: [u8; 2] = [0x05, 0x02];
    // A SOCKS5 AuthenticationRes message: success.
    const AUTH_RES: [u8; 2] = [0x01, 0x00];

    #[tokio::test]
    async fn it_works_in_one_read() {
        let (client, mut server) = tokio::io::duplex(SEG1.len() + SEG2.len());
        server.write_all(&SEG1).await.unwrap();
        server.write_all(&SEG2).await.unwrap();

        let mut conn = TokioIo::new(client);
        let mut buf = BytesMut::new();

        let m: Result<ProxyRes, SocksError<()>> = read_message(&mut conn, &mut buf).await;
        assert!(m.is_ok());
        assert_eq!(m.unwrap(), ProxyRes(Status::Success));
    }

    #[tokio::test]
    async fn it_works_in_multiple_reads() {
        // Bounded stream ensures message arrives in two reads
        let (client, mut server) = tokio::io::duplex(SEG1.len());
        let _writer = tokio::spawn(async move {
            server.write_all(&SEG1).await.unwrap();
            server.write_all(&SEG2).await.unwrap();
        });

        let mut conn = TokioIo::new(client);
        let mut buf = BytesMut::new();

        let m: Result<ProxyRes, SocksError<()>> = read_message(&mut conn, &mut buf).await;
        assert!(m.is_ok());
        assert_eq!(m.unwrap(), ProxyRes(Status::Success));

        _writer.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn optimistic_sending_works_in_single_read() {
        // Messages will arrive in a single read
        let message = [&NEG_RES[..], &AUTH_RES[..], &SEG1[..], &SEG2[..]].concat();
        let (client, mut server) = tokio::io::duplex(message.len());
        server.write_all(&message).await.unwrap();

        let mut conn = TokioIo::new(client);
        let mut buf = BytesMut::new();

        let m: Result<NegotiationRes, SocksError<()>> = read_message(&mut conn, &mut buf).await;
        assert_eq!(m.unwrap(), NegotiationRes(AuthMethod::UserPass));

        let m: Result<AuthenticationRes, SocksError<()>> = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            read_message(&mut conn, &mut buf),
        )
        .await
        .expect("second message should be parsed from the buffer, not read from the socket");
        assert_eq!(m.unwrap(), AuthenticationRes(true));

        let m: Result<ProxyRes, SocksError<()>> = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            read_message(&mut conn, &mut buf),
        )
        .await
        .expect("third message should be parsed from the buffer, not read from the socket");
        assert_eq!(m.unwrap(), ProxyRes(Status::Success));

        assert!(buf.is_empty(), "all handshake bytes should be consumed");
        drop(server);
    }

    #[tokio::test]
    async fn optimistic_sending_works_in_multiple_reads() {
        // Bounded stream ensures message arrive in multiple reads
        let message = [&NEG_RES[..], &AUTH_RES[..], &SEG1[..], &SEG2[..]].concat();
        let (client, mut server) = tokio::io::duplex(message.len() / 4);
        let _writer = tokio::spawn(async move {
            server.write_all(&message).await.unwrap();
            server
        });

        let mut conn = TokioIo::new(client);
        let mut buf = BytesMut::new();

        let m: Result<NegotiationRes, SocksError<()>> = read_message(&mut conn, &mut buf).await;
        assert_eq!(m.unwrap(), NegotiationRes(AuthMethod::UserPass));

        let m: Result<AuthenticationRes, SocksError<()>> = read_message(&mut conn, &mut buf).await;
        assert_eq!(m.unwrap(), AuthenticationRes(true));

        let m: Result<ProxyRes, SocksError<()>> = read_message(&mut conn, &mut buf).await;
        assert_eq!(m.unwrap(), ProxyRes(Status::Success));

        assert!(buf.is_empty(), "all handshake bytes should be consumed");

        let server = _writer.await.unwrap();
        drop(server);
    }
}
