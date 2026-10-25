use std::future::Future;
use std::io::{self, BufRead as _};
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, RawFd};
#[cfg(windows)]
use std::os::windows::io::{AsRawSocket, RawSocket};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use rustls::server::AcceptedAlert;
use rustls::{ServerConfig, ServerConnection};
use tokio::io::{AsyncBufRead, AsyncRead, AsyncWrite, ReadBuf};

use crate::common::{IoSession, MidHandshake, Stream, SyncReadAdapter, SyncWriteAdapter, TlsState};

/// A wrapper around a `rustls::ServerConfig`, providing an async `accept` method.
#[derive(Clone)]
pub struct TlsAcceptor {
    inner: Arc<ServerConfig>,
}

impl From<Arc<ServerConfig>> for TlsAcceptor {
    fn from(inner: Arc<ServerConfig>) -> Self {
        Self { inner }
    }
}

impl TlsAcceptor {
    /// Returns a future for completing a TLS handshake for a client using `stream`.
    ///
    /// You likely want to wrap this in a timeout (for example with [`tokio::time::timeout`][])
    /// to bound the handshake time.
    ///
    /// [`tokio::time::timeout`]: https://docs.rs/tokio/latest/tokio/time/fn.timeout.html
    #[inline]
    pub fn accept<IO>(&self, stream: IO) -> Accept<IO>
    where
        IO: AsyncRead + AsyncWrite + Unpin,
    {
        self.accept_with(stream, |_| ())
    }

    /// Similar to [`Self::accept()`], but calls `f` before performing the handshake.
    ///
    /// As with [`Self::accept()`] you likely want to wrap this in a timeout to
    /// bound the handshake time.
    ///
    /// The `f` handler is given a mutable reference to a [`ServerConnection`][] that can be used
    /// to configure the connection before the handshake, for example, adjusting the buffer limit.
    ///
    /// Because no data has been read from `stream` yet when `f` is called ClientHello
    /// dependent state (like early data) is not yet available.
    ///
    /// [`ServerConnection`]: https://docs.rs/rustls/latest/rustls/server/struct.ServerConnection.html
    pub fn accept_with<IO, F>(&self, stream: IO, f: F) -> Accept<IO>
    where
        IO: AsyncRead + AsyncWrite + Unpin,
        F: FnOnce(&mut ServerConnection),
    {
        let mut session = match ServerConnection::new(self.inner.clone()) {
            Ok(session) => session,
            Err(error) => {
                return Accept(MidHandshake::Error {
                    io: stream,
                    // TODO(eliza): should this really return an `io::Error`?
                    // Probably not...
                    error: io::Error::new(io::ErrorKind::Other, error),
                });
            }
        };
        f(&mut session);

        Accept(MidHandshake::Handshaking(TlsStream {
            session,
            io: stream,
            state: TlsState::Stream,
            need_flush: false,
            error: None,
        }))
    }

    /// Get a read-only reference to underlying config
    pub fn config(&self) -> &Arc<ServerConfig> {
        &self.inner
    }
}

/// A future for reading a `ClientHello` from `io` without committing to a [`ServerConfig`][].
///
/// Awaiting it yields a [`StartHandshake`], which exposes the
/// [`ClientHello`][] (for example, to choose a config based on SNI) and performs
/// the rest of the handshake via [`StartHandshake::into_stream()`].
///
/// [`ServerConfig`]: https://docs.rs/rustls/latest/rustls/server/struct.ServerConfig.html
/// [`ClientHello`]: https://docs.rs/rustls/latest/rustls/server/struct.ClientHello.html
pub struct LazyConfigAcceptor<IO> {
    acceptor: rustls::server::Acceptor,
    io: Option<IO>,
    alert: Option<(rustls::Error, AcceptedAlert)>,
}

impl<IO> LazyConfigAcceptor<IO>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    /// Returns a new `LazyConfigAcceptor` that reads a `ClientHello` from `io`.
    ///
    /// You likely want to wrap awaiting the acceptor in a timeout to bound how long the
    /// peer may take to send the `ClientHello`.
    ///
    /// Note that awaiting the acceptor is only the first half of the handshake and
    /// [`StartHandshake::into_stream()`] performs the rest.
    ///
    /// To bound the time for the complete handshake, share one deadline across
    /// both awaits (for example with [`tokio::time::timeout_at`][]) rather than giving each
    /// its own timeout.
    ///
    /// If a timeout elapses before the `ClientHello` arrives, [`Self::take_io()`] can
    /// recover the `io`, for example to answer the peer in plaintext before closing.
    ///
    /// [`tokio::time::timeout_at`]: https://docs.rs/tokio/latest/tokio/time/fn.timeout_at.html
    #[inline]
    pub fn new(acceptor: rustls::server::Acceptor, io: IO) -> Self {
        Self {
            acceptor,
            io: Some(io),
            alert: None,
        }
    }

    /// Takes back the client connection. Will return `None` if called more than once or if the
    /// connection has been accepted.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn choose_server_config(
    /// #     _: rustls::server::ClientHello,
    /// # ) -> std::sync::Arc<rustls::ServerConfig> {
    /// #     unimplemented!();
    /// # }
    /// # #[allow(unused_variables)]
    /// # async fn listen() {
    /// use tokio::io::AsyncWriteExt;
    /// let listener = tokio::net::TcpListener::bind("127.0.0.1:4443").await.unwrap();
    /// let (stream, _) = listener.accept().await.unwrap();
    ///
    /// let acceptor = tokio_rustls::LazyConfigAcceptor::new(rustls::server::Acceptor::default(), stream);
    /// tokio::pin!(acceptor);
    ///
    /// match acceptor.as_mut().await {
    ///     Ok(start) => {
    ///         let clientHello = start.client_hello();
    ///         let config = choose_server_config(clientHello);
    ///         let stream = start.into_stream(config).await.unwrap();
    ///         // Proceed with handling the ServerConnection...
    ///     }
    ///     Err(err) => {
    ///         if let Some(mut stream) = acceptor.take_io() {
    ///             stream
    ///                 .write_all(
    ///                     format!("HTTP/1.1 400 Invalid Input\r\n\r\n\r\n{:?}\n", err)
    ///                         .as_bytes()
    ///                 )
    ///                 .await
    ///                 .unwrap();
    ///         }
    ///     }
    /// }
    /// # }
    /// ```
    pub fn take_io(&mut self) -> Option<IO> {
        self.io.take()
    }
}

impl<IO> Future for LazyConfigAcceptor<IO>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    type Output = Result<StartHandshake<IO>, io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        loop {
            let io = match this.io.as_mut() {
                Some(io) => io,
                None => {
                    return Poll::Ready(Err(io::Error::new(
                        io::ErrorKind::Other,
                        "acceptor cannot be polled after acceptance",
                    )));
                }
            };

            if let Some((err, mut alert)) = this.alert.take() {
                match alert.write(&mut SyncWriteAdapter { io, cx }) {
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        this.alert = Some((err, alert));
                        return Poll::Pending;
                    }
                    Ok(0) | Err(_) => {
                        return Poll::Ready(Err(io::Error::new(io::ErrorKind::InvalidData, err)));
                    }
                    Ok(_) => {
                        this.alert = Some((err, alert));
                        continue;
                    }
                };
            }

            let mut reader = SyncReadAdapter { io, cx };
            match this.acceptor.read_tls(&mut reader) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()).into(),
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Poll::Pending,
                Err(e) => return Err(e).into(),
            }

            match this.acceptor.accept() {
                Ok(Some(accepted)) => {
                    let io = this.io.take().unwrap();
                    return Poll::Ready(Ok(StartHandshake { accepted, io }));
                }
                Ok(None) => {}
                Err((err, alert)) => {
                    this.alert = Some((err, alert));
                }
            }
        }
    }
}

/// An incoming connection received through [`LazyConfigAcceptor`].
///
/// This contains the generic `IO` asynchronous transport,
/// [`ClientHello`](rustls::server::ClientHello) data,
/// and all the state required to continue the TLS handshake (e.g. via
/// [`StartHandshake::into_stream`]).
#[non_exhaustive]
#[derive(Debug)]
pub struct StartHandshake<IO> {
    pub accepted: rustls::server::Accepted,
    pub io: IO,
}

impl<IO> StartHandshake<IO>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    /// Create a new object from an `IO` transport and prior TLS metadata.
    pub fn from_parts(accepted: rustls::server::Accepted, transport: IO) -> Self {
        Self {
            accepted,
            io: transport,
        }
    }

    pub fn client_hello(&self) -> rustls::server::ClientHello<'_> {
        self.accepted.client_hello()
    }

    /// Returns a future that performs the rest of the TLS handshake using `config`.
    ///
    /// You likely want to wrap this in a timeout to bound the handshake time. Ideally
    /// with [`tokio::time::timeout_at`][], reusing the deadline that also bounded
    /// awaiting the [`LazyConfigAcceptor`] so both halves of the handshake share
    /// one budget. See [`LazyConfigAcceptor::new()`].
    ///
    /// [`tokio::time::timeout_at`]: https://docs.rs/tokio/latest/tokio/time/fn.timeout_at.html
    pub fn into_stream(self, config: Arc<ServerConfig>) -> Accept<IO> {
        self.into_stream_with(config, |_| ())
    }

    /// Similar to [`Self::into_stream()`], but calls `f` before performing the handshake.
    ///
    /// As with [`Self::into_stream()`] you likely want to wrap this in a timeout to
    /// bound the handshake time.
    ///
    /// The `f` handler is given a mutable reference to a [`ServerConnection`][] that can be
    /// used to configure the connection before the handshake.
    ///
    /// [`ServerConnection`]: https://docs.rs/rustls/latest/rustls/server/struct.ServerConnection.html
    pub fn into_stream_with<F>(self, config: Arc<ServerConfig>, f: F) -> Accept<IO>
    where
        F: FnOnce(&mut ServerConnection),
    {
        let mut conn = match self.accepted.into_connection(config) {
            Ok(conn) => conn,
            Err((error, alert)) => {
                return Accept(MidHandshake::SendAlert {
                    io: self.io,
                    alert,
                    // TODO(eliza): should this really return an `io::Error`?
                    // Probably not...
                    error: io::Error::new(io::ErrorKind::InvalidData, error),
                });
            }
        };
        f(&mut conn);

        Accept(MidHandshake::Handshaking(TlsStream {
            session: conn,
            io: self.io,
            state: TlsState::Stream,
            need_flush: false,
            error: None,
        }))
    }
}

/// Future returned from `TlsAcceptor::accept` which will resolve
/// once the accept handshake has finished.
pub struct Accept<IO>(MidHandshake<TlsStream<IO>>);

impl<IO> Accept<IO> {
    #[inline]
    pub fn into_fallible(self) -> FallibleAccept<IO> {
        FallibleAccept(self.0)
    }

    pub fn get_ref(&self) -> Option<&IO> {
        match &self.0 {
            MidHandshake::Handshaking(sess) => Some(sess.get_ref().0),
            MidHandshake::SendAlert { io, .. } => Some(io),
            MidHandshake::Error { io, .. } => Some(io),
            MidHandshake::End => None,
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut IO> {
        match &mut self.0 {
            MidHandshake::Handshaking(sess) => Some(sess.get_mut().0),
            MidHandshake::SendAlert { io, .. } => Some(io),
            MidHandshake::Error { io, .. } => Some(io),
            MidHandshake::End => None,
        }
    }
}

impl<IO: AsyncRead + AsyncWrite + Unpin> Future for Accept<IO> {
    type Output = io::Result<TlsStream<IO>>;

    #[inline]
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx).map_err(|(err, _)| err)
    }
}

/// Like [Accept], but returns `IO` on failure.
pub struct FallibleAccept<IO>(MidHandshake<TlsStream<IO>>);

impl<IO: AsyncRead + AsyncWrite + Unpin> Future for FallibleAccept<IO> {
    type Output = Result<TlsStream<IO>, (io::Error, IO)>;

    #[inline]
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.0).poll(cx)
    }
}

/// A wrapper around an underlying raw stream which implements the TLS or SSL
/// protocol.
#[derive(Debug)]
pub struct TlsStream<IO> {
    pub(crate) io: IO,
    pub(crate) session: ServerConnection,
    pub(crate) state: TlsState,
    pub(crate) need_flush: bool,
    /// Buffered error that occurred during batch reading
    pub(crate) error: Option<io::Error>,
}

impl<IO> TlsStream<IO> {
    #[inline]
    pub fn get_ref(&self) -> (&IO, &ServerConnection) {
        (&self.io, &self.session)
    }

    #[inline]
    pub fn get_mut(&mut self) -> (&mut IO, &mut ServerConnection) {
        (&mut self.io, &mut self.session)
    }

    #[inline]
    pub fn into_inner(self) -> (IO, ServerConnection) {
        (self.io, self.session)
    }
}

impl<IO> IoSession for TlsStream<IO> {
    type Io = IO;
    type Session = ServerConnection;

    #[inline]
    fn skip_handshake(&self) -> bool {
        false
    }

    #[inline]
    fn get_mut(&mut self) -> (&mut TlsState, &mut Self::Io, &mut Self::Session, &mut bool) {
        (
            &mut self.state,
            &mut self.io,
            &mut self.session,
            &mut self.need_flush,
        )
    }

    #[inline]
    fn into_io(self) -> Self::Io {
        self.io
    }
}

impl<IO> AsyncRead for TlsStream<IO>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if let Some(err) = self.error.take() {
            return Poll::Ready(Err(err));
        };
        let data = ready!(self.as_mut().poll_fill_buf(cx))?;
        let len = data.len().min(buf.remaining());
        if len == 0 {
            return Poll::Ready(Ok(()));
        }
        buf.put_slice(&data[..len]);
        self.as_mut().consume(len);

        while buf.remaining() > 0 {
            let data = match self.as_mut().poll_fill_buf(cx) {
                Poll::Ready(Ok([])) => break,
                Poll::Ready(Ok(data)) => data,
                Poll::Ready(Err(err)) => {
                    self.error = Some(err);
                    break;
                }
                Poll::Pending => break,
            };
            let len = Ord::min(data.len(), buf.remaining());
            buf.put_slice(&data[..len]);
            self.as_mut().consume(len);
        }
        Poll::Ready(Ok(()))
    }
}

impl<IO> AsyncBufRead for TlsStream<IO>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<&[u8]>> {
        match self.state {
            TlsState::Stream | TlsState::WriteShutdown => {
                let this = self.get_mut();
                let stream =
                    Stream::new(&mut this.io, &mut this.session).set_eof(!this.state.readable());

                match stream.poll_fill_buf(cx) {
                    Poll::Ready(Ok(buf)) => {
                        if buf.is_empty() {
                            this.state.shutdown_read();
                        }

                        Poll::Ready(Ok(buf))
                    }
                    Poll::Ready(Err(err)) if err.kind() == io::ErrorKind::ConnectionAborted => {
                        this.state.shutdown_read();
                        Poll::Ready(Err(err))
                    }
                    output => output,
                }
            }
            TlsState::ReadShutdown | TlsState::FullyShutdown => Poll::Ready(Ok(&[])),
            #[cfg(feature = "early-data")]
            ref s => unreachable!("server TLS can not hit this state: {:?}", s),
        }
    }

    fn consume(mut self: Pin<&mut Self>, amt: usize) {
        self.session.reader().consume(amt);
    }
}

impl<IO> AsyncWrite for TlsStream<IO>
where
    IO: AsyncRead + AsyncWrite + Unpin,
{
    /// Note: that it does not guarantee the final data to be sent.
    /// To be cautious, you must manually call `flush`.
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut stream =
            Stream::new(&mut this.io, &mut this.session).set_eof(!this.state.readable());
        stream.as_mut_pin().poll_write(cx, buf)
    }

    /// Note: that it does not guarantee the final data to be sent.
    /// To be cautious, you must manually call `flush`.
    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let mut stream =
            Stream::new(&mut this.io, &mut this.session).set_eof(!this.state.readable());
        stream.as_mut_pin().poll_write_vectored(cx, bufs)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        true
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let mut stream =
            Stream::new(&mut this.io, &mut this.session).set_eof(!this.state.readable());
        stream.as_mut_pin().poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.state.writeable() {
            self.session.send_close_notify();
            self.state.shutdown_write();
        }

        let this = self.get_mut();
        let mut stream =
            Stream::new(&mut this.io, &mut this.session).set_eof(!this.state.readable());
        stream.as_mut_pin().poll_shutdown(cx)
    }
}

#[cfg(unix)]
impl<IO> AsRawFd for TlsStream<IO>
where
    IO: AsRawFd,
{
    fn as_raw_fd(&self) -> RawFd {
        self.get_ref().0.as_raw_fd()
    }
}

#[cfg(windows)]
impl<IO> AsRawSocket for TlsStream<IO>
where
    IO: AsRawSocket,
{
    fn as_raw_socket(&self) -> RawSocket {
        self.get_ref().0.as_raw_socket()
    }
}
