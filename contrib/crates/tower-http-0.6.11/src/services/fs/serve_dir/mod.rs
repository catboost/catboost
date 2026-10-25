use self::future::ResponseFuture;
use crate::{
    body::UnsyncBoxBody,
    content_encoding::{encodings, SupportedEncodings},
    set_status::SetStatus,
};
use bytes::Bytes;
use futures_util::FutureExt;
use http::{header, HeaderValue, Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Empty};
use percent_encoding::percent_decode;
use std::{
    convert::Infallible,
    io,
    path::{Component, Path, PathBuf},
    task::{Context, Poll},
};
use tower_service::Service;

pub(crate) mod future;
mod headers;
mod open_file;

#[cfg(test)]
mod tests;

// default capacity 64KiB
const DEFAULT_CAPACITY: usize = 65536;

/// Service that serves files from a given directory and all its sub directories.
///
/// The `Content-Type` will be guessed from the file extension.
///
/// An empty response with status `404 Not Found` will be returned if:
///
/// - The file doesn't exist
/// - Any segment of the path contains `..`
/// - Any segment of the path contains a backslash
/// - On unix, any segment of the path referenced as directory is actually an
///   existing file (`/file.html/something`)
/// - We don't have necessary permissions to read the file
///
/// # Example
///
/// ```
/// use tower_http::services::ServeDir;
///
/// // This will serve files in the "assets" directory and
/// // its subdirectories
/// let service = ServeDir::new("assets");
/// ```
#[derive(Clone, Debug)]
pub struct ServeDir<F = DefaultServeDirFallback> {
    base: PathBuf,
    buf_chunk_size: usize,
    precompressed_variants: Option<PrecompressedVariants>,
    // This is used to specialise implementation for
    // single files
    variant: ServeVariant,
    fallback: Option<F>,
    call_fallback_on_method_not_allowed: bool,
}

impl ServeDir<DefaultServeDirFallback> {
    /// Create a new [`ServeDir`].
    pub fn new<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        let mut base = PathBuf::from(".");
        base.push(path.as_ref());

        Self {
            base,
            buf_chunk_size: DEFAULT_CAPACITY,
            precompressed_variants: None,
            variant: ServeVariant::Directory {
                append_index_html_on_directories: true,
            },
            fallback: None,
            call_fallback_on_method_not_allowed: false,
        }
    }

    pub(crate) fn new_single_file<P>(path: P, mime: HeaderValue) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            base: path.as_ref().to_owned(),
            buf_chunk_size: DEFAULT_CAPACITY,
            precompressed_variants: None,
            variant: ServeVariant::SingleFile { mime },
            fallback: None,
            call_fallback_on_method_not_allowed: false,
        }
    }
}

impl<F> ServeDir<F> {
    /// If the requested path is a directory append `index.html`.
    ///
    /// This is useful for static sites.
    ///
    /// Defaults to `true`.
    pub fn append_index_html_on_directories(mut self, append: bool) -> Self {
        match &mut self.variant {
            ServeVariant::Directory {
                append_index_html_on_directories,
            } => {
                *append_index_html_on_directories = append;
                self
            }
            ServeVariant::SingleFile { mime: _ } => self,
        }
    }

    /// Set a specific read buffer chunk size.
    ///
    /// The default capacity is 64kb.
    pub fn with_buf_chunk_size(mut self, chunk_size: usize) -> Self {
        self.buf_chunk_size = chunk_size;
        self
    }

    /// Informs the service that it should also look for a precompressed gzip
    /// version of _any_ file in the directory.
    ///
    /// Assuming the `dir` directory is being served and `dir/foo.txt` is requested,
    /// a client with an `Accept-Encoding` header that allows the gzip encoding
    /// will receive the file `dir/foo.txt.gz` instead of `dir/foo.txt`.
    /// If the precompressed file is not available, or the client doesn't support it,
    /// the uncompressed version will be served instead.
    /// Both the precompressed version and the uncompressed version are expected
    /// to be present in the directory. Different precompressed variants can be combined.
    pub fn precompressed_gzip(mut self) -> Self {
        self.precompressed_variants
            .get_or_insert(Default::default())
            .gzip = true;
        self
    }

    /// Informs the service that it should also look for a precompressed brotli
    /// version of _any_ file in the directory.
    ///
    /// Assuming the `dir` directory is being served and `dir/foo.txt` is requested,
    /// a client with an `Accept-Encoding` header that allows the brotli encoding
    /// will receive the file `dir/foo.txt.br` instead of `dir/foo.txt`.
    /// If the precompressed file is not available, or the client doesn't support it,
    /// the uncompressed version will be served instead.
    /// Both the precompressed version and the uncompressed version are expected
    /// to be present in the directory. Different precompressed variants can be combined.
    pub fn precompressed_br(mut self) -> Self {
        self.precompressed_variants
            .get_or_insert(Default::default())
            .br = true;
        self
    }

    /// Informs the service that it should also look for a precompressed deflate
    /// version of _any_ file in the directory.
    ///
    /// Assuming the `dir` directory is being served and `dir/foo.txt` is requested,
    /// a client with an `Accept-Encoding` header that allows the deflate encoding
    /// will receive the file `dir/foo.txt.zz` instead of `dir/foo.txt`.
    /// If the precompressed file is not available, or the client doesn't support it,
    /// the uncompressed version will be served instead.
    /// Both the precompressed version and the uncompressed version are expected
    /// to be present in the directory. Different precompressed variants can be combined.
    pub fn precompressed_deflate(mut self) -> Self {
        self.precompressed_variants
            .get_or_insert(Default::default())
            .deflate = true;
        self
    }

    /// Informs the service that it should also look for a precompressed zstd
    /// version of _any_ file in the directory.
    ///
    /// Assuming the `dir` directory is being served and `dir/foo.txt` is requested,
    /// a client with an `Accept-Encoding` header that allows the zstd encoding
    /// will receive the file `dir/foo.txt.zst` instead of `dir/foo.txt`.
    /// If the precompressed file is not available, or the client doesn't support it,
    /// the uncompressed version will be served instead.
    /// Both the precompressed version and the uncompressed version are expected
    /// to be present in the directory. Different precompressed variants can be combined.
    pub fn precompressed_zstd(mut self) -> Self {
        self.precompressed_variants
            .get_or_insert(Default::default())
            .zstd = true;
        self
    }

    /// Set the fallback service.
    ///
    /// This service will be called if there is no file at the path of the request.
    ///
    /// The status code returned by the fallback will not be altered. Use
    /// [`ServeDir::not_found_service`] to set a fallback and always respond with `404 Not Found`.
    ///
    /// # Example
    ///
    /// This can be used to respond with a different file:
    ///
    /// ```rust
    /// use tower_http::services::{ServeDir, ServeFile};
    ///
    /// let service = ServeDir::new("assets")
    ///     // respond with `not_found.html` for missing files
    ///     .fallback(ServeFile::new("assets/not_found.html"));
    /// ```
    pub fn fallback<F2>(self, new_fallback: F2) -> ServeDir<F2> {
        ServeDir {
            base: self.base,
            buf_chunk_size: self.buf_chunk_size,
            precompressed_variants: self.precompressed_variants,
            variant: self.variant,
            fallback: Some(new_fallback),
            call_fallback_on_method_not_allowed: self.call_fallback_on_method_not_allowed,
        }
    }

    /// Set the fallback service and override the fallback's status code to `404 Not Found`.
    ///
    /// This service will be called if there is no file at the path of the request.
    ///
    /// # Example
    ///
    /// This can be used to respond with a different file:
    ///
    /// ```rust
    /// use tower_http::services::{ServeDir, ServeFile};
    ///
    /// let service = ServeDir::new("assets")
    ///     // respond with `404 Not Found` and the contents of `not_found.html` for missing files
    ///     .not_found_service(ServeFile::new("assets/not_found.html"));
    /// ```
    ///
    /// Setups like this are often found in single page applications.
    pub fn not_found_service<F2>(self, new_fallback: F2) -> ServeDir<SetStatus<F2>> {
        self.fallback(SetStatus::new(new_fallback, StatusCode::NOT_FOUND))
    }

    /// Customize whether or not to call the fallback for requests that aren't `GET` or `HEAD`.
    ///
    /// Defaults to not calling the fallback and instead returning `405 Method Not Allowed`.
    pub fn call_fallback_on_method_not_allowed(mut self, call_fallback: bool) -> Self {
        self.call_fallback_on_method_not_allowed = call_fallback;
        self
    }

    /// Call the service and get a future that contains any `std::io::Error` that might have
    /// happened.
    ///
    /// By default `<ServeDir as Service<_>>::call` will handle IO errors and convert them into
    /// responses. It does that by converting [`std::io::ErrorKind::NotFound`] and
    /// [`std::io::ErrorKind::PermissionDenied`] to `404 Not Found` and any other error to `500
    /// Internal Server Error`. The error will also be logged with `tracing` in case the `tracing`
    /// crate feature is enabled.
    ///
    /// If you want to manually control how the error response is generated you can make a new
    /// service that wraps a `ServeDir` and calls `try_call` instead of `call`.
    ///
    /// # Example
    ///
    /// ```
    /// use tower_http::services::ServeDir;
    /// use std::{io, convert::Infallible};
    /// use http::{Request, Response, StatusCode};
    /// use http_body::Body as _;
    /// use http_body_util::{Full, BodyExt, combinators::UnsyncBoxBody};
    /// use bytes::Bytes;
    /// use tower::{service_fn, ServiceExt, BoxError};
    ///
    /// async fn serve_dir(
    ///     request: Request<Full<Bytes>>
    /// ) -> Result<Response<UnsyncBoxBody<Bytes, BoxError>>, Infallible> {
    ///     let mut service = ServeDir::new("assets");
    ///
    ///     // You only need to worry about backpressure, and thus call `ServiceExt::ready`, if
    ///     // your adding a fallback to `ServeDir` that cares about backpressure.
    ///     //
    ///     // Its shown here for demonstration but you can do `service.try_call(request)`
    ///     // otherwise
    ///     let ready_service = match ServiceExt::<Request<Full<Bytes>>>::ready(&mut service).await {
    ///         Ok(ready_service) => ready_service,
    ///         Err(infallible) => match infallible {},
    ///     };
    ///
    ///     match ready_service.try_call(request).await {
    ///         Ok(response) => {
    ///             Ok(response.map(|body| body.map_err(Into::into).boxed_unsync()))
    ///         }
    ///         Err(err) => {
    ///             let body = Full::from("Something went wrong...")
    ///                 .map_err(Into::into)
    ///                 .boxed_unsync();
    ///             let response = Response::builder()
    ///                 .status(StatusCode::INTERNAL_SERVER_ERROR)
    ///                 .body(body)
    ///                 .unwrap();
    ///             Ok(response)
    ///         }
    ///     }
    /// }
    /// ```
    pub fn try_call<ReqBody, FResBody>(
        &mut self,
        req: Request<ReqBody>,
    ) -> ResponseFuture<ReqBody, F>
    where
        F: Service<Request<ReqBody>, Response = Response<FResBody>, Error = Infallible> + Clone,
        F::Future: Send + 'static,
        FResBody: http_body::Body<Data = Bytes> + Send + 'static,
        FResBody::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        if req.method() != Method::GET && req.method() != Method::HEAD {
            if self.call_fallback_on_method_not_allowed {
                if let Some(fallback) = &mut self.fallback {
                    return ResponseFuture {
                        inner: future::call_fallback(fallback, req),
                    };
                }
            }

            return ResponseFuture::method_not_allowed();
        }

        // `ServeDir` doesn't care about the request body but the fallback might. So move out the
        // body and pass it to the fallback, leaving an empty body in its place
        //
        // this is necessary because we cannot clone bodies
        let (mut parts, body) = req.into_parts();
        // same goes for extensions
        let extensions = std::mem::take(&mut parts.extensions);
        let req = Request::from_parts(parts, Empty::<Bytes>::new());

        let fallback_and_request = self.fallback.as_mut().map(|fallback| {
            let mut fallback_req = Request::new(body);
            *fallback_req.method_mut() = req.method().clone();
            *fallback_req.uri_mut() = req.uri().clone();
            *fallback_req.headers_mut() = req.headers().clone();
            *fallback_req.extensions_mut() = extensions;

            // get the ready fallback and leave a non-ready clone in its place
            let clone = fallback.clone();
            let fallback = std::mem::replace(fallback, clone);

            (fallback, fallback_req)
        });

        let path_to_file = match self
            .variant
            .build_and_validate_path(&self.base, req.uri().path())
        {
            Some(path_to_file) => path_to_file,
            None => {
                return ResponseFuture::invalid_path(fallback_and_request);
            }
        };

        let buf_chunk_size = self.buf_chunk_size;
        let range_header = req
            .headers()
            .get(header::RANGE)
            .and_then(|value| value.to_str().ok())
            .map(|s| s.to_owned());

        let negotiated_encodings: Vec<_> = encodings(
            req.headers(),
            self.precompressed_variants.unwrap_or_default(),
        )
        .collect();

        let variant = self.variant.clone();

        let open_file_future = Box::pin(open_file::open_file(
            variant,
            path_to_file,
            req,
            negotiated_encodings,
            range_header,
            buf_chunk_size,
        ));

        ResponseFuture::open_file_future(open_file_future, fallback_and_request)
    }
}

impl<ReqBody, F, FResBody> Service<Request<ReqBody>> for ServeDir<F>
where
    F: Service<Request<ReqBody>, Response = Response<FResBody>, Error = Infallible> + Clone,
    F::Future: Send + 'static,
    FResBody: http_body::Body<Data = Bytes> + Send + 'static,
    FResBody::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    type Response = Response<ResponseBody>;
    type Error = Infallible;
    type Future = InfallibleResponseFuture<ReqBody, F>;

    #[inline]
    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        if let Some(fallback) = &mut self.fallback {
            fallback.poll_ready(cx)
        } else {
            Poll::Ready(Ok(()))
        }
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let future = self
            .try_call(req)
            .map(|result: Result<_, _>| -> Result<_, Infallible> {
                let response = result.unwrap_or_else(|_err| {
                    #[cfg(feature = "tracing")]
                    tracing::error!(error = %_err, "Failed to read file");

                    let body = ResponseBody::new(UnsyncBoxBody::new(
                        Empty::new().map_err(|err| match err {}).boxed_unsync(),
                    ));
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(body)
                        .unwrap()
                });
                Ok(response)
            } as _);

        InfallibleResponseFuture::new(future)
    }
}

opaque_future! {
    /// Response future of [`ServeDir`].
    pub type InfallibleResponseFuture<ReqBody, F> =
        futures_util::future::Map<
            ResponseFuture<ReqBody, F>,
            fn(Result<Response<ResponseBody>, io::Error>) -> Result<Response<ResponseBody>, Infallible>,
        >;
}

// Allow the ServeDir service to be used in the ServeFile service
// with almost no overhead
#[derive(Clone, Debug)]
enum ServeVariant {
    Directory {
        append_index_html_on_directories: bool,
    },
    SingleFile {
        mime: HeaderValue,
    },
}

impl ServeVariant {
    fn build_and_validate_path(&self, base_path: &Path, requested_path: &str) -> Option<PathBuf> {
        match self {
            ServeVariant::Directory {
                append_index_html_on_directories: _,
            } => {
                let path = requested_path.trim_start_matches('/');

                let path_decoded = percent_decode(path.as_ref()).decode_utf8().ok()?;
                let path_decoded = Path::new(&*path_decoded);

                let mut path_to_file = base_path.to_path_buf();
                for component in path_decoded.components() {
                    match component {
                        Component::Normal(comp) => {
                            // protect against paths like `/foo/c:/bar/baz` (#204)
                            if Path::new(&comp)
                                .components()
                                .all(|c| matches!(c, Component::Normal(_)))
                            {
                                #[cfg(windows)]
                                {
                                    use std::os::windows::ffi::OsStrExt;
                                    if is_reserved_dos_name(|| comp.encode_wide()) {
                                        return None;
                                    }
                                }

                                path_to_file.push(comp)
                            } else {
                                return None;
                            }
                        }
                        Component::CurDir => {}
                        Component::Prefix(_) | Component::RootDir | Component::ParentDir => {
                            return None;
                        }
                    }
                }
                Some(path_to_file)
            }
            ServeVariant::SingleFile { mime: _ } => Some(base_path.to_path_buf()),
        }
    }
}

/// Check whether a component name matches a reserved Windows DOS device name.
/// See: https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file#naming-conventions
///
/// We explicitly check for Unicode superscript characters `¹` (0x00B9), `²` (0x00B2),
/// and `³` (0x00B3) because older character tables (ISO/IEC 8859-1) define these values,
/// which legacy Win32 file parsing resolves natively as valid port numbers (0..9).
///
/// This uses an iterator and stack array to avoid allocating. A closure is used because it
/// iterates the characters twice. The closure must return the same iterator each time it is
/// called.
#[cfg(any(windows, test))]
fn is_reserved_dos_name<F, I>(mut get_iter: F) -> bool
where
    F: FnMut() -> I,
    I: Iterator<Item = u16>,
{
    const CON: [u16; 3] = [b'C' as u16, b'O' as u16, b'N' as u16];
    const PRN: [u16; 3] = [b'P' as u16, b'R' as u16, b'N' as u16];
    const AUX: [u16; 3] = [b'A' as u16, b'U' as u16, b'X' as u16];
    const NUL: [u16; 3] = [b'N' as u16, b'U' as u16, b'L' as u16];
    const CONIN: [u16; 6] = [
        b'C' as u16,
        b'O' as u16,
        b'N' as u16,
        b'I' as u16,
        b'N' as u16,
        b'$' as u16,
    ];
    const CONOUT: [u16; 7] = [
        b'C' as u16,
        b'O' as u16,
        b'N' as u16,
        b'O' as u16,
        b'U' as u16,
        b'T' as u16,
        b'$' as u16,
    ];

    const COM: [u16; 3] = [b'C' as u16, b'O' as u16, b'M' as u16];
    const LPT: [u16; 3] = [b'L' as u16, b'P' as u16, b'T' as u16];

    const ZERO: u16 = b'0' as u16;
    const NINE: u16 = b'9' as u16;
    const SUPERSCRIPT_ONE: u16 = 0x00B9;
    const SUPERSCRIPT_TWO: u16 = 0x00B2;
    const SUPERSCRIPT_THREE: u16 = 0x00B3;

    fn is_whitespace(c: u16) -> bool {
        c <= 0x7F && ((c as u8).is_ascii_whitespace() || c == 0x000B)
    }

    // In a first pass over the string, obtain the length of the basename.
    let trimmed_len = get_iter()
        .enumerate()
        // We want the base name, so stop at '.' or ':' characters.
        .take_while(|&(_idx, c)| c != b'.' as u16 && c != b':' as u16)
        // We want to trim whitespace from the end, so ignore whitespace chars.
        .filter(|&(_idx, c)| !is_whitespace(c))
        // Get the last non-whitespace char before the first '.'/':' character.
        .last()
        // Convert index of that char into length of string.
        .map(|(idx, _)| idx + 1)
        .unwrap_or(0);

    // If the trimmed base name is longer than 7, it cannot be a reserved name.
    if trimmed_len > 7 {
        return false;
    }

    // At this point, we can store the string in an array, which is more convenient to work with.
    let mut buf = [0u16; 7];
    get_iter()
        .take(trimmed_len)
        .enumerate()
        .for_each(|(i, c)| buf[i] = c);

    for b in &mut buf {
        if *b <= 0x7F {
            *b = (*b as u8).to_ascii_uppercase() as u16;
        }
        if *b == SUPERSCRIPT_ONE {
            *b = b'1' as u16;
        }
        if *b == SUPERSCRIPT_TWO {
            *b = b'2' as u16;
        }
        if *b == SUPERSCRIPT_THREE {
            *b = b'3' as u16;
        }
    }
    let name = &buf[..trimmed_len];

    // Check basic fixed-length strings
    if name == CON || name == PRN || name == AUX || name == NUL || name == CONIN || name == CONOUT {
        return true;
    }

    // COMx / LPTx
    if name.len() == 4 {
        let prefix = &name[..3];
        let suffix = name[3];

        if (prefix == COM || prefix == LPT) && matches!(suffix, ZERO..=NINE) {
            return true;
        }
    }

    false
}

opaque_body! {
    /// Response body for [`ServeDir`] and [`ServeFile`][super::ServeFile].
    #[derive(Default)]
    pub type ResponseBody = UnsyncBoxBody<Bytes, io::Error>;
}

/// The default fallback service used with [`ServeDir`].
#[derive(Debug, Clone, Copy)]
pub struct DefaultServeDirFallback(Infallible);

impl<ReqBody> Service<Request<ReqBody>> for DefaultServeDirFallback
where
    ReqBody: Send + 'static,
{
    type Response = Response<ResponseBody>;
    type Error = Infallible;
    type Future = InfallibleResponseFuture<ReqBody, Self>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        match self.0 {}
    }

    fn call(&mut self, _req: Request<ReqBody>) -> Self::Future {
        match self.0 {}
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PrecompressedVariants {
    gzip: bool,
    deflate: bool,
    br: bool,
    zstd: bool,
}

impl SupportedEncodings for PrecompressedVariants {
    fn gzip(&self) -> bool {
        self.gzip
    }

    fn deflate(&self) -> bool {
        self.deflate
    }

    fn br(&self) -> bool {
        self.br
    }

    fn zstd(&self) -> bool {
        self.zstd
    }
}
