// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`RetryConfig`] connection retry policy

use crate::client::backoff::{Backoff, BackoffConfig};
use crate::client::builder::HttpRequestBuilder;
use crate::client::{HttpClient, HttpError, HttpErrorKind, HttpRequest, HttpResponse};
use crate::PutPayload;
use futures::future::BoxFuture;
use http::{Method, Uri};
use reqwest::header::LOCATION;
use reqwest::StatusCode;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::{Duration, Instant};
use tracing::info;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use web_time::{Duration, Instant};

/// Retry request error
#[derive(Debug)]
pub struct RetryError(Box<RetryErrorImpl>);

/// Box error to avoid large error variant
#[derive(Debug)]
struct RetryErrorImpl {
    method: Method,
    uri: Option<Uri>,
    retries: usize,
    max_retries: usize,
    elapsed: Duration,
    retry_timeout: Duration,
    inner: RequestError,
}

impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error performing {} ", self.0.method)?;
        match &self.0.uri {
            Some(uri) => write!(f, "{uri} ")?,
            None => write!(f, "REDACTED ")?,
        }
        write!(f, "in {:?}", self.0.elapsed)?;
        if self.0.retries != 0 {
            write!(
                f,
                ", after {} retries, max_retries: {}, retry_timeout: {:?} ",
                self.0.retries, self.0.max_retries, self.0.retry_timeout
            )?;
        }
        write!(f, " - {}", self.0.inner)
    }
}

impl std::error::Error for RetryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0.inner)
    }
}

/// Context of the retry loop
///
/// Most use-cases should use [`RetryExt`] and [`RetryableRequestBuilder`], however,
/// [`RetryContext`] allows preserving retry state across multiple [`RetryableRequest`]
pub(crate) struct RetryContext {
    backoff: Backoff,
    retries: usize,
    max_retries: usize,
    retry_timeout: Duration,
    start: Instant,
}

impl RetryContext {
    pub(crate) fn new(config: &RetryConfig) -> Self {
        Self {
            max_retries: config.max_retries,
            retry_timeout: config.retry_timeout,
            backoff: Backoff::new(&config.backoff),
            retries: 0,
            start: Instant::now(),
        }
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.retries >= self.max_retries || self.start.elapsed() > self.retry_timeout
    }

    pub(crate) fn backoff(&mut self) -> Duration {
        self.retries += 1;
        self.backoff.next()
    }
}

/// The reason a request failed
#[derive(Debug, thiserror::Error)]
pub enum RequestError {
    #[error("Received redirect without LOCATION, this normally indicates an incorrectly configured region"
    )]
    BareRedirect,

    #[error("Server returned non-2xx status code: {status}: {}", body.as_deref().unwrap_or(""))]
    Status {
        status: StatusCode,
        body: Option<String>,
    },

    #[error("Server returned error response: {body}")]
    Response { status: StatusCode, body: String },

    #[error(transparent)]
    Http(#[from] HttpError),
}

impl RetryError {
    /// Returns the underlying [`RequestError`]
    pub fn inner(&self) -> &RequestError {
        &self.0.inner
    }

    /// Returns the status code associated with this error if any
    pub fn status(&self) -> Option<StatusCode> {
        match self.inner() {
            RequestError::Status { status, .. } | RequestError::Response { status, .. } => {
                Some(*status)
            }
            RequestError::BareRedirect | RequestError::Http(_) => None,
        }
    }

    /// Returns the error body if any
    pub fn body(&self) -> Option<&str> {
        match self.inner() {
            RequestError::Status { body, .. } => body.as_deref(),
            RequestError::Response { body, .. } => Some(body),
            RequestError::BareRedirect | RequestError::Http(_) => None,
        }
    }

    pub fn error(self, store: &'static str, path: String) -> crate::Error {
        match self.status() {
            Some(StatusCode::NOT_FOUND) => crate::Error::NotFound {
                path,
                source: Box::new(self),
            },
            Some(StatusCode::NOT_MODIFIED) => crate::Error::NotModified {
                path,
                source: Box::new(self),
            },
            Some(StatusCode::PRECONDITION_FAILED) => crate::Error::Precondition {
                path,
                source: Box::new(self),
            },
            Some(StatusCode::CONFLICT) => crate::Error::AlreadyExists {
                path,
                source: Box::new(self),
            },
            Some(StatusCode::FORBIDDEN) => crate::Error::PermissionDenied {
                path,
                source: Box::new(self),
            },
            Some(StatusCode::UNAUTHORIZED) => crate::Error::Unauthenticated {
                path,
                source: Box::new(self),
            },
            _ => crate::Error::Generic {
                store,
                source: Box::new(self),
            },
        }
    }
}

impl From<RetryError> for std::io::Error {
    fn from(err: RetryError) -> Self {
        use std::io::ErrorKind;
        let kind = match err.status() {
            Some(StatusCode::NOT_FOUND) => ErrorKind::NotFound,
            Some(StatusCode::BAD_REQUEST) => ErrorKind::InvalidInput,
            Some(StatusCode::UNAUTHORIZED) | Some(StatusCode::FORBIDDEN) => {
                ErrorKind::PermissionDenied
            }
            _ => match err.inner() {
                RequestError::Http(h) => match h.kind() {
                    HttpErrorKind::Timeout => ErrorKind::TimedOut,
                    HttpErrorKind::Connect => ErrorKind::NotConnected,
                    _ => ErrorKind::Other,
                },
                _ => ErrorKind::Other,
            },
        };
        Self::new(kind, err)
    }
}

pub(crate) type Result<T, E = RetryError> = std::result::Result<T, E>;

/// The configuration for how to respond to request errors
///
/// The following categories of error will be retried:
///
/// * 5xx server errors
/// * Connection errors
/// * Dropped connections
/// * Timeouts for [safe] / read-only requests
///
/// Requests will be retried up to some limit, using exponential
/// backoff with jitter. See [`BackoffConfig`] for more information
///
/// [safe]: https://datatracker.ietf.org/doc/html/rfc7231#section-4.2.1
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// The backoff configuration
    pub backoff: BackoffConfig,

    /// The maximum number of times to retry a request
    ///
    /// Set to 0 to disable retries
    pub max_retries: usize,

    /// The maximum length of time from the initial request
    /// after which no further retries will be attempted
    ///
    /// This not only bounds the length of time before a server
    /// error will be surfaced to the application, but also bounds
    /// the length of time a request's credentials must remain valid.
    ///
    /// As requests are retried without renewing credentials or
    /// regenerating request payloads, this number should be kept
    /// below 5 minutes to avoid errors due to expired credentials
    /// and/or request payloads
    pub retry_timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            backoff: Default::default(),
            max_retries: 10,
            retry_timeout: Duration::from_secs(3 * 60),
        }
    }
}

fn body_contains_error(response_body: &str) -> bool {
    response_body.contains("InternalError") || response_body.contains("SlowDown")
}

/// Combines a [`RetryableRequest`] with a [`RetryContext`]
pub(crate) struct RetryableRequestBuilder {
    request: RetryableRequest,
    context: RetryContext,
}

impl RetryableRequestBuilder {
    /// Set whether this request is idempotent
    ///
    /// An idempotent request will be retried on timeout even if the request
    /// method is not [safe](https://datatracker.ietf.org/doc/html/rfc7231#section-4.2.1)
    pub(crate) fn idempotent(mut self, idempotent: bool) -> Self {
        self.request.idempotent = Some(idempotent);
        self
    }

    /// Set whether this request should be retried on a 409 Conflict response.
    #[cfg(feature = "aws")]
    pub(crate) fn retry_on_conflict(mut self, retry_on_conflict: bool) -> Self {
        self.request.retry_on_conflict = retry_on_conflict;
        self
    }

    /// Set whether this request contains sensitive data
    ///
    /// This will avoid printing out the URL in error messages
    #[allow(unused)]
    pub(crate) fn sensitive(mut self, sensitive: bool) -> Self {
        self.request.sensitive = sensitive;
        self
    }

    /// Provide a [`PutPayload`]
    pub(crate) fn payload(mut self, payload: Option<PutPayload>) -> Self {
        self.request.payload = payload;
        self
    }

    #[allow(unused)]
    pub(crate) fn retry_error_body(mut self, retry_error_body: bool) -> Self {
        self.request.retry_error_body = retry_error_body;
        self
    }

    pub(crate) async fn send(mut self) -> Result<HttpResponse> {
        self.request.send(&mut self.context).await
    }
}

/// A retryable request
pub(crate) struct RetryableRequest {
    client: HttpClient,
    http: HttpRequest,

    sensitive: bool,
    idempotent: Option<bool>,
    retry_on_conflict: bool,
    payload: Option<PutPayload>,

    retry_error_body: bool,
}

impl RetryableRequest {
    #[allow(unused)]
    pub(crate) fn sensitive(self, sensitive: bool) -> Self {
        Self { sensitive, ..self }
    }

    fn err(&self, error: RequestError, ctx: &RetryContext) -> RetryError {
        RetryError(Box::new(RetryErrorImpl {
            uri: (!self.sensitive).then(|| self.http.uri().clone()),
            method: self.http.method().clone(),
            retries: ctx.retries,
            max_retries: ctx.max_retries,
            elapsed: ctx.start.elapsed(),
            retry_timeout: ctx.retry_timeout,
            inner: error,
        }))
    }

    pub(crate) async fn send(self, ctx: &mut RetryContext) -> Result<HttpResponse> {
        loop {
            let mut request = self.http.clone();

            if let Some(payload) = &self.payload {
                *request.body_mut() = payload.clone().into();
            }

            match self.client.execute(request).await {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        // For certain S3 requests, 200 response may contain `InternalError` or
                        // `SlowDown` in the message. These responses should be handled similarly
                        // to r5xx errors.
                        // More info here: https://repost.aws/knowledge-center/s3-resolve-200-internalerror
                        if !self.retry_error_body {
                            return Ok(r);
                        }

                        let (parts, body) = r.into_parts();
                        let body = match body.text().await {
                            Ok(body) => body,
                            Err(e) => return Err(self.err(RequestError::Http(e), ctx)),
                        };

                        if !body_contains_error(&body) {
                            // Success response and no error, clone and return response
                            return Ok(HttpResponse::from_parts(parts, body.into()));
                        } else {
                            // Retry as if this was a 5xx response
                            if ctx.exhausted() {
                                return Err(self.err(RequestError::Response { body, status }, ctx));
                            }

                            let sleep = ctx.backoff();
                            info!(
                                "Encountered a response status of {} but body contains Error, backing off for {} seconds, retry {} of {}",
                                status,
                                sleep.as_secs_f32(),
                                ctx.retries,
                                ctx.max_retries,
                            );
                            tokio::time::sleep(sleep).await;
                        }
                    } else if status == StatusCode::NOT_MODIFIED {
                        return Err(self.err(RequestError::Status { status, body: None }, ctx));
                    } else if status.is_redirection() {
                        let is_bare_redirect = !r.headers().contains_key(LOCATION);
                        return match is_bare_redirect {
                            true => Err(self.err(RequestError::BareRedirect, ctx)),
                            false => Err(self.err(
                                RequestError::Status {
                                    body: None,
                                    status: r.status(),
                                },
                                ctx,
                            )),
                        };
                    } else {
                        let status = r.status();
                        if ctx.exhausted()
                            || !(status.is_server_error()
                                || status == StatusCode::TOO_MANY_REQUESTS
                                || status == StatusCode::REQUEST_TIMEOUT
                                || (self.retry_on_conflict && status == StatusCode::CONFLICT))
                        {
                            let source = match status.is_client_error() {
                                true => match r.into_body().text().await {
                                    Ok(body) => RequestError::Status {
                                        status,
                                        body: Some(body),
                                    },
                                    Err(e) => RequestError::Http(e),
                                },
                                false => RequestError::Status { status, body: None },
                            };
                            return Err(self.err(source, ctx));
                        };

                        let sleep = ctx.backoff();
                        info!(
                            "Encountered server error with status {}, backing off for {} seconds, retry {} of {}",
                            status,
                            sleep.as_secs_f32(),
                            ctx.retries,
                            ctx.max_retries,
                        );
                        tokio::time::sleep(sleep).await;
                    }
                }
                Err(e) => {
                    let is_idempotent = self
                        .idempotent
                        .unwrap_or_else(|| self.http.method().is_safe());

                    let do_retry = match e.kind() {
                        HttpErrorKind::Connect | HttpErrorKind::Request => true, // Request not sent, can retry
                        HttpErrorKind::Timeout | HttpErrorKind::Interrupted => is_idempotent,
                        HttpErrorKind::Unknown | HttpErrorKind::Decode => false,
                    };

                    if ctx.exhausted() || !do_retry {
                        return Err(self.err(RequestError::Http(e), ctx));
                    }
                    let sleep = ctx.backoff();
                    info!(
                        "Encountered transport error of kind {:?}, backing off for {} seconds, retry {} of {}: {}",
                        e.kind(),
                        sleep.as_secs_f32(),
                        ctx.retries,
                        ctx.max_retries,
                        e,
                    );
                    tokio::time::sleep(sleep).await;
                }
            }
        }
    }
}

pub(crate) trait RetryExt {
    /// Return a [`RetryableRequestBuilder`]
    fn retryable(self, config: &RetryConfig) -> RetryableRequestBuilder;

    /// Return a [`RetryableRequest`]
    fn retryable_request(self) -> RetryableRequest;

    /// Dispatch a request with the given retry configuration
    ///
    /// # Panic
    ///
    /// This will panic if the request body is a stream
    fn send_retry(self, config: &RetryConfig) -> BoxFuture<'static, Result<HttpResponse>>;
}

impl RetryExt for HttpRequestBuilder {
    fn retryable(self, config: &RetryConfig) -> RetryableRequestBuilder {
        RetryableRequestBuilder {
            request: self.retryable_request(),
            context: RetryContext::new(config),
        }
    }

    fn retryable_request(self) -> RetryableRequest {
        let (client, request) = self.into_parts();
        let request = request.expect("request must be valid");

        RetryableRequest {
            client,
            http: request,
            idempotent: None,
            payload: None,
            sensitive: false,
            retry_on_conflict: false,
            retry_error_body: false,
        }
    }

    fn send_retry(self, config: &RetryConfig) -> BoxFuture<'static, Result<HttpResponse>> {
        let request = self.retryable(config);
        Box::pin(async move { request.send().await })
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {
    use crate::client::mock_server::MockServer;
    use crate::client::retry::{body_contains_error, RequestError, RetryContext, RetryExt};
    use crate::client::{HttpClient, HttpResponse};
    use crate::RetryConfig;
    use http::StatusCode;
    use hyper::header::LOCATION;
    use hyper::server::conn::http1;
    use hyper::service::service_fn;
    use hyper::Response;
    use hyper_util::rt::TokioIo;
    use reqwest::{Client, Method};
    use std::convert::Infallible;
    use std::error::Error;
    use std::time::Duration;
    use tokio::net::TcpListener;
    use tokio::time::timeout;

    #[test]
    fn test_body_contains_error() {
        // Example error message provided by https://repost.aws/knowledge-center/s3-resolve-200-internalerror
        let error_response = "AmazonS3Exception: We encountered an internal error. Please try again. (Service: Amazon S3; Status Code: 200; Error Code: InternalError; Request ID: 0EXAMPLE9AAEB265)";
        assert!(body_contains_error(error_response));

        let error_response_2 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message><RequestId>123</RequestId><HostId>456</HostId></Error>";
        assert!(body_contains_error(error_response_2));

        // Example success response from https://docs.aws.amazon.com/AmazonS3/latest/API/API_CopyObject.html
        let success_response = "<CopyObjectResult><LastModified>2009-10-12T17:50:30.000Z</LastModified><ETag>\"9b2cf535f27731c974343645a3985328\"</ETag></CopyObjectResult>";
        assert!(!body_contains_error(success_response));
    }

    #[tokio::test]
    async fn test_retry() {
        let mock = MockServer::new().await;

        let retry = RetryConfig {
            backoff: Default::default(),
            max_retries: 2,
            retry_timeout: Duration::from_secs(1000),
        };

        let client = HttpClient::new(
            Client::builder()
                .timeout(Duration::from_millis(100))
                .build()
                .unwrap(),
        );

        let do_request = || client.request(Method::GET, mock.url()).send_retry(&retry);

        // Simple request should work
        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Returns client errors immediately with a status message
        mock.push(
            Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body("cupcakes".to_string())
                .unwrap(),
        );

        let e = do_request().await.unwrap_err();
        assert_eq!(e.status().unwrap(), StatusCode::BAD_REQUEST);
        assert_eq!(e.body(), Some("cupcakes"));
        assert_eq!(
            e.inner().to_string(),
            "Server returned non-2xx status code: 400 Bad Request: cupcakes"
        );

        // Handles client errors with no payload
        mock.push(
            Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body("NAUGHTY NAUGHTY".to_string())
                .unwrap(),
        );

        let e = do_request().await.unwrap_err();
        assert_eq!(e.status().unwrap(), StatusCode::BAD_REQUEST);
        assert_eq!(e.body(), Some("NAUGHTY NAUGHTY"));
        assert_eq!(
            e.inner().to_string(),
            "Server returned non-2xx status code: 400 Bad Request: NAUGHTY NAUGHTY"
        );

        // Should retry server error request
        mock.push(
            Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(String::new())
                .unwrap(),
        );

        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Should retry 429 Too Many Requests
        mock.push(
            Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .body(String::new())
                .unwrap(),
        );

        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Should retry 408 Request Timeout
        mock.push(
            Response::builder()
                .status(StatusCode::REQUEST_TIMEOUT)
                .body(String::new())
                .unwrap(),
        );

        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Accepts 204 status code
        mock.push(
            Response::builder()
                .status(StatusCode::NO_CONTENT)
                .body(String::new())
                .unwrap(),
        );

        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::NO_CONTENT);

        // Follows 402 redirects
        mock.push(
            Response::builder()
                .status(StatusCode::FOUND)
                .header(LOCATION, "/foo")
                .body(String::new())
                .unwrap(),
        );

        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Follows 401 redirects
        mock.push(
            Response::builder()
                .status(StatusCode::FOUND)
                .header(LOCATION, "/bar")
                .body(String::new())
                .unwrap(),
        );

        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Handles redirect loop
        for _ in 0..11 {
            mock.push(
                Response::builder()
                    .status(StatusCode::FOUND)
                    .header(LOCATION, "/bar")
                    .body(String::new())
                    .unwrap(),
            );
        }

        let e = do_request().await.unwrap_err().to_string();
        assert!(e.contains("error following redirect"), "{}", e);

        // Handles redirect missing location
        mock.push(
            Response::builder()
                .status(StatusCode::FOUND)
                .body(String::new())
                .unwrap(),
        );

        let e = do_request().await.unwrap_err();
        assert!(matches!(e.inner(), RequestError::BareRedirect));
        assert_eq!(e.inner().to_string(), "Received redirect without LOCATION, this normally indicates an incorrectly configured region");

        // Gives up after the retrying the specified number of times
        for _ in 0..=retry.max_retries {
            mock.push(
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .body("ignored".to_string())
                    .unwrap(),
            );
        }

        let e = do_request().await.unwrap_err();
        assert!(
            e.to_string().contains(" after 2 retries, max_retries: 2, retry_timeout: 1000s  - Server returned non-2xx status code: 502 Bad Gateway"),
            "{e}"
        );
        // verify e.source() is available as well for users who need programmatic access
        assert_eq!(
            e.source().unwrap().to_string(),
            "Server returned non-2xx status code: 502 Bad Gateway: ",
        );

        // Panic results in an incomplete message error in the client
        mock.push_fn::<_, String>(|_| panic!());
        let r = do_request().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        // Gives up after retrying multiple panics
        for _ in 0..=retry.max_retries {
            mock.push_fn::<_, String>(|_| panic!());
        }
        let e = do_request().await.unwrap_err();
        assert!(
            e.to_string().contains("after 2 retries, max_retries: 2, retry_timeout: 1000s  - HTTP error: error sending request"),
            "{e}"
        );
        // verify e.source() is available as well for users who need programmatic access
        assert_eq!(
            e.source().unwrap().to_string(),
            "HTTP error: error sending request",
        );

        // Retries on client timeout
        mock.push_async_fn(|_| async move {
            tokio::time::sleep(Duration::from_secs(10)).await;
            panic!()
        });
        do_request().await.unwrap();

        // Does not retry PUT request
        mock.push_async_fn(|_| async move {
            tokio::time::sleep(Duration::from_secs(10)).await;
            panic!()
        });
        let res = client.request(Method::PUT, mock.url()).send_retry(&retry);
        let e = res.await.unwrap_err().to_string();
        assert!(
            !e.contains("retries") && e.contains("error sending request"),
            "{e}"
        );

        let url = format!("{}/SENSITIVE", mock.url());
        for _ in 0..=retry.max_retries {
            mock.push(
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .body("ignored".to_string())
                    .unwrap(),
            );
        }
        let res = client.request(Method::GET, url).send_retry(&retry).await;
        let err = res.unwrap_err().to_string();
        assert!(err.contains("SENSITIVE"), "{err}");

        let url = format!("{}/SENSITIVE", mock.url());
        for _ in 0..=retry.max_retries {
            mock.push(
                Response::builder()
                    .status(StatusCode::BAD_GATEWAY)
                    .body("ignored".to_string())
                    .unwrap(),
            );
        }

        // Sensitive requests should strip URL from error
        let req = client
            .request(Method::GET, &url)
            .retryable(&retry)
            .sensitive(true);
        let err = req.send().await.unwrap_err().to_string();
        assert!(!err.contains("SENSITIVE"), "{err}");

        for _ in 0..=retry.max_retries {
            mock.push_fn::<_, String>(|_| panic!());
        }

        let req = client
            .request(Method::GET, &url)
            .retryable(&retry)
            .sensitive(true);
        let err = req.send().await.unwrap_err().to_string();
        assert!(!err.contains("SENSITIVE"), "{err}");

        // Success response with error in body is retried
        mock.push(
            Response::builder()
                .status(StatusCode::OK)
                .body("InternalError".to_string())
                .unwrap(),
        );
        let req = client
            .request(Method::PUT, &url)
            .retryable(&retry)
            .idempotent(true)
            .retry_error_body(true);
        let r = req.send().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        // Response with InternalError should have been retried
        let b = r.into_body().text().await.unwrap();
        assert!(!b.contains("InternalError"));

        // Should not retry success response with no error in body
        mock.push(
            Response::builder()
                .status(StatusCode::OK)
                .body("success".to_string())
                .unwrap(),
        );
        let req = client
            .request(Method::PUT, &url)
            .retryable(&retry)
            .idempotent(true)
            .retry_error_body(true);
        let r = req.send().await.unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        let b = r.into_body().text().await.unwrap();
        assert!(b.contains("success"));

        // Shutdown
        mock.shutdown().await
    }

    #[tokio::test]
    async fn test_connection_reset_is_retried() {
        let retry = RetryConfig {
            backoff: Default::default(),
            max_retries: 2,
            retry_timeout: Duration::from_secs(1),
        };
        assert!(retry.max_retries > 0);

        // Setup server which resets a connection and then quits
        let listener = TcpListener::bind("::1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            // Reset the connection on the first n-1 attempts
            for _ in 0..retry.max_retries {
                let (stream, _) = listener.accept().await.unwrap();
                stream.set_linger(Some(Duration::from_secs(0))).unwrap();
            }
            // Succeed on the last attempt
            let (stream, _) = listener.accept().await.unwrap();
            http1::Builder::new()
                // we want the connection to end after responding
                .keep_alive(false)
                .serve_connection(
                    TokioIo::new(stream),
                    service_fn(move |_req| async {
                        Ok::<_, Infallible>(HttpResponse::new("Success!".to_string().into()))
                    }),
                )
                .await
                .unwrap();
        });

        // Perform the request
        let client = HttpClient::new(reqwest::Client::new());
        let ctx = &mut RetryContext::new(&retry);
        let res = client
            .get(url)
            .retryable_request()
            .send(ctx)
            .await
            .expect("request should eventually succeed");
        assert_eq!(res.status(), StatusCode::OK);
        assert!(ctx.exhausted());

        // Wait for server to shutdown
        let _ = timeout(Duration::from_secs(1), handle)
            .await
            .expect("shutdown shouldn't hang");
    }
}
