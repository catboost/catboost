//! Middleware for setting headers on requests and responses.
//!
//! See [request] and [response] for more details.

use std::fmt;

use http::{header::HeaderName, HeaderMap, HeaderValue, Request, Response};

pub mod request;
pub mod response;

#[doc(inline)]
pub use self::{
    request::{SetRequestHeader, SetRequestHeaderLayer},
    response::{SetResponseHeader, SetResponseHeaderLayer},
};

/// Trait for producing header values.
///
/// Used by [`SetRequestHeader`] and [`SetResponseHeader`].
///
/// This trait is implemented for closures with the correct type signature. Typically users will
/// not have to implement this trait for their own types.
///
/// It is also implemented directly for [`HeaderValue`]. When a fixed header value should be added
/// to all responses, it can be supplied directly to the middleware.
pub trait MakeHeaderValue<T> {
    /// Try to create a header value from the request or response.
    fn make_header_value(&mut self, message: &T) -> Option<HeaderValue>;
}

impl<F, T> MakeHeaderValue<T> for F
where
    F: FnMut(&T) -> Option<HeaderValue>,
{
    fn make_header_value(&mut self, message: &T) -> Option<HeaderValue> {
        self(message)
    }
}

impl<T> MakeHeaderValue<T> for HeaderValue {
    fn make_header_value(&mut self, _message: &T) -> Option<HeaderValue> {
        Some(self.clone())
    }
}

impl<T> MakeHeaderValue<T> for Option<HeaderValue> {
    fn make_header_value(&mut self, _message: &T) -> Option<HeaderValue> {
        self.clone()
    }
}

#[derive(Debug, Clone, Copy)]
enum InsertHeaderMode {
    Override,
    Append,
    IfNotPresent,
}

impl InsertHeaderMode {
    fn apply<T, M>(self, header_name: &HeaderName, target: &mut T, make: &mut M)
    where
        T: Headers,
        M: MakeHeaderValue<T>,
    {
        match self {
            InsertHeaderMode::Override => {
                if let Some(value) = make.make_header_value(target) {
                    target.headers_mut().insert(header_name.clone(), value);
                }
            }
            InsertHeaderMode::IfNotPresent => {
                if !target.headers().contains_key(header_name) {
                    if let Some(value) = make.make_header_value(target) {
                        target.headers_mut().insert(header_name.clone(), value);
                    }
                }
            }
            InsertHeaderMode::Append => {
                if let Some(value) = make.make_header_value(target) {
                    target.headers_mut().append(header_name.clone(), value);
                }
            }
        }
    }
}

trait Headers {
    fn headers(&self) -> &HeaderMap;

    fn headers_mut(&mut self) -> &mut HeaderMap;
}

impl<B> Headers for Request<B> {
    fn headers(&self) -> &HeaderMap {
        Request::headers(self)
    }

    fn headers_mut(&mut self) -> &mut HeaderMap {
        Request::headers_mut(self)
    }
}

impl<B> Headers for Response<B> {
    fn headers(&self) -> &HeaderMap {
        Response::headers(self)
    }

    fn headers_mut(&mut self) -> &mut HeaderMap {
        Response::headers_mut(self)
    }
}

/// A trait that combines MakeHeaderValue and Clone capability for trait objects.
trait CloneableMakeHeaderValue<T>: MakeHeaderValue<T> + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneableMakeHeaderValue<T>>;
}

impl<T, M> CloneableMakeHeaderValue<T> for M
where
    M: MakeHeaderValue<T> + Clone + Send + Sync + 'static,
{
    fn clone_box(&self) -> Box<dyn CloneableMakeHeaderValue<T>> {
        Box::new(self.clone())
    }
}

/// A "Bridge" struct that allows for trait object-based header value generation.
struct BoxedMakeHeaderValue<T>(Box<dyn CloneableMakeHeaderValue<T>>);

impl<T> BoxedMakeHeaderValue<T> {
    /// Create a new BoxedMakeHeaderValue from any maker that implements MakeHeaderValue and Clone.
    fn new<M>(maker: M) -> Self
    where
        M: MakeHeaderValue<T> + Clone + Send + Sync + 'static,
    {
        Self(Box::new(maker))
    }
}

impl<T> Clone for BoxedMakeHeaderValue<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone_box())
    }
}

impl<T> MakeHeaderValue<T> for BoxedMakeHeaderValue<T> {
    fn make_header_value(&mut self, message: &T) -> Option<HeaderValue> {
        self.0.make_header_value(message)
    }
}

impl<T> fmt::Debug for BoxedMakeHeaderValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxedMakeHeaderValue").finish()
    }
}

/// Metadata describing a request or response header to be set.
#[derive(Clone, Debug)]
pub struct HeaderMetadata<T> {
    /// The name of the header to set.
    header_name: HeaderName,
    /// The value or value factory for the header.
    make: BoxedMakeHeaderValue<T>,
}

impl<T> HeaderMetadata<T> {
    /// Create a new HeaderMetadata with the given header name and value factory.
    fn new<M: MakeHeaderValue<T> + Clone + 'static + Send + Sync>(
        header_name: HeaderName,
        make: M,
    ) -> Self {
        Self {
            header_name,
            make: BoxedMakeHeaderValue::new(make),
        }
    }

    /// Convert this metadata into a [`HeaderInsertionConfig`] with the given insertion mode.
    fn build_config(self, mode: InsertHeaderMode) -> HeaderInsertionConfig<T> {
        HeaderInsertionConfig {
            header_name: self.header_name,
            make: self.make,
            mode,
        }
    }
}

impl<T, M> From<(HeaderName, M)> for HeaderMetadata<T>
where
    M: MakeHeaderValue<T> + Clone + 'static + Send + Sync,
{
    fn from((header_name, make): (HeaderName, M)) -> Self {
        HeaderMetadata::new(header_name, make)
    }
}

/// Configuration for inserting a header into a response or request.
struct HeaderInsertionConfig<T> {
    header_name: HeaderName,
    make: BoxedMakeHeaderValue<T>,
    mode: InsertHeaderMode,
}

impl<T> Clone for HeaderInsertionConfig<T>
where
    BoxedMakeHeaderValue<T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            header_name: self.header_name.clone(),
            make: self.make.clone(),
            mode: self.mode,
        }
    }
}

impl<T> fmt::Debug for HeaderInsertionConfig<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeaderInsertionConfig")
            .field("header_name", &self.header_name)
            .field("mode", &self.mode)
            .field("make", &"BoxedMakeHeaderValue")
            .finish()
    }
}
