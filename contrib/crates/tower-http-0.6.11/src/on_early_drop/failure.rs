//! Failure classifications produced by [`EarlyDropsAsFailures`].
//!
//! [`EarlyDropsAsFailures`]: super::EarlyDropsAsFailures

use http::StatusCode;
use std::fmt;

/// Classification for early-drop events reported through
/// [`EarlyDropsAsFailures`].
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use tower_http::on_early_drop::DroppedFailure;
/// use tower_http::trace::OnFailure;
/// use tracing::Span;
///
/// #[derive(Clone)]
/// struct MyOnFailure;
///
/// impl OnFailure<DroppedFailure> for MyOnFailure {
///     fn on_failure(&mut self, class: DroppedFailure, latency: Duration, _span: &Span) {
///         match class {
///             DroppedFailure::Future(_) => {
///                 tracing::warn!(?latency, "future dropped")
///             }
///             DroppedFailure::Body(body) => {
///                 tracing::warn!(?latency, status = %body.status, "body dropped")
///             }
///             _ => {}
///         }
///     }
/// }
/// ```
///
/// [`EarlyDropsAsFailures`]: super::EarlyDropsAsFailures
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DroppedFailure {
    /// The response future was dropped before producing any response.
    Future(FutureDropped),
    /// The response body was dropped before reaching end-of-stream.
    Body(BodyDropped),
}

/// Context for [`DroppedFailure::Future`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FutureDropped;

/// Context for [`DroppedFailure::Body`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BodyDropped {
    /// Status of the already-emitted response.
    pub status: StatusCode,
}

impl fmt::Display for DroppedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DroppedFailure::Future(_) => f.write_str("response future dropped before completion"),
            DroppedFailure::Body(body) => {
                write!(
                    f,
                    "response body dropped before end-of-stream (status: {})",
                    body.status
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn future_display() {
        assert_eq!(
            DroppedFailure::Future(FutureDropped).to_string(),
            "response future dropped before completion",
        );
    }

    #[test]
    fn body_display_carries_status() {
        assert_eq!(
            DroppedFailure::Body(BodyDropped {
                status: StatusCode::INTERNAL_SERVER_ERROR
            })
            .to_string(),
            "response body dropped before end-of-stream (status: 500 Internal Server Error)",
        );
    }
}
