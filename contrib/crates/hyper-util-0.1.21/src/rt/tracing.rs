//! Runtime components for use with [`tracing`].
//!
//! This module provides [`Executor`] implementations that configure
//! instrumentation of spawned futures. These [`Executor`]s can propagate
//! tracing [`Span`]s to futures spawned onto the async runtime. See the
//! crate-level documentation of [`tracing`] for [more information] about spans.
//!
//! # Choosing an [`Executor`].
//!
//! Hyper spawns [`Future`]s onto an [`Executor`], to avoid tightly coupling
//! APIs to any particular async runtime. This includes background tasks that
//! might help service I/O for the lifetime of a connection, for example.
//!
//! Some [`Subscriber`][tracing::subscriber] implementations have different
//! semantics regarding the lifecycle of [`Span`]s. Integrations with
//! OpenTelemetry collectors, for example, might not emit the events within
//! the context of a span until it is closed. Conversely, subscribers that
//! print traces to the terminal may not have to contend with these details when
//! instrumenting long-lived tasks that run in the background.
//!
//! This module provides different executors to help pass tracing context in
//! the manner appropriate for your application. For most typical applications,
//! [`CurrentSpanExecutor<E>`] should suffice.
//!
//! # Examples
//!
//! Run spawned tasks within a provided span.
//!
//! ```
//! # #[cfg(feature = "tokio")]
//! # {
//! use hyper_util::rt::{TokioExecutor, WithSpanExecutor};
//!
//! let span = tracing::info_span!("example");
//! let executor = WithSpanExecutor::new(TokioExecutor::new(), span);
//! # }
//! ```
//!
//! Run spawned tasks within the current span when [`Executor::execute()`] is
//! called.
//!
//! ```
//! # #[cfg(feature = "tokio")]
//! # {
//! use hyper_util::rt::{TokioExecutor, CurrentSpanExecutor};
//!
//! let executor = CurrentSpanExecutor::new(TokioExecutor::new());
//! # }
//! ```
//!
//! Run spawned tasks within distinct spans that are marked as following from
//! the active span when [`Executor::execute()`] is called.
//!
//! ```
//! # #[cfg(feature = "tokio")]
//! # {
//! use hyper_util::rt::{MkSpanExecutor, TokioExecutor};
//! use tracing::{info_span, Span};
//!
//! let mk = || {
//!     let span = info_span!("example");
//!     span.follows_from(Span::current());
//!     span
//! };
//! let executor = MkSpanExecutor::new(TokioExecutor::new(), mk);
//! # }
//! ```
//!
//! [more information]: tracing#spans-1

use hyper::rt::Executor;
use tracing::{
    Span,
    instrument::{Instrument, Instrumented},
};

/// An executor that propagates the current tracing span to its futures.
///
/// The span is captured when [`execute`](Executor::execute) is called, and is
/// entered each time the future is polled or dropped. Execution is delegated to
/// the wrapped executor, without requiring a particular runtime.
///
/// Requires the `tracing` feature.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "tokio")]
/// # {
/// use hyper_util::rt::{TokioExecutor, CurrentSpanExecutor};
///
/// let executor = CurrentSpanExecutor::new(TokioExecutor::new());
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct CurrentSpanExecutor<E> {
    inner: E,
}

/// An executor that propagates a provided tracing span to its futures.
///
/// The span provided to this executor is entered each time the future is
/// polled or dropped. Execution is delegated to the wrapped executor, without
/// requiring a particular runtime.
///
/// Requires the `tracing` feature.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "tokio")]
/// # {
/// use hyper_util::rt::{TokioExecutor, WithSpanExecutor};
///
/// let span = tracing::info_span!("example");
/// let executor = WithSpanExecutor::new(TokioExecutor::new(), span);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct WithSpanExecutor<E> {
    inner: E,
    span: Span,
}

/// An executor that uses a callback to propagate tracing span to its futures.
///
/// The callback is invoked each time a future is spawned, creating a span that
/// will be entered each time that future is polled or dropped. Execution is
/// delegated to the wrapped executor, without requiring a particular runtime.
///
/// Requires the `tracing` feature.
///
/// # Example
///
/// Spawned tasks can be marked as "following from" the execution context.
///
/// See [`tracing::Span::follows_from()`] for more information about
/// indicating causal relationships between spans.
///
/// ```
/// # #[cfg(feature = "tokio")]
/// # {
/// use hyper_util::rt::{MkSpanExecutor, TokioExecutor};
/// use tracing::{info_span, Span};
///
/// let mk = || {
///     let span = info_span!("example");
///     span.follows_from(Span::current());
///     span
/// };
/// let executor = MkSpanExecutor::new(TokioExecutor::new(), mk);
/// # }
/// ```
///
/// Spawned tasks can be marked as children of the execution context.
///
/// ```
/// # #[cfg(feature = "tokio")]
/// # {
/// use hyper_util::rt::{MkSpanExecutor, TokioExecutor};
/// use tracing::{info_span, Span};
///
/// let mk = || info_span!(parent: Span::current(), "example");
/// let executor = MkSpanExecutor::new(TokioExecutor::new(), mk);
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct MkSpanExecutor<E, F> {
    inner: E,
    mk: F,
}

// ===== impl CurrentSpanExecutor =====

impl<E> CurrentSpanExecutor<E> {
    /// Wrap an executor to propagate the current tracing span to its futures.
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E, F> Executor<F> for CurrentSpanExecutor<E>
where
    E: Executor<Instrumented<F>>,
    F: Future,
{
    fn execute(&self, future: F) {
        self.inner.execute(future.in_current_span());
    }
}

// ===== impl WithSpanExecutor =====

impl<E> WithSpanExecutor<E> {
    /// Wrap an executor to propagate the provided tracing span to its futures.
    pub fn new(inner: E, span: Span) -> Self {
        Self { inner, span }
    }

    /// Wrap an executor to propagate the current tracing span to its futures.
    ///
    /// This will instrument futures with the span that is active at the call-site of _this_
    /// function. Use [`CurrentSpanExecutor<E>`] if you would prefer to propagate the current span
    /// when [`Executor::execute()`] is called, rather than span that is active when initializating
    /// the executor.
    pub fn current(inner: E) -> Self {
        Self {
            inner,
            span: Span::current(),
        }
    }
}

impl<E, F> Executor<F> for WithSpanExecutor<E>
where
    E: Executor<Instrumented<F>>,
    F: Future,
{
    fn execute(&self, future: F) {
        self.inner.execute(future.instrument(self.span.clone()));
    }
}

// ===== impl MkSpanExecutor =====

impl<E, F> MkSpanExecutor<E, F> {
    /// Wrap an executor that creates new spans to instrument spawned futures.
    pub fn new(inner: E, mk: F) -> Self {
        Self { inner, mk }
    }
}

impl<E, F, Fut> Executor<Fut> for MkSpanExecutor<E, F>
where
    E: Executor<Instrumented<Fut>>,
    F: Fn() -> Span,
    Fut: Future,
{
    fn execute(&self, future: Fut) {
        let span = (self.mk)();
        self.inner.execute(future.instrument(span));
    }
}

#[cfg(test)]
mod tests {
    use super::{CurrentSpanExecutor, MkSpanExecutor, WithSpanExecutor};
    use hyper::rt::Executor;
    use std::{
        cell::RefCell,
        future::poll_fn,
        pin::Pin,
        sync::{Arc, Mutex},
        task::Poll,
    };

    #[derive(Default)]
    struct DeferredExecutor<'a> {
        future: RefCell<Option<Pin<Box<dyn Future<Output = ()> + 'a>>>>,
    }

    impl<'a, F: Future<Output = ()> + 'a> Executor<F> for &DeferredExecutor<'a> {
        fn execute(&self, future: F) {
            *self.future.borrow_mut() = Some(Box::pin(future));
        }
    }

    #[test]
    fn current_span_executor_propagates_span_from_execute_on_each_poll() {
        let _subscriber = tracing::subscriber::set_default(tracing_subscriber::registry());
        let construction_span = tracing::info_span!("construction");
        let execution_span = tracing::info_span!("execution");
        let polling_span = tracing::info_span!("polling");
        assert!(execution_span.id().is_some());

        // Borrowing a local executor and future also checks that the wrapper
        // does not impose Send or 'static bounds on the inner executor.
        let polls = RefCell::new(0);
        let inner = DeferredExecutor::default();
        let executor = construction_span.in_scope(|| CurrentSpanExecutor::new(&inner));
        execution_span.in_scope(|| {
            executor.execute(poll_fn(|_| {
                assert_eq!(tracing::Span::current().id(), execution_span.id());
                *polls.borrow_mut() += 1;
                if *polls.borrow() == 1 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }));
        });

        let _entered = polling_span.enter();
        let mut task = tokio_test::task::spawn(inner.future.borrow_mut().take().unwrap());
        assert!(task.poll().is_pending());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert!(task.poll().is_ready());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert_eq!(*polls.borrow(), 2);
    }

    #[test]
    fn with_span_executor_propagates_given_span_on_each_poll() {
        let _subscriber = tracing::subscriber::set_default(tracing_subscriber::registry());
        let construction_span = tracing::info_span!("construction");
        let execution_span = tracing::info_span!("execution");
        let polling_span = tracing::info_span!("polling");
        let with_span = tracing::info_span!("with");
        assert!(execution_span.id().is_some());

        // Borrowing a local executor and future also checks that the wrapper
        // does not impose Send or 'static bounds on the inner executor.
        let polls = RefCell::new(0);
        let inner = DeferredExecutor::default();
        let executor =
            construction_span.in_scope(|| WithSpanExecutor::new(&inner, with_span.clone()));
        execution_span.in_scope(|| {
            executor.execute(poll_fn(|_| {
                // Execution happens within the given span.
                assert_eq!(tracing::Span::current().id(), with_span.id());
                *polls.borrow_mut() += 1;
                if *polls.borrow() == 1 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }));
        });

        let _entered = polling_span.enter();
        let mut task = tokio_test::task::spawn(inner.future.borrow_mut().take().unwrap());
        assert!(task.poll().is_pending());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert!(task.poll().is_ready());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert_eq!(*polls.borrow(), 2);
    }

    #[test]
    fn with_span_executor_current_propagates_construction_span() {
        let _subscriber = tracing::subscriber::set_default(tracing_subscriber::registry());
        let construction_span = tracing::info_span!("construction");
        let execution_span = tracing::info_span!("execution");
        let polling_span = tracing::info_span!("polling");
        assert!(execution_span.id().is_some());

        // Borrowing a local executor and future also checks that the wrapper
        // does not impose Send or 'static bounds on the inner executor.
        let polls = RefCell::new(0);
        let inner = DeferredExecutor::default();
        let executor = construction_span.in_scope(|| WithSpanExecutor::current(&inner));
        execution_span.in_scope(|| {
            executor.execute(poll_fn(|_| {
                // Execution happens within the given span.
                assert_eq!(tracing::Span::current().id(), construction_span.id());
                *polls.borrow_mut() += 1;
                if *polls.borrow() == 1 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }));
        });

        let _entered = polling_span.enter();
        let mut task = tokio_test::task::spawn(inner.future.borrow_mut().take().unwrap());
        assert!(task.poll().is_pending());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert!(task.poll().is_ready());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert_eq!(*polls.borrow(), 2);
    }

    #[test]
    fn mk_span_executor_current_propagates_child_span() {
        let _subscriber = tracing::subscriber::set_default(tracing_subscriber::registry());
        let construction_span = tracing::info_span!("construction");
        let execution_span = tracing::info_span!("execution");
        let polling_span = tracing::info_span!("polling");
        assert!(execution_span.id().is_some());

        // A callback that creates a new child of the given span.
        let mk = || tracing::info_span!(parent: tracing::Span::current(), "child");

        // Borrowing a local executor and future also checks that the wrapper
        // does not impose Send or 'static bounds on the inner executor.
        let polls = RefCell::new(0);
        let inner = DeferredExecutor::default();
        let executor = construction_span.in_scope(|| MkSpanExecutor::new(&inner, mk));
        execution_span.in_scope(|| {
            executor.execute(poll_fn(|_| {
                // Execution happens within the created child span.
                let span = tracing::Span::current();
                assert_eq!(span.metadata().unwrap().name(), "child");
                *polls.borrow_mut() += 1;
                if *polls.borrow() == 1 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }));
        });

        let _entered = polling_span.enter();
        let mut task = tokio_test::task::spawn(inner.future.borrow_mut().take().unwrap());
        assert!(task.poll().is_pending());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert!(task.poll().is_ready());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert_eq!(*polls.borrow(), 2);
    }

    /// A subscriber that records causal `follows_from` relationships.
    struct FollowsFromSubscriber<S> {
        inner: S,
        follows_from: Arc<Mutex<Vec<FollowsFrom>>>,
    }

    /// A tuple representing a causal relationship between two spans.
    ///
    /// This means that the span with the former id followed from the span with the latter id.
    type FollowsFrom = (tracing::span::Id, tracing::span::Id);

    impl<S> FollowsFromSubscriber<S> {
        fn new(inner: S) -> Self {
            Self {
                inner,
                follows_from: Default::default(),
            }
        }

        /// Returns a reference to the set of relationships observed.
        fn follows_from(&self) -> Arc<Mutex<Vec<FollowsFrom>>> {
            Arc::clone(&self.follows_from)
        }
    }

    impl<S> tracing::Subscriber for FollowsFromSubscriber<S>
    where
        S: tracing::Subscriber,
    {
        fn record_follows_from(&self, span: &tracing::span::Id, follows: &tracing::span::Id) {
            self.follows_from
                .lock()
                .unwrap()
                .push((span.clone(), follows.clone()));
            self.inner.record_follows_from(span, follows);
        }

        fn current_span(&self) -> tracing_core::span::Current {
            self.inner.current_span()
        }

        // Other methods delegate to `inner`...

        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            self.inner.enabled(metadata)
        }

        fn enter(&self, span: &tracing::span::Id) {
            self.inner.enter(span);
        }

        fn event(&self, event: &tracing::Event<'_>) {
            self.inner.event(event);
        }

        fn exit(&self, span: &tracing::span::Id) {
            self.inner.exit(span);
        }

        fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            self.inner.new_span(span)
        }

        fn record(&self, span: &tracing::span::Id, values: &tracing::span::Record<'_>) {
            self.inner.record(span, values);
        }
    }

    #[test]
    fn mk_span_executor_current_propagates_causal_span_relationships() {
        // Use a subscriber that records `follows_from` relationships.
        let subscriber = FollowsFromSubscriber::new(tracing_subscriber::registry());
        let relationships = subscriber.follows_from();
        let _subscriber = tracing::subscriber::set_default(subscriber);

        let construction_span = tracing::info_span!("construction");
        let execution_a_span = tracing::info_span!("execution_a");
        let execution_b_span = tracing::info_span!("execution_b");
        let polling_span = tracing::info_span!("polling");

        // A callback that creates a span that `follows_from` the execution span.
        let mk = || {
            let span = tracing::info_span!("spawned");
            span.follows_from(tracing::Span::current());
            span
        };

        let polls = RefCell::new(0);
        let inner = DeferredExecutor::default();
        let executor = construction_span.in_scope(|| MkSpanExecutor::new(&inner, mk));
        execution_a_span.in_scope(|| {
            executor.execute(poll_fn(|_| {
                // Execution happens within the created child span.
                let span = tracing::Span::current();
                assert_eq!(span.metadata().unwrap().name(), "spawned");
                *polls.borrow_mut() += 1;
                if *polls.borrow() == 1 {
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }));
        });

        let _entered = polling_span.enter();
        let mut task = tokio_test::task::spawn(inner.future.borrow_mut().take().unwrap());
        assert!(task.poll().is_pending());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert_eq!(relationships.lock().unwrap().len(), 1);
        assert!(task.poll().is_ready());
        assert_eq!(tracing::Span::current().id(), polling_span.id());
        assert_eq!(*polls.borrow(), 2);
        assert_eq!(relationships.lock().unwrap().len(), 1);

        execution_b_span.in_scope(|| {
            executor.execute(poll_fn(|_| {
                let span = tracing::Span::current();
                assert_eq!(span.metadata().unwrap().name(), "spawned");
                Poll::Ready(())
            }));
        });

        let _entered = polling_span.enter();
        let mut task = tokio_test::task::spawn(inner.future.borrow_mut().take().unwrap());
        assert!(task.poll().is_ready());

        // The first task followed from the `execution_a` span. The second task
        // followed from the `execution_b` span.
        let relationships = relationships.lock().unwrap();
        assert_eq!(relationships.len(), 2);
        assert_eq!(relationships[0].1, execution_a_span.id().unwrap());
        assert_eq!(relationships[1].1, execution_b_span.id().unwrap());
    }
}
