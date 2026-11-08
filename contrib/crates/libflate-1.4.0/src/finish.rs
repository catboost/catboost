//! `Finish` and related types.

#[cfg(feature = "no_std")]
use core::ops::{Deref, DerefMut};
#[cfg(feature = "no_std")]
use core2::io::{self, Write};
#[cfg(not(feature = "no_std"))]
use std::{
    io::{self, Write},
    ops::{Deref, DerefMut},
};

/// `Finish` is a type that represents a value which
/// may have an error occurred during the computation.
///
/// Logically, `Finish<T, E>` is equivalent to `Result<T, (T, E)>`.
#[derive(Debug, Default, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Finish<T, E> {
    value: T,
    error: Option<E>,
}
impl<T, E> Finish<T, E> {
    /// Makes a new instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::Finish;
    ///
    /// // The result value of a succeeded computation
    /// let succeeded = Finish::new("value", None as Option<()>);
    /// assert_eq!(succeeded.into_result(), Ok("value"));
    ///
    /// // The result value of a failed computation
    /// let failed = Finish::new("value", Some("error"));
    /// assert_eq!(failed.into_result(), Err("error"));
    /// ```
    pub fn new(value: T, error: Option<E>) -> Self {
        Finish { value, error }
    }

    /// Unwraps the instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::Finish;
    ///
    /// let succeeded = Finish::new("value", None as Option<()>);
    /// assert_eq!(succeeded.unwrap(), ("value", None));
    ///
    /// let failed = Finish::new("value", Some("error"));
    /// assert_eq!(failed.unwrap(), ("value", Some("error")));
    /// ```
    pub fn unwrap(self) -> (T, Option<E>) {
        (self.value, self.error)
    }

    /// Converts from `Finish<T, E>` to `Result<T, E>`.
    ///
    /// # Examples
    /// ```
    /// use libflate::Finish;
    ///
    /// let succeeded = Finish::new("value", None as Option<()>);
    /// assert_eq!(succeeded.into_result(), Ok("value"));
    ///
    /// let failed = Finish::new("value", Some("error"));
    /// assert_eq!(failed.into_result(), Err("error"));
    /// ```
    pub fn into_result(self) -> Result<T, E> {
        if let Some(e) = self.error {
            Err(e)
        } else {
            Ok(self.value)
        }
    }

    /// Converts from `Finish<T, E>` to `Result<&T, &E>`.
    ///
    /// # Examples
    /// ```
    /// use libflate::Finish;
    ///
    /// let succeeded = Finish::new("value", None as Option<()>);
    /// assert_eq!(succeeded.as_result(), Ok(&"value"));
    ///
    /// let failed = Finish::new("value", Some("error"));
    /// assert_eq!(failed.as_result(), Err(&"error"));
    /// ```
    pub fn as_result(&self) -> Result<&T, &E> {
        if let Some(ref e) = self.error {
            Err(e)
        } else {
            Ok(&self.value)
        }
    }
}

/// A wrapper struct that completes the processing of the underlying instance when drops.
///
/// This calls `Complete:::complete` method of `T` when drops.
///
/// # Panics
///
/// If the invocation of `Complete::complete(T)` returns an error, `AutoFinish::drop()` will panic.
#[derive(Debug)]
pub struct AutoFinish<T: Complete> {
    inner: Option<T>,
}
impl<T: Complete> AutoFinish<T> {
    /// Makes a new `AutoFinish` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// use libflate::finish::AutoFinish;
    /// use libflate::gzip::Encoder;
    ///
    /// let plain = b"Hello World!";
    /// let mut buf = Vec::new();
    /// let mut encoder = AutoFinish::new(Encoder::new(&mut buf).unwrap());
    /// encoder.write_all(plain.as_ref()).unwrap();
    /// ```
    pub fn new(inner: T) -> Self {
        AutoFinish { inner: Some(inner) }
    }

    /// Unwraps this `AutoFinish` instance, returning the underlying instance.
    pub fn into_inner(mut self) -> T {
        self.inner.take().expect("Never fails")
    }
}
impl<T: Complete> Drop for AutoFinish<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            if let Err(e) = inner.complete() {
                panic!("{}", e);
            }
        }
    }
}
impl<T: Complete> Deref for AutoFinish<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().expect("Never fails")
    }
}
impl<T: Complete> DerefMut for AutoFinish<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().expect("Never fails")
    }
}
impl<T: Complete + Write> Write for AutoFinish<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.deref_mut().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.deref_mut().flush()
    }
}

/// A wrapper struct that completes the processing of the underlying instance when drops.
///
/// This calls `Complete:::complete` method of `T` when drops.
///
/// Note that this ignores the result of the invocation of `Complete::complete(T)`.
#[derive(Debug)]
pub struct AutoFinishUnchecked<T: Complete> {
    inner: Option<T>,
}
impl<T: Complete> AutoFinishUnchecked<T> {
    /// Makes a new `AutoFinishUnchecked` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// #[cfg(feature = "no_std")]
    /// use core2::io::Write;
    /// #[cfg(not(feature = "no_std"))]
    /// use std::io::Write;
    /// use libflate::finish::AutoFinishUnchecked;
    /// use libflate::gzip::Encoder;
    ///
    /// let plain = b"Hello World!";
    /// let mut buf = Vec::new();
    /// let mut encoder = AutoFinishUnchecked::new(Encoder::new(&mut buf).unwrap());
    /// encoder.write_all(plain.as_ref()).unwrap();
    /// ```
    pub fn new(inner: T) -> Self {
        AutoFinishUnchecked { inner: Some(inner) }
    }

    /// Unwraps this `AutoFinishUnchecked` instance, returning the underlying instance.
    pub fn into_inner(mut self) -> T {
        self.inner.take().expect("Never fails")
    }
}
impl<T: Complete> Drop for AutoFinishUnchecked<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = inner.complete();
        }
    }
}
impl<T: Complete> Deref for AutoFinishUnchecked<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().expect("Never fails")
    }
}
impl<T: Complete> DerefMut for AutoFinishUnchecked<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().expect("Never fails")
    }
}
impl<T: Complete + Write> Write for AutoFinishUnchecked<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.deref_mut().write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.deref_mut().flush()
    }
}

/// This trait allows to complete an I/O related processing.
pub trait Complete {
    /// Completes the current processing and returns the result.
    fn complete(self) -> io::Result<()>;
}
