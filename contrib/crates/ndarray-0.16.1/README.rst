ndarray
=========

The ``ndarray`` crate provides an *n*-dimensional container for general elements
and for numerics.

Please read the `API documentation on docs.rs`__
or take a look at the `quickstart tutorial <./README-quick-start.md>`_.

__ https://docs.rs/ndarray/

|build_status|_ |crates|_ |matrix-chat|_ |irc|_

.. |build_status| image:: https://github.com/rust-ndarray/ndarray/actions/workflows/ci.yaml/badge.svg
   :alt: CI build status
.. _build_status: https://github.com/rust-ndarray/ndarray/actions

.. |crates| image:: https://img.shields.io/crates/v/ndarray.svg
   :alt: ndarray at crates.io
.. _crates: https://crates.io/crates/ndarray

.. |matrix-chat| image:: https://img.shields.io/badge/Matrix-%23rust--sci%3Amatrix.org-lightgrey
   :alt: Matrix chat at #rust-sci:matrix.org
.. _matrix-chat: https://matrix.to/#/#rust-sci:matrix.org

.. |irc| image:: https://img.shields.io/badge/IRC-%23rust--sci%20on%20OFTC-lightgrey
   :alt: IRC at #rust-sci on OFTC
.. _irc: https://webchat.oftc.net/?channels=rust-sci

Highlights
----------

- Generic 1, 2, ..., *n*-dimensional arrays
- Owned arrays and array views
- Slicing, also with arbitrary step size, and negative indices to mean
  elements from the end of the axis.
- Views and subviews of arrays; iterators that yield subviews.

Status and Lookout
------------------

- Still iterating on and evolving the crate

  + The crate is continuously developing, and breaking changes are expected
    during evolution from version to version. We adopt the newest stable
    rust features if we need them.

- Performance:

  + Prefer higher order methods and arithmetic operations on arrays first,
    then iteration, and as a last priority using indexed algorithms.
  + Efficient floating point matrix multiplication even for very large
    matrices; can optionally use BLAS to improve it further.

Crate Feature Flags
-------------------

The following crate feature flags are available. They are configured in
your `Cargo.toml`.

- ``std``

  - Rust standard library (enabled by default)

  - This crate can be used without the standard library by disabling the
    default `std` feature. To do so, use this in your `Cargo.toml`:

    ::
      
      [dependencies]
      ndarray = { version = "0.x.y", default-features = false }
    

  - The `geomspace` `linspace` `logspace` `range` `std` `var` `var_axis` and `std_axis`
    methods are only available when `std` is enabled.

- ``serde``

  - Enables serialization support for serde 1.x

- ``rayon``

  - Enables parallel iterators, parallelized methods and ``par_azip!``.
  - Implies std

- ``approx``

  - Implementations of traits from version 0.5 of the [`approx`] crate.

- ``blas``

  - Enable transparent BLAS support for matrix multiplication.
    Uses ``blas-src`` for pluggable backend, which needs to be configured
    separately (see below).

- ``matrixmultiply-threading``

  - Enable the ``threading`` feature in the matrixmultiply package

- ``portable-atomic-critical-section``

  - Whether ``portable-atomic`` should use ``critical-section``

How to use with cargo
---------------------

::

    [dependencies]
    ndarray = "0.16.0"

How to enable BLAS integration
------------------------------

Blas integration is an optional add-on. Without BLAS, ndarray uses the
``matrixmultiply`` crate for matrix multiplication for ``f64`` and ``f32``
arrays (and it's always enabled as a fallback since it supports matrices of
arbitrary strides in both dimensions).

Depend and link to ``blas-src`` directly to pick a blas provider. Ndarray
presently requires a blas provider that provides the ``cblas-sys`` interface.  If
further feature selection is wanted or needed then you might need to depend directly on
the backend crate's source too.  The backend version **must** be the one that
``blas-src`` also depends on.

An example configuration using system openblas is shown below. Note that only
end-user projects (not libraries) should select provider::

    [dependencies]
    ndarray = { version = "0.16.0", features = ["blas"] }
    blas-src = { version = "0.10", features = ["openblas"] }
    openblas-src = { version = "0.10", features = ["cblas", "system"] }

Using system-installed dependencies can save a long time building dependencies.
An example configuration using (compiled) netlib is shown below anyway::

    [dependencies]
    ndarray = { version = "0.16.0", features = ["blas"] }
    blas-src = { version = "0.10.0", default-features = false, features = ["netlib"] }

When this is done, your program must also link to ``blas_src`` by using it or
explicitly including it in your code::

    extern crate blas_src;

The following versions have been verified to work together. For ndarray 0.15 or later,
there is no tight coupling to the ``blas-src`` version, so version selection is more flexible.

=========== ============ ================ ==============
``ndarray`` ``blas-src`` ``openblas-src`` ``netlib-src``
=========== ============ ================ ==============
0.16        0.10         0.10             0.8
0.15        0.8          0.10             0.8
0.15        0.7          0.9              0.8
0.14        0.6.1        0.9.0
0.13        0.2.0        0.6.0
=========== ============ ================ ==============

Recent Changes
--------------

See `RELEASES.md <./RELEASES.md>`_.

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.

