# Boost.System

The Boost.System library, part of [Boost C++ Libraries](https://boost.org),
implements an extensible framework for error reporting in the form of an
`error_code` class and supporting facilities.

It has been proposed for the C++11 standard, has been accepted, and
is now available as part of the standard library in the `<system_error>`
header. However, the Boost implementation has continued to evolve and
gain enhancements and additional functionality, such as support for
attaching [source locations](https://www.boost.org/doc/libs/release/libs/assert/doc/html/assert.html#source_location_support)
to `error_code`, and a `result<T>` class that can carry either a value
or an error code.

See [the documentation of System](http://boost.org/libs/system) for more
information.

Since `<system_error>` is a relatively undocumented portion of the C++
standard library, the documentation of Boost.System may be useful to you
even if you use the standard components.

## License

Distributed under the
[Boost Software License, Version 1.0](http://boost.org/LICENSE_1_0.txt).
