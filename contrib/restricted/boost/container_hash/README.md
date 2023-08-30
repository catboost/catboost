# Boost.ContainerHash

The Boost.ContainerHash library, part of [Boost C++ Libraries](https://boost.org),
provides `boost::hash`, an enhanced implementation of the
[hash function](https://en.wikipedia.org/wiki/Hash_function) object specified
by C++11 as `std::hash`, and several support facilities (`hash_combine`,
`hash_range`, `hash_unordered_range`).

`boost::hash` supports most standard types and some user-defined types out of
the box, and is extensible; it's possible for a user-defined type `X` to make
iself hashable via `boost::hash<X>` by defining an appropriate overload of the
function `hash_value`.

See [the documentation of the library](https://www.boost.org/libs/container_hash)
for more information.

## License

Distributed under the
[Boost Software License, Version 1.0](http://boost.org/LICENSE_1_0.txt).
