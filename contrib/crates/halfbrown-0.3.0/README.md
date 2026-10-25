# Halfbrown HashMap &emsp; [![Build Status]][drone.io] [![Windows Build Status]][appveyor.com] [![Latest Version]][crates.io]

[Build Status]: https://cloud.drone.io/api/badges/Licenser/halfbrown/status.svg
[drone.io]: https://cloud.drone.io/Licenser/halfbrown
[Windows Build Status]: https://ci.appveyor.com/api/projects/status/0kf0v6hj5v2gite9?svg=true
[appveyor.com]: https://ci.appveyor.com/project/Licenser/halfbrown
[Latest Version]: https://img.shields.io/crates/v/halfbrown.svg
[crates.io]: https://crates.io/crates/halfbrown

**Hashmap implementation that dynamically switches from a vector based backend to a hashbrown based backend as the number of keys grows**

---

Note: The heavy lifting in this is done in [hashbrown](https://github.com/rust-lang/hashbrown), and the docs and API are copied from them.

Halfbrown, is a hashmap implementation that uses two backends to optimize for different cernairos:

## VecMap

For less then 32 key value pairs it uses a dumb vector based map implementation. This trades the need to iterator through the
vector for not having to hash strings on lookup or inserts.

## Hashbrown

For more then 32 elements it upgrades the map to aq hashbrown base map to account for longer itteration times.

## License

halfbrown itself is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
* MIT license ([LICENSE-MIT](http://opensource.org/licenses/MIT))

at your option.

Code / docs copied from [hashbrown](https://github.com/rust-lang/hashbrown) are obviously licensed under their License.
