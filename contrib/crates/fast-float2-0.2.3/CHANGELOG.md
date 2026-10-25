## 0.2.2

- Fix `no_std` support.
- Remove most uses of unsafe.
- Remove non-local safety invariants to prevent unsoundness.

## 0.2.1

- Fix undefined behavior in checking the buffer length.

## 0.2.0

- Fixed an edge case where long decimals with trailing zeros were truncated.
- Minor micro-optimization fixes in the fast path parser.
- Remove the use of unsafe when querying power-of-10 tables.
- Added float64 roundtrip fuzz target.
- Added tests for the power-of-5 table using num-bigint.
- Improvements and new options in the bench tool.
- Updated benchmark timings, added Apple M1 and AMD Rome timings.

## 0.1.0

Initial release, fully tested and benchmarked.
