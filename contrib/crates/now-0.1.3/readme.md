<div align="center">
    <h1>now</h1>
    <p>a time toolkit for chrono inspired by <a href="https://github.com/jinzhu/now">jinzhu/now</a></p>
    <img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/kilerd/now/Develop%20Build">
    <img alt="Crates.io" src="https://img.shields.io/crates/v/now">
    <a href="https://docs.rs/now"><img alt="docs.rs" src="https://img.shields.io/docsrs/now/latest"></a>
    <img src='https://coveralls.io/repos/github/Kilerd/now/badge.svg?branch=master' alt='Coverage Status' />
    <img alt="Crates.io (recent)" src="https://img.shields.io/crates/dr/now">
    <img alt="Crates.io" src="https://img.shields.io/crates/l/now">
</div>

## Installation

add dependency to `Cargo.toml`:

```toml
# Cargo.toml
[dependencies]
now = "0.1"
```

or, you can use cargo-edit to add dependency via:

```shell
cargo add now
```

## Usage
Trait `TimeZoneNow` provide convenient and human-readable method for `chrono::Timezone`:
```rust
use chrono::FixedOffset;
use now::TimeZoneNow;

let offset = FixedOffset::east(60 * 60 * 8);

offset.now();                   // 2021-07-21T12:48:58.626947+08:00

offset.beginning_of_minute();   // 2021-07-21T12:48:00+08:00
offset.beginning_of_hour();     // 2021-07-21T12:00:00+08:00
offset.beginning_of_day();      // 2021-07-21T00:00:00+08:00
offset.beginning_of_week();     // 2021-07-19T00:00:00+08:00
offset.beginning_of_month();    // 2021-07-01T00:00:00+08:00
offset.beginning_of_quarter();  // 2021-07-01T00:00:00+08:00
offset.beginning_of_year();     // 2021-01-01T00:00:00+08:00

offset.end_of_minute();         // 2021-07-21T12:48:59.999999999+08:00
offset.end_of_hour();           // 2021-07-21T12:59:59.999999999+08:00
offset.end_of_day();            // 2021-07-21T23:59:59.999999999+08:00
offset.end_of_week();           // 2021-07-25T23:59:59.999999999+08:00
offset.end_of_month();          // 2021-07-31T23:59:59.999999999+08:00
offset.end_of_quarter();        // 2021-09-30T23:59:59.999999999+08:00
offset.end_of_year();           // 2021-12-31T23:59:59.999999999+08:00
```

And Trait `DateTimeNow` support those methods for `chrono:DateTime<T:Timezone>`:
```rust
use now::DateTimeNow;

let time = Utc::now();          // 2021-07-21T05:18:25.011480Z

time.beginning_of_minute();     // 2021-07-21T05:18:00Z
time.beginning_of_hour();       // 2021-07-21T05:00:00Z
time.beginning_of_day();        // 2021-07-21T00:00:00Z
time.beginning_of_week();       // 2021-07-19T00:00:00Z
time.beginning_of_month();      // 2021-07-01T00:00:00Z
time.beginning_of_quarter();    // 2021-07-01T00:00:00Z
time.beginning_of_year();       // 2021-01-01T00:00:00Z

time.end_of_minute();           // 2021-07-21T05:18:59.999999999Z
time.end_of_hour();             // 2021-07-21T05:59:59.999999999Z
time.end_of_day();              // 2021-07-21T23:59:59.999999999Z
time.end_of_week();             // 2021-07-25T23:59:59.999999999Z
time.end_of_month();            // 2021-07-31T23:59:59.999999999Z
time.end_of_quarter();          // 2021-09-30T23:59:59.999999999Z
time.end_of_year();             // 2021-12-31T23:59:59.999999999Z
```

Because now is based on `chrono`, so features like **Leap Year** and **daynight time**(provided by `chrono-tz`) are also supoprted by now.

```rust
let naive_date_time = NaiveDate::from_ymd(2024, 2, 10).and_hms(0, 0, 1);
let date_time: DateTime<Utc> = Utc.from_local_datetime(&naive_date_time).unwrap();
let time = date_time.end_of_month();

assert_eq!(29, time.day()); // 2024 is leap year
```