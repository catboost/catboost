#[cfg(any(feature = "filter-by-regex", feature = "case-insensitive"))]
use std::{env, path::Path};

#[cfg(feature = "filter-by-regex")]
use chrono_tz_build::FILTER_ENV_VAR_NAME;

fn main() {
    #[cfg(feature = "filter-by-regex")]
    println!("cargo:rerun-if-env-changed={FILTER_ENV_VAR_NAME}");
    #[cfg(any(feature = "filter-by-regex", feature = "case-insensitive"))]
    chrono_tz_build::main(
        Path::new(&env::var("OUT_DIR").unwrap()),
        cfg!(feature = "filter-by-regex"),
        cfg!(feature = "case-insensitive"),
    );
}
