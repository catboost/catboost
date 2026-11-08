cd fuzz
RUSTFLAGS="-C target-cpu=native" cargo +nightly fuzz run passing -j 24 -- -max_total_time=300
RUSTFLAGS="-C target-cpu=native" cargo +nightly fuzz run failing -j 24 -- -max_total_time=300
RUSTFLAGS="-C target-cpu=native" cargo +nightly fuzz run real -j 24 -- -max_total_time=300
cd ..