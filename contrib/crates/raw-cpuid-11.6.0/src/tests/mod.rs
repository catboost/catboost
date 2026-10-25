#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod i5_3337u;

mod i7_12700k;
mod ryzen_matisse;
mod xeon_gold_6252;

use crate::*;

#[test]
fn cpuid_impls_debug() {
    fn debug_required<T: Debug>(_t: T) {}

    debug_required(CpuId::new());
}
