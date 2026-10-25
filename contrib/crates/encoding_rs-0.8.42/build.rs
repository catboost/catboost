fn main() {
    println!("cargo::rustc-check-cfg=cfg(slow_mm_packus_epi16)");
    if rustversion::cfg!(all(since(1.96), before(1.99))) {
        println!("cargo:rustc-cfg=slow_mm_packus_epi16");
    }
}
