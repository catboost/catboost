extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cb_model_interface_root = Path::new("../../libs/model_interface/")
        .canonicalize()
        .unwrap();

    Command::new("../../../ya")
        .args(&[
            "make",
            "-r",
            cb_model_interface_root.to_str().unwrap(),
            // "--sanitize=address",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| {
            panic!("Failed to yamake libcatboostmodel: {}", e);
        });

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cb_model_interface_root.display()))
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings.");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!(
        "cargo:rustc-link-search={}",
        out_dir.join("catboost/libs/model_interface").display()
    );

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
    println!("cargo:rustc-link-lib=dylib=catboostmodel");
}
