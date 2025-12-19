extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

const _CATBOOST_VERSION: &str = "1.2.8";
// URL to prebuilt binaries
const _GITHUB_URL: &str = "https://github.com/catboost/catboost/releases/download/";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cb_model_interface_root = Path::new("../../libs/model_interface/")
        .canonicalize()
        .unwrap();

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cb_model_interface_root.display()))
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings.");
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    #[cfg(feature = "use_prebuilt")]
    {
        let deps_path = dunce::canonicalize(Path::new(&format!(
            "{}/../../../deps",
            env::var("OUT_DIR").unwrap()
        )))
        .unwrap();
        let deps_path = deps_path.to_string_lossy();

        let target = env::var("TARGET").unwrap();
        let library_file: &str;
        if cfg!(target_os = "macos") {
            library_file = "libcatboostmodel.dylib";
        } else if cfg!(target_os = "linux") {
            library_file = "libcatboostmodel.so";
        } else if cfg!(target_os = "windows") {
            library_file = "catboostmodel.dll";
        } else {
            panic!(
                "Prebuilt CatBoost library is not available for target: {}",
                target
            );
        }
        let deps_target = format!("{deps_path}/{library_file}");
        if !std::fs::exists(&deps_target).unwrap() {
            let download_url = format!("{_GITHUB_URL}/v{_CATBOOST_VERSION}/{library_file}");
            web_copy(&format!("{download_url}"), &deps_target)
                .expect("Failed to download prebuilt CatBoost library");
        }
        println!("cargo:rustc-link-search=native={}", deps_path);
    }
    #[cfg(not(feature = "use_prebuilt"))]
    {
        let mut build_native_args = vec![
            "../../../build/build_native.py",
            "--targets",
            "catboostmodel",
            "--build-root-dir",
            out_dir.to_str().unwrap(),
        ];
        let debug = env::var("DEBUG").unwrap();
        if debug == "true" {
            build_native_args.push("--build-type=Debug");
        } else {
            build_native_args.push("--build-type=Release");
        }

        #[cfg(feature = "gpu")]
        build_native_args.push("--have-cuda");

        let build_cmd_status = std::process::Command::new("python")
            .args(&build_native_args)
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to run build_native.py : {}", e);
            });

        if !build_cmd_status.success() {
            panic!("Building with build_native.py failed");
        }
        println!(
            "cargo:rustc-link-search={}",
            out_dir.join("catboost/libs/model_interface").display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=catboostmodel");
}

#[cfg(feature = "use_prebuilt")]
fn web_copy(web_src: &str, target: &str) -> Result<(), Box<dyn std::error::Error>> {
    dbg!(&web_src);
    let resp = reqwest::blocking::get(web_src)?;
    let body = resp.bytes()?;
    std::fs::write(target, &body)?;
    Ok(())
}
