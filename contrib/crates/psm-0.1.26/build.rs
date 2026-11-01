fn find_assembly(
    arch: &str,
    endian: &str,
    os: &str,
    env: &str,
    masm: bool,
) -> Option<(&'static str, bool)> {
    match (arch, endian, os, env) {
        // The implementations for stack switching exist, but, officially, doing so without Fibers
        // is not supported in Windows. For x86_64 the implementation actually works locally,
        // but failed tests in CI (???). Might want to have a feature for experimental support
        // here.
        ("x86", _, "windows", _) => {
            if masm {
                Some(("src/arch/x86_msvc.asm", false))
            } else {
                Some(("src/arch/x86_windows_gnu.s", false))
            }
        }
        ("x86_64", _, "windows", _) => {
            if masm {
                Some(("src/arch/x86_64_msvc.asm", false))
            } else {
                Some(("src/arch/x86_64_windows_gnu.s", false))
            }
        }
        ("x86_64", _, "cygwin", _) => {
            Some(("src/arch/x86_64_windows_gnu.s", false))
        }
        ("arm", _, "windows", "msvc") => Some(("src/arch/arm_armasm.asm", false)),
        ("aarch64", _, "windows", _) => {
            if masm {
                Some(("src/arch/aarch64_armasm.asm", false))
            } else {
                Some(("src/arch/aarch_aapcs64.s", false))
            }
        }
        ("x86", _, _, _) => Some(("src/arch/x86.s", true)),
        ("x86_64", _, _, _) => Some(("src/arch/x86_64.s", true)),
        ("arm", _, _, _) => Some(("src/arch/arm_aapcs.s", true)),
        ("aarch64", _, _, _) => Some(("src/arch/aarch_aapcs64.s", true)),
        ("powerpc", _, _, _) => Some(("src/arch/powerpc32.s", true)),
        ("powerpc64", _, _, "musl") => Some(("src/arch/powerpc64_openpower.s", true)),
        ("powerpc64", "little", _, _) => Some(("src/arch/powerpc64_openpower.s", true)),
        ("powerpc64", _, "aix", _) => Some(("src/arch/powerpc64_aix.s", true)),
        ("powerpc64", _, _, _) => Some(("src/arch/powerpc64.s", true)),
        ("s390x", _, _, _) => Some(("src/arch/zseries_linux.s", true)),
        ("mips", _, _, _) => Some(("src/arch/mips_eabi.s", true)),
        ("mips64", _, _, _) => Some(("src/arch/mips64_eabi.s", true)),
        ("sparc64", _, _, _) => Some(("src/arch/sparc64.s", true)),
        ("sparc", _, _, _) => Some(("src/arch/sparc_sysv.s", true)),
        ("riscv32", _, _, _) => Some(("src/arch/riscv.s", true)),
        ("riscv64", _, _, _) => Some(("src/arch/riscv64.s", true)),
        ("wasm32", _, _, _) => Some(("src/arch/wasm32.o", true)),
        ("loongarch64", _, _, _) => Some(("src/arch/loongarch64.s", true)),
        _ => None,
    }
}

fn main() {
    use std::env::var;

    println!("cargo:rustc-check-cfg=cfg(switchable_stack,asm,link_asm)");

    if var("CARGO_CFG_MIRI").is_ok() {
        // Miri doesn't have a stack limit and the inline asm wouldn't work on miri anyway.
        return;
    }

    let arch = var("CARGO_CFG_TARGET_ARCH").unwrap();
    let env = var("CARGO_CFG_TARGET_ENV").unwrap();
    let os = var("CARGO_CFG_TARGET_OS").unwrap();
    let endian = var("CARGO_CFG_TARGET_ENDIAN").unwrap();

    let mut cfg = cc::Build::new();

    let msvc = cfg.get_compiler().is_like_msvc();
    // If we're targeting msvc, either via regular MS toolchain or clang-cl, we
    // will _usually_ want to use the regular Microsoft assembler if it exists,
    // which is done for us within cc, however it _probably_ won't exist if
    // we're in a cross-compilation context pm a platform that can't natively
    // run Windows executables, so in that case we instead use the the equivalent
    // GAS assembly file instead. This logic can be removed once LLVM natively
    // supports compiling MASM, but that is not stable yet
    let masm = msvc && var("HOST").expect("HOST env not set").contains("windows");

    let asm = if let Some((asm, canswitch)) = find_assembly(&arch, &endian, &os, &env, masm) {
        println!("cargo:rustc-cfg=asm");
        println!("cargo:rustc-cfg=link_asm");
        if canswitch {
            println!("cargo:rustc-cfg=switchable_stack")
        }
        asm
    } else {
        println!(
            "cargo:warning=Target {}-{}-{} has no assembly files!",
            arch, os, env
        );
        return;
    };

    if !msvc {
        cfg.flag("-xassembler-with-cpp");
        cfg.define(&*format!("CFG_TARGET_OS_{}", os), None);
        cfg.define(&*format!("CFG_TARGET_ARCH_{}", arch), None);
        cfg.define(&*format!("CFG_TARGET_ENV_{}", env), None);
    }

    // For wasm targets we ship a precompiled `*.o` file so we just pass that
    // directly to `ar` to assemble an archive. Otherwise we're actually
    // compiling the source assembly file.
    if asm.ends_with(".o") {
        cfg.object(asm);
    } else {
        cfg.file(asm);
    }

    cfg.compile("libpsm_s.a");
}
