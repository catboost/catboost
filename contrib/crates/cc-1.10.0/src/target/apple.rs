use super::TargetInfo;

impl TargetInfo<'_> {
    pub(crate) fn apple_sdk_name(&self) -> &'static str {
        match (self.os, self.env) {
            // The target_env variable, as written here:
            // https://doc.rust-lang.org/reference/conditional-compilation.html#target_env
            // is only really used for disambiguation, so we use a "default" case instead of
            // checking for a blank string.
            ("macos", _) => "macosx",
            ("ios", "sim") => "iphonesimulator",
            ("ios", "macabi") => "macosx",
            ("ios", _) => "iphoneos",
            ("tvos", "sim") => "appletvsimulator",
            ("tvos", _) => "appletvos",
            ("watchos", "sim") => "watchsimulator",
            ("watchos", _) => "watchos",
            ("visionos", "sim") => "xrsimulator",
            ("visionos", _) => "xros",
            (os, _) => panic!("invalid Apple target OS {}", os),
        }
    }

    /// The variable that sets the deployment target on this Apple target, read
    /// by cc and by the compiler.
    pub(crate) fn apple_deployment_target_var(&self) -> Option<&'static str> {
        match self.os {
            "macos" => Some("MACOSX_DEPLOYMENT_TARGET"),
            "ios" => Some("IPHONEOS_DEPLOYMENT_TARGET"),
            "tvos" => Some("TVOS_DEPLOYMENT_TARGET"),
            "watchos" => Some("WATCHOS_DEPLOYMENT_TARGET"),
            "visionos" => Some("XROS_DEPLOYMENT_TARGET"),
            _ => None,
        }
    }

    pub(crate) fn apple_version_flag(&self, min_version: &str) -> String {
        // There are many aliases for these, and `-mtargetos=` is preferred on Clang nowadays, but
        // for compatibility with older Clang, we use the earliest supported name here.
        //
        // NOTE: GCC does not support `-miphoneos-version-min=` etc. (because it does not support
        // iOS in general), but we specify them anyhow in case we actually have a Clang-like
        // compiler disguised as a GNU-like compiler, or in case GCC adds support for these in the
        // future.
        //
        // See also:
        // https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-mmacos-version-min
        // https://clang.llvm.org/docs/AttributeReference.html#availability
        // https://gcc.gnu.org/onlinedocs/gcc/Darwin-Options.html#index-mmacosx-version-min
        match (self.os, self.env) {
            // The target_env variable, as written here:
            // https://doc.rust-lang.org/reference/conditional-compilation.html#target_env
            // is only really used for disambiguation, so we use a "default" case instead of
            // checking for a blank string.
            ("macos", _) => format!("-mmacosx-version-min={min_version}"),
            ("ios", "sim") => format!("-mios-simulator-version-min={min_version}"),
            ("ios", "macabi") => format!("-mtargetos=ios{min_version}-macabi"),
            ("ios", _) => format!("-miphoneos-version-min={min_version}"),
            ("tvos", "sim") => format!("-mappletvsimulator-version-min={min_version}"),
            ("tvos", _) => format!("-mappletvos-version-min={min_version}"),
            ("watchos", "sim") => format!("-mwatchsimulator-version-min={min_version}"),
            ("watchos", _) => format!("-mwatchos-version-min={min_version}"),
            // `-mxros-version-min` does not exist
            // https://github.com/llvm/llvm-project/issues/88271
            ("visionos", "sim") => format!("-mtargetos=xros{min_version}-simulator"),
            ("visionos", _) => format!("-mtargetos=xros{min_version}"),
            (os, _) => panic!("invalid Apple target OS {}", os),
        }
    }
}
