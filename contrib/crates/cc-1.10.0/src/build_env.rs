//! The environment a [`Build`](crate::Build) reads its configuration from and
//! runs its tools in.

use std::{
    collections::hash_map::DefaultHasher,
    env,
    ffi::OsStr,
    fmt,
    hash::{Hash, Hasher},
    process::Command,
    sync::Arc,
};

use crate::{
    flags::RustcCodegenFlags,
    target::{TargetInfo, TargetInfoParser},
    utilities::OnceLock,
    Error, ErrorKind,
};

/// Environment variables, in the order they are applied.
pub(crate) type EnvVars = [(Arc<OsStr>, Arc<OsStr>)];

/// The environment of a [`Build`](crate::Build): a snapshot of the process
/// environment, taken on the first read or given with
/// `Build::set_envs_snapshot`, and used for cc's own lookups, then the entries
/// set with [`Build::env`](crate::Build::env), which win over it in child
/// processes.
#[derive(Clone, Debug, Default)]
pub(crate) struct BuildEnv {
    /// Private, so that nothing can read the environment without taking the
    /// snapshot first.
    inherited: OnceLock<EnvSnapshot>,
    pub(crate) explicit: Vec<(Arc<OsStr>, Arc<OsStr>)>,
}

impl BuildEnv {
    /// An environment whose snapshot is already taken.
    pub(crate) fn with_inherited(
        inherited: EnvSnapshot,
        explicit: Vec<(Arc<OsStr>, Arc<OsStr>)>,
    ) -> Self {
        Self {
            inherited: inherited.into(),
            explicit,
        }
    }

    /// The process environment as it was when this was first called, unless
    /// [`BuildEnv::set_inherited`] replaced it.
    pub(crate) fn inherited(&self) -> &EnvSnapshot {
        self.inherited.get_or_init(EnvSnapshot::capture)
    }

    /// Use `inherited` instead of the process environment, also if the
    /// snapshot was already taken.
    pub(crate) fn set_inherited(&mut self, inherited: EnvSnapshot) {
        self.inherited = inherited.into();
    }

    /// Make `cmd` run in this environment. Like [`EnvSnapshot::apply`], this
    /// clears anything set on `cmd`'s environment before.
    pub(crate) fn apply(&self, cmd: &mut Command) {
        self.inherited().apply(cmd);
        cmd.envs(pairs(&self.explicit));
    }
}

/// A copy of the process environment, or the variables given to
/// `Build::set_envs_snapshot`. Clones share the copy, and what cc parses from
/// it.
#[derive(Clone)]
pub(crate) struct EnvSnapshot(Arc<EnvSnapshotInner>);

/// What is parsed from `vars` lives here, so that a new snapshot can't be
/// paired with values parsed from an old one.
struct EnvSnapshotInner {
    vars: Box<EnvVars>,
    /// Hash of `vars`, worked out once.
    hash: u64,
    /// The target Cargo describes in `vars`, parsed on first use.
    cargo_target: TargetInfoParser,
    /// `CARGO_ENCODED_RUSTFLAGS` from `vars`, parsed on first use, or `None`
    /// if it is not set.
    rustc_codegen_flags: OnceLock<Result<Option<RustcCodegenFlags>, Error>>,
}

impl EnvSnapshot {
    fn new(vars: Box<EnvVars>) -> Self {
        // A fixed hasher, so that equal snapshots get equal hashes for the
        // whole process.
        let mut hasher = DefaultHasher::new();
        vars.hash(&mut hasher);
        Self(Arc::new(EnvSnapshotInner {
            hash: hasher.finish(),
            vars,
            cargo_target: TargetInfoParser::default(),
            rustc_codegen_flags: OnceLock::new(),
        }))
    }

    /// Copy the process environment as it is now. A variable it holds twice,
    /// such as `Path` and `PATH` on Windows, takes its last value, as in the
    /// tools cc runs.
    pub(crate) fn capture() -> Self {
        Self::from_vars(&mut env::vars_os().map(|(key, value)| (key.into(), value.into())))
    }

    /// A snapshot holding `vars`. A variable given more than once takes its
    /// last value, as with [`Command::envs`].
    pub(crate) fn from_vars(vars: &mut dyn Iterator<Item = (Arc<OsStr>, Arc<OsStr>)>) -> Self {
        let mut deduped: Vec<(Arc<OsStr>, Arc<OsStr>)> = Vec::with_capacity(vars.size_hint().0);
        for (key, value) in vars {
            match deduped.iter_mut().find(|(k, _)| is_same_key(k, &key)) {
                Some(entry) => entry.1 = value,
                None => deduped.push((key, value)),
            }
        }
        Self::new(deduped.into_boxed_slice())
    }

    #[cfg(test)]
    pub(crate) fn from_pairs(vars: &[(&str, &str)]) -> Self {
        Self::new(
            vars.iter()
                .map(|(key, value)| (OsStr::new(key).into(), OsStr::new(value).into()))
                .collect(),
        )
    }

    /// Look up `key` the way [`env::var_os`] would have when the snapshot was
    /// taken, except that a variable held twice gives its last value.
    pub(crate) fn get(&self, key: &OsStr) -> Option<&Arc<OsStr>> {
        self.0
            .vars
            .iter()
            .find(|(k, _)| is_same_key(k, key))
            .map(|(_, value)| value)
    }

    /// Look up a variable that Cargo sets for build scripts, such as `TARGET`,
    /// as a string.
    pub(crate) fn get_cargo_var(&self, key: &str) -> Result<&str, Error> {
        let Some(value) = self.get(OsStr::new(key)) else {
            return Err(Error::new(
                ErrorKind::EnvVarNotFound,
                format!("environment variable {key} not defined"),
            ));
        };
        value.to_str().ok_or_else(|| {
            Error::new(
                ErrorKind::EnvVarNotFound,
                format!("environment variable {key} is not valid utf-8: {value:?}"),
            )
        })
    }

    /// The target Cargo builds for, from `TARGET` and `CARGO_CFG_TARGET_*`.
    pub(crate) fn cargo_target(&self) -> Result<TargetInfo<'_>, Error> {
        self.0
            .cargo_target
            .parse_from_cargo_environment_variables(self)
    }

    /// The flags in `CARGO_ENCODED_RUSTFLAGS`, or `None` if it is not set.
    pub(crate) fn rustc_codegen_flags(&self) -> Result<Option<&RustcCodegenFlags>, Error> {
        self.0
            .rustc_codegen_flags
            .get_or_init(|| match self.get(OsStr::new("CARGO_ENCODED_RUSTFLAGS")) {
                Some(flags) => RustcCodegenFlags::parse(&flags.to_string_lossy()).map(Some),
                None => Ok(None),
            })
            .as_ref()
            .map(Option::as_ref)
            .map_err(Error::clone)
    }

    /// Make `cmd` run in this environment rather than the current one.
    ///
    /// This clears the environment `cmd` would inherit and sets every variable
    /// of the snapshot, so the child does not depend on the process environment
    /// at the time it is spawned, which another thread may be changing. It also
    /// clears anything set on `cmd`'s environment before, so call it first.
    pub(crate) fn apply(&self, cmd: &mut Command) {
        cmd.env_clear();
        cmd.envs(pairs(&self.0.vars));
    }
}

impl PartialEq for EnvSnapshot {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.hash == other.0.hash && self.0.vars == other.0.vars)
    }
}

impl Eq for EnvSnapshot {}

impl fmt::Debug for EnvSnapshot {
    // The values may hold secrets, and a `Build` can end up in a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EnvSnapshot(<{} variables>)", self.0.vars.len())
    }
}

fn pairs(vars: &EnvVars) -> impl Iterator<Item = (&OsStr, &OsStr)> {
    vars.iter().map(|(key, value)| (&**key, &**value))
}

/// Environment variable names are case-insensitive on Windows.
fn is_same_key(a: &OsStr, b: &OsStr) -> bool {
    if cfg!(windows) {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn snapshot(vars: &[(&str, &str)]) -> EnvSnapshot {
        EnvSnapshot::from_pairs(vars)
    }

    fn envs(cmd: &Command) -> Vec<(OsString, Option<OsString>)> {
        cmd.get_envs()
            .map(|(key, value)| (key.to_owned(), value.map(OsStr::to_owned)))
            .collect()
    }

    #[test]
    fn apply_replaces_the_whole_environment() {
        let mut cmd = Command::new("cc");
        cmd.env("CC_TEST_SET_BEFORE", "1")
            .env_remove("CC_TEST_REMOVED_BEFORE");
        snapshot(&[("CC_TEST_IN_SNAPSHOT", "1")]).apply(&mut cmd);
        assert_eq!(
            envs(&cmd),
            [("CC_TEST_IN_SNAPSHOT".into(), Some("1".into()))]
        );
    }

    #[test]
    fn build_env_wins_over_snapshot() {
        let env = BuildEnv::with_inherited(
            snapshot(&[("CC_TEST_ORDER", "inherited")]),
            vec![(
                OsStr::new("CC_TEST_ORDER").into(),
                OsStr::new("explicit").into(),
            )],
        );
        let mut cmd = Command::new("cc");
        env.apply(&mut cmd);
        assert_eq!(
            envs(&cmd),
            [("CC_TEST_ORDER".into(), Some("explicit".into()))]
        );
    }

    #[test]
    fn snapshot_lookup_shares_the_value() {
        let snapshot = snapshot(&[("CC_TEST_SHARED", "a")]);
        let value = snapshot.get(OsStr::new("CC_TEST_SHARED")).unwrap();
        assert!(Arc::ptr_eq(value, &snapshot.0.vars[0].1));
    }

    #[test]
    fn snapshot_lookup_follows_the_platform_case_rule() {
        let snapshot = snapshot(&[("Path", "a")]);
        assert_eq!(
            snapshot.get(OsStr::new("Path")).map(|value| &**value),
            Some(OsStr::new("a"))
        );
        let other_case = snapshot.get(OsStr::new("PATH")).map(|value| &**value);
        if cfg!(windows) {
            assert_eq!(other_case, Some(OsStr::new("a")));
        } else {
            assert_eq!(other_case, None);
        }
    }

    #[test]
    fn set_inherited_replaces_a_taken_snapshot() {
        let mut env = BuildEnv::default();
        env.inherited();
        env.set_inherited(snapshot(&[("CC_TEST_REPLACED", "1")]));
        assert_eq!(env.inherited(), &snapshot(&[("CC_TEST_REPLACED", "1")]));
    }

    /// A variable given twice takes its last value, in cc's own lookups as in
    /// the tools it runs.
    #[test]
    fn snapshot_from_vars_keeps_the_last_value() {
        let snapshot = EnvSnapshot::from_vars(
            &mut [
                ("CC_TEST_TWICE", "first"),
                ("Path", "a"),
                ("CC_TEST_TWICE", "last"),
                ("PATH", "b"),
            ]
            .into_iter()
            .map(|(key, value)| (OsStr::new(key).into(), OsStr::new(value).into())),
        );
        assert_eq!(
            snapshot
                .get(OsStr::new("CC_TEST_TWICE"))
                .map(|value| &**value),
            Some(OsStr::new("last"))
        );
        let mut cmd = Command::new("cc");
        snapshot.apply(&mut cmd);
        for key in ["CC_TEST_TWICE", "Path", "PATH"] {
            let key = OsStr::new(key);
            let in_child = cmd
                .get_envs()
                .filter(|(k, _)| is_same_key(k, key))
                .last()
                .and_then(|(_, value)| value);
            assert_eq!(snapshot.get(key).map(|value| &**value), in_child, "{key:?}");
        }
    }

    #[test]
    fn snapshot_comparison_follows_the_variables() {
        let a = snapshot(&[("CC_TEST_A", "1"), ("CC_TEST_B", "2")]);
        // Same variables, captured separately.
        let same = snapshot(&[("CC_TEST_A", "1"), ("CC_TEST_B", "2")]);
        assert!(!Arc::ptr_eq(&a.0, &same.0));
        assert_eq!(a, same);
        assert_eq!(a.0.hash, same.0.hash);
        assert_eq!(a, a.clone());

        for other in [
            snapshot(&[("CC_TEST_A", "1")]),
            snapshot(&[("CC_TEST_A", "1"), ("CC_TEST_B", "3")]),
            // A boundary moved between name and value.
            snapshot(&[("CC_TEST_A", "1"), ("CC_TEST_B2", "")]),
            // Names are compared exactly, also on Windows.
            snapshot(&[("CC_TEST_A", "1"), ("cc_test_b", "2")]),
        ] {
            assert_ne!(a, other);
        }
    }

    #[test]
    fn snapshot_debug_hides_values() {
        let debug = format!("{:?}", snapshot(&[("CC_TEST_SECRET", "hunter2")]));
        assert!(!debug.contains("hunter2"), "{debug}");
    }
}
