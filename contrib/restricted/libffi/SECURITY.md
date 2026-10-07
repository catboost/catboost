# The libffi Security Policy

This document describes the policy followed by the libffi maintainers to
handle bugs that may have a security impact.  This includes determining if a
bug has a security impact, reporting such bugs privately, and handling them
through to public disclosure.  This policy may evolve over time, so if you
are reading this from a release tarball, please check the latest copy in the
[libffi repository](https://github.com/libffi/libffi/blob/master/SECURITY.md)
for current reporting instructions.

## Scope and threat model

libffi is a low-level runtime library.  Its callers provide call interface
descriptions (`ffi_cif`), function pointers, argument buffers, return-value
buffers, and (for closures) user-supplied handler functions.  libffi
**trusts** its caller to:

* describe argument and return types accurately for the function being
  called or wrapped,
* supply argument buffers sized and aligned to match the described types,
* pass function pointers that follow the declared ABI, and
* manage the lifetime of CIFs, closures, and any buffers passed in.

A program that constructs CIFs, argument buffers, or function pointers from
untrusted input is responsible for validating them before calling into
libffi.  Crashes, memory corruption, or undefined behavior resulting from a
caller violating these contracts are caller bugs, not libffi
vulnerabilities.

Within that threat model, libffi has security obligations of its own.  When
a caller uses the library correctly, libffi must not introduce memory
corruption, leak information across calls, or compromise the executable
memory it manages for closures.  Bugs that break those obligations are
in scope for this policy.

## What is a security bug?

The following guidelines are used to decide whether a defect is treated as
a security vulnerability.  In every case, the question is whether libffi
itself is the cause when it is used in a manner consistent with its
documentation.

* Buffer overflows, out-of-bounds reads, or other memory-corruption bugs in
  the libffi sources (including the architecture-specific call and closure
  trampolines) triggered by valid CIFs, ABI-conforming arguments, or
  valid closure invocations.
* Bugs in the closure allocator that leave memory simultaneously writable
  and executable when libffi was built and configured to enforce W^X, or
  that allow a caller's data to be executed as code.
* Type-confusion or ABI bugs in libffi's generic code or per-target backends
  that cause memory corruption when the caller's CIF and arguments are
  consistent with the target ABI.
* Information disclosure through return-value or argument-marshalling paths
  that copy beyond the bounds of the described types.
* Use-after-free, double-free, or other lifetime bugs inside libffi's own
  data structures (CIFs, closures, internal caches).
* Crashes or denial of service triggered by valid, ABI-conforming inputs
  in environments where applications routinely expose libffi to such inputs
  (for example, language runtimes built on top of libffi).

The following are generally **not** treated as security bugs:

* ABI mismatches caused by an incorrect CIF, wrong argument types, or a
  function pointer whose real signature does not match the description
  given to libffi.  These are caller errors even though they may corrupt
  memory.
* Failures that require the caller to construct a hostile CIF or closure
  handler from untrusted data without validation.  The application, not
  libffi, must validate untrusted FFI descriptions.
* Build, configuration, or porting issues that prevent libffi from running
  on a platform but do not affect correctly built deployments.
* Missing or weakened post-exploitation hardening (see below) when it is not
  itself the root cause of a vulnerability.

### Closures and executable memory

libffi allocates executable trampolines so that runtime-generated closures
can be called like ordinary C functions.  On platforms that support it,
libffi uses static trampolines, `memfd_create`, dual-mapping, or other
mechanisms to keep writable and executable views of the trampoline memory
separate.

Bugs that defeat the W^X guarantee on platforms where libffi advertises
support for it — for example, by leaving a page both writable and
executable, by exposing a writable alias to attacker-controllable data, or
by emitting a trampoline that branches into caller-controlled memory — are
security bugs.

On platforms where W^X enforcement is not available and libffi falls back
to a writable-and-executable allocation (as documented), the absence of
W^X is not itself a vulnerability.  The documented fallback is a
portability concession, and callers that need stronger guarantees should
configure libffi to disable closures or run on a supported platform.

### Architecture-specific backends

libffi contains hand-written assembly and ABI-specific C for each target.
Bugs that corrupt memory, mishandle the stack, or skip required register
saves in those backends are in scope when triggered by ABI-conforming
input.  Bugs that merely produce a wrong-but-non-corrupting result for an
exotic-but-valid type combination are treated as ordinary correctness bugs
unless the wrong result is itself a security exposure (for example, a
silently truncated pointer that bypasses a caller's bounds check).

### Post-exploitation countermeasures

Hardening such as stack protectors, control-flow integrity, and PaX/MPROTECT
interactions in the closure allocator are included to make exploitation of
unrelated bugs more difficult.  Failure of these countermeasures to stop
exploitation of a different vulnerability is not, on its own, a security
bug in libffi.  The underlying vulnerability must still be fixed.

## Reporting security bugs

Please report suspected security vulnerabilities **privately** using
GitHub's private vulnerability reporting at
<https://github.com/libffi/libffi/security/advisories/new>.

Helpful information to include:

* the libffi version (and git commit, if building from source),
* the target architecture, operating system, and toolchain,
* configure options used to build libffi,
* a minimal reproducer (source plus the exact `ffi_cif` and call sequence),
  preferably as a standalone C program,
* the observed behavior (crash, memory corruption, information disclosure)
  and why you believe the bug is in libffi rather than in the caller's CIF
  or arguments, and
* any CVE identifier already assigned, if applicable.

If you cannot use GitHub's reporting flow, contact the maintainers listed
in `README.md` directly.  Please do not open a public GitHub issue, send
a pull request, or post on a mailing list for an unfixed vulnerability
until disclosure has been coordinated.

Routine correctness bugs, build failures, and questions should be filed as
ordinary public issues at
<https://github.com/libffi/libffi/issues>.

## Triage and fixing

This section is aimed at maintainers.

When a report is received:

1. Confirm that the bug is reproducible and that it falls within the scope
   described above.  Reports that turn out to be caller misuse should be
   closed with an explanation; if the misuse is easy to make, consider
   improving the documentation or adding diagnostics.
2. Identify the first released version that contains the vulnerable code,
   the commits that introduced it, and the set of supported branches that
   need a backport.
3. Prepare the fix on a private branch.  Avoid pushing the fix or any
   regression test that reveals the issue to the public repository until
   the embargo lifts.
4. Request a CVE identifier (see below) before public disclosure when the
   bug warrants one.
5. Coordinate the release: land the fix on `master`, backport to the
   supported release branches, cut a release, and publish the advisory
   describing the affected versions, the impact, and the fix commits.

## CVE assignment

Security bugs that affect released versions of libffi should normally
receive a [CVE identifier](https://www.cve.org/About/Overview).  CVE IDs
can be requested via GitHub's security advisory workflow when publishing
the advisory, or directly from a CVE Numbering Authority.  Include the CVE
identifier in the advisory, in the commit message of the fix, and in the
release notes for the fixed release.
