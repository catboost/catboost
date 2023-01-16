version_info = (5, 1, 1)

# unlike `.dev`, alpha, beta and rc _must not_ have dots,
# or the wheel and tgz won't look to pip like the same version.

__version__ = (
    ".".join(map(str, version_info))
    .replace(".b", "b")
    .replace(".a", "a")
    .replace(".rc", "rc")
)
assert ".b" not in __version__
assert ".a" not in __version__
assert ".rc" not in __version__
