from lib.nots.semver import Version

# it is crucial to keep this array sorted
SUPPORTED_NODE_VERSIONS = [
    Version.from_str("12.18.4"),
    Version.from_str("12.22.12"),
    Version.from_str("14.21.1"),
    Version.from_str("16.18.1"),
    Version.from_str("18.12.1")
]

DEFAULT_NODE_VERSION = SUPPORTED_NODE_VERSIONS[0]
