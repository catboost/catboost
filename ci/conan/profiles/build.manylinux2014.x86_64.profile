{% set libc, libc_version = detect_api.detect_libc() %}

include(default)

[settings]
os.libc={{libc}}
os.libc_version={{libc_version}}
