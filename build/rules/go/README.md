# PEERDIR policy for Go

## IMPORTANT
None of files in this directory can be moved or split into multiple files without explicit OK from go-com@yandex-team.ru
Doing otherwise will break vendoring tooling.

 * `contrib.policy` - special exceptions for patched contribs
 * `migrations.yaml` - exceptions for linters (no additions allowed except when implementing new linters or updating old ones)
 * `vendor.policy` - general PEERDIR policy for contribs
