# Releases

## 0.13.0

### Breakage

- Make `try_lock_*` return `std::io::Result<bool>`, which is compatible with the upcoming `std::fs::File::try_lock*` in `std`.
