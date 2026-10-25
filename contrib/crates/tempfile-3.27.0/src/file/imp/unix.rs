use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io;

use crate::util;
use std::path::Path;

use {
    rustix::fs::{rename, unlink},
    std::fs::hard_link,
};

pub fn create_named(
    path: &Path,
    open_options: &mut OpenOptions,
    #[cfg_attr(target_os = "wasi", allow(unused))] permissions: Option<&std::fs::Permissions>,
) -> io::Result<File> {
    open_options.read(true).write(true).create_new(true);

    #[cfg(not(target_os = "wasi"))]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        open_options.mode(permissions.map(|p| p.mode()).unwrap_or(0o600));
    }

    open_options.open(path)
}

fn create_unlinked(path: &Path) -> io::Result<File> {
    let tmp;
    // shadow this to decrease the lifetime. It can't live longer than `tmp`.
    let mut path = path;
    if !path.is_absolute() {
        let cur_dir = std::env::current_dir()?;
        tmp = cur_dir.join(path);
        path = &tmp;
    }

    let f = create_named(path, &mut OpenOptions::new(), None)?;
    // don't care whether the path has already been unlinked,
    // but perhaps there are some IO error conditions we should send up?
    let _ = fs::remove_file(path);
    Ok(f)
}

#[cfg(target_os = "linux")]
pub fn create(dir: &Path) -> io::Result<File> {
    use rustix::{fs::OFlags, io::Errno};
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(OFlags::TMPFILE.bits() as i32) // do not mix with `create_new(true)`
        .open(dir)
        .or_else(|e| {
            match Errno::from_io_error(&e) {
                // These are the three "not supported" error codes for O_TMPFILE.
                Some(Errno::OPNOTSUPP) | Some(Errno::ISDIR) | Some(Errno::NOENT) => {
                    create_unix(dir)
                }
                _ => Err(e),
            }
        })
}

#[cfg(not(target_os = "linux"))]
pub fn create(dir: &Path) -> io::Result<File> {
    create_unix(dir)
}

fn create_unix(dir: &Path) -> io::Result<File> {
    util::create_helper(
        dir,
        OsStr::new(".tmp"),
        OsStr::new(""),
        crate::NUM_RAND_CHARS,
        |path| create_unlinked(&path),
    )
}

pub fn reopen(file: &File, path: &Path) -> io::Result<File> {
    let new_file = OpenOptions::new().read(true).write(true).open(path)?;
    let old_meta = rustix::fs::fstat(file)?;
    let new_meta = rustix::fs::fstat(&new_file)?;
    if old_meta.st_dev != new_meta.st_dev || old_meta.st_ino != new_meta.st_ino {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "original tempfile has been replaced",
        ));
    }
    Ok(new_file)
}

pub fn persist(old_path: &Path, new_path: &Path, overwrite: bool) -> io::Result<()> {
    if overwrite {
        rename(old_path, new_path)?;
    } else {
        // On Linux, apple and redox operating systems, use `renameat_with` to avoid overwriting an
        // existing name, if the kernel and the filesystem support it.
        #[cfg(any(
            target_os = "android",
            target_os = "linux",
            target_os = "macos",
            target_os = "ios",
            target_os = "tvos",
            target_os = "visionos",
            target_os = "watchos",
            target_os = "redox",
        ))]
        {
            use rustix::fs::{renameat_with, RenameFlags, CWD};
            use rustix::io::Errno;
            use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

            static NOSYS: AtomicBool = AtomicBool::new(false);
            if !NOSYS.load(Relaxed) {
                match renameat_with(CWD, old_path, CWD, new_path, RenameFlags::NOREPLACE) {
                    Ok(()) => return Ok(()),
                    Err(Errno::NOSYS) => NOSYS.store(true, Relaxed),
                    Err(Errno::INVAL) => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }

        // Otherwise use `hard_link` to create the new filesystem name, which
        // will fail if the name already exists, and then `unlink` to remove
        // the old name.
        hard_link(old_path, new_path)?;

        // Ignore unlink errors. Can we do better?
        let _ = unlink(old_path);
    }
    Ok(())
}

pub fn keep(_: &Path) -> io::Result<()> {
    Ok(())
}
