"""Normalize compiler JAR envelopes, never class, source or resource payloads."""

import argparse
import os
import stat
import sys
import tempfile
import zipfile

TIMESTAMP = (2010, 1, 1, 0, 0, 0)  # Matches ijar.


def _entry_order(name):
    if name == "META-INF/":
        return (0, name)
    if name == "META-INF/MANIFEST.MF":
        return (1, name)
    return (2, name)


def canonicalize(path, *, ijar=False, preserve_member_attributes=False):
    """Atomically repack, retaining each occurrence and its compression method.

    Equal names keep their original relative order: differing duplicate payloads
    are intentionally outside the determinism guarantee. Generated full/source
    JARs can retain member platform/permission attributes; ABI defaults stay fixed.
    """
    temporary = None
    try:
        with zipfile.ZipFile(path) as source:
            entries = []
            for info in source.infolist():
                name = info.filename
                if ijar and not info.flag_bits & 0x800:
                    # ijar copies raw UTF-8 names but clears the UTF-8 flag.
                    # Preflight ALL names; never partially rewrite a legacy IJ.
                    try:
                        name = name.encode("cp437").decode("utf-8")
                    except UnicodeDecodeError:
                        print(
                            "canonicalize_java_abi_jar: leaving {!s} unchanged: "
                            "invalid UTF-8 ijar entry {!r}".format(path, name),
                            file=sys.stderr,
                        )
                        return False
                entries.append((name, info))

            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(os.path.abspath(path)), prefix=".java-abi-", suffix=".jar", delete=False
            ) as output:
                temporary = output.name
                with zipfile.ZipFile(output, "w") as target:
                    for name, original in sorted(entries, key=lambda entry: _entry_order(entry[0])):
                        info = zipfile.ZipInfo(name, TIMESTAMP)
                        if preserve_member_attributes:
                            info.create_system = original.create_system
                            info.external_attr = original.external_attr
                        else:
                            info.create_system = 3
                            info.external_attr = (0o40755 << 16 | 0x10) if info.is_dir() else (0o100644 << 16)
                        # Read the ORIGINAL ZipInfo (also for duplicates and
                        # recovered names); fresh write metadata drops extras,
                        # comments and timestamp-dependent attributes.
                        target.writestr(
                            info, source.read(original), compress_type=original.compress_type, compresslevel=6
                        )
                        if preserve_member_attributes:
                            # zipfile synthesizes permissions for a zero value.
                            # Restore it before the central directory is written.
                            info.external_attr = original.external_attr
        os.chmod(temporary, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(temporary, path)
        temporary = None
        return True
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ijar", action="store_true", help="recover ijar's unflagged UTF-8 entry names")
    parser.add_argument("jar")
    args = parser.parse_args()
    canonicalize(args.jar, ijar=args.ijar)


if __name__ == "__main__":
    main()
