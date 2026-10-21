#!/usr/bin/env python3
"""Check the complete Kaggle results ZIP against its embedded SHA256SUMS.txt."""
import argparse
import hashlib
import sys
import zipfile
from pathlib import PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to causal_ddos_validated_outputs.zip")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        names = set(z.namelist())
        checksum_names = [n for n in names if n.endswith("/SHA256SUMS.txt")]
        if len(checksum_names) != 1:
            raise SystemExit("Expected exactly one SHA256SUMS.txt inside archive")
        checksum_name = checksum_names[0]
        prefix = checksum_name.rsplit("/", 1)[0] + "/"
        lines = z.read(checksum_name).decode("utf-8").splitlines()
        failures = []
        listed = set()
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                failures.append(f"Malformed checksum line: {line}")
                continue
            expected, filename = parts
            if (not filename or filename.startswith("/") or ".." in PurePosixPath(filename).parts):
                failures.append(f"Unsafe filename: {filename}")
                continue
            member = prefix + filename
            listed.add(member)
            if member not in names:
                failures.append(f"Missing: {member}")
                continue
            h = hashlib.sha256()
            with z.open(member) as f:
                while True:
                    block = f.read(1024 * 1024)
                    if not block:
                        break
                    h.update(block)
            if h.hexdigest().lower() != expected.lower():
                failures.append(f"Hash mismatch: {member}")
        unlisted = {n for n in names if n.startswith(prefix) and not n.endswith("/")} - listed - {checksum_name}
        if unlisted:
            failures.extend(f"Unlisted file: {n}" for n in sorted(unlisted))
        print(f"Verified {len(listed)} listed files; extra unlisted files: {len(unlisted)}")
        for error in failures:
            print("ERROR:", error)
        if failures:
            sys.exit(1)
        print("PASS: ZIP contents match SHA256SUMS.txt")

if __name__ == "__main__":
    main()
