# Results and archive boundary

Existing files in this directory are historical v1.0.0 artifacts.
The files prefixed `five_seed_` belong to the major-revision experiments.
The new 108,576,782-byte ZIP must **not** be committed to GitHub.

- `five_seed_validated_all_results.csv`: supplied 30-row aggregate results
- `five_seed_SHA256SUMS.txt`: 121 SHA-256 checksums for files *inside* the
  Zenodo ZIP, relative to its `validated_outputs/` directory

The original filenames `validated_all_results.csv` and `SHA256SUMS.txt`
inside the ZIP remain unchanged. The `five_seed_` prefixes in GitHub are
used solely to prevent confusion with the historical v1.0.0 files.

After publishing Zenodo v1.1.0, update README with the exact version DOI.
