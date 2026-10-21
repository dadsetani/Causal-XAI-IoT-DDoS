# Publication checklist — READ BEFORE CREATING A GITHUB RELEASE

**Do not publish a GitHub Release until the Zenodo integration workflow is
settled.** An enabled GitHub-to-Zenodo integration may auto-create a new Zenodo
record version when a GitHub Release is published. Release assets are not
necessarily archived with the GitHub source snapshot.

Target: one new **version** of the existing Zenodo record, under concept DOI
`10.5281/zenodo.22280769`, with both a code snapshot and
`causal_ddos_validated_outputs.zip`. Zenodo will allocate a new version DOI
for v1.1.0; this is normal and does not require a new concept DOI.

Before publishing:

1. Confirm the source files and notebook match the revised manuscript.
2. Run `python -m py_compile src/five_seed_kaggle.py` (syntax check only).
3. Run `python tools/verify_zenodo_outputs.py <full-output-zip>` locally.
4. Upload GitHub files/commit changes to `main` without creating a Release.
5. Inspect the existing unpublished Zenodo new-version draft and its files.
6. Decide on **manual new-version publication** vs GitHub automatic ingest;
   do not trigger both for the same intended v1.1.0 release.
7. For manual publication, package the exact GitHub source snapshot and full
   output ZIP, and ensure the new version is linked to the existing concept DOI.
8. Check Zenodo creators, version metadata, license, and file availability.
9. Publish the Zenodo version; record its new version-specific DOI.
10. Update README to replace 'not yet published' with the new DOI and ZIP link.
11. Only create a GitHub Release if doing so will not trigger a duplicate
    Zenodo version; otherwise use a Git tag and link the Zenodo version.

Do not claim the new ZIP is available until the Zenodo record is published.
