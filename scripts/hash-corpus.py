#!/usr/bin/env python3
"""Record sha256 and byte size for every fetched corpus image into its manifest.

Run once after the first fetch of a new corpus set, then commit the manifest. From then
on fetch-corpus.sh verifies rather than trusts, and the manifest hash carried in every
result row's provenance block (§M7) makes a changed image set impossible to miss.
"""
import hashlib
import json
import os
import sys

manifest_path = sys.argv[1] if len(sys.argv) > 1 else "corpus/kodak.manifest.json"
with open(manifest_path) as f:
    m = json.load(f)

missing = []
for e in m["images"]:
    path = os.path.join(m["dir"], e["name"])
    if not os.path.exists(path):
        missing.append(e["name"])
        continue
    with open(path, "rb") as f:
        e["sha256"] = hashlib.sha256(f.read()).hexdigest()
    e["bytes"] = os.path.getsize(path)

if missing:
    print(
        f"refusing to write a partial manifest; {len(missing)} missing: {missing}",
        file=sys.stderr,
    )
    sys.exit(1)

with open(manifest_path, "w") as f:
    json.dump(m, f, indent=2)
    f.write("\n")
print(f"pinned {len(m['images'])} hashes in {manifest_path}")
