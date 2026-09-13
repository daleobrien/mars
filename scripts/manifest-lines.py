#!/usr/bin/env python3
"""Print a corpus manifest as US-separated name|url|sha256|dir lines for shell consumption."""
import json
import sys

m = json.load(open(sys.argv[1]))
for e in m["images"]:
    print("\x1f".join([e["name"], e["url"], e.get("sha256", ""), m["dir"]]))
