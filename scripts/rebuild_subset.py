"""Rebuild the 900-image subset (100 per class) used in the paper from the Kaggle wound dataset.

usage: python scripts/rebuild_subset.py <kaggle_dir> <out_dir>

<kaggle_dir> contains one sub-folder per class (Abrasions, Bruises, ...), as in the Kaggle archive.
Every copied file is checked against the SHA-256 in data/subset_manifest.csv.
"""
import csv
import hashlib
import shutil
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "subset_manifest.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(kaggle_dir: str, out_dir: str) -> int:
    src, dst = Path(kaggle_dir), Path(out_dir)
    ok, bad = 0, []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = src / row["class"] / row["filename"]
            if not source.exists():
                bad.append((row["class"], row["filename"], "missing"))
                continue
            if sha256(source) != row["sha256"]:
                bad.append((row["class"], row["filename"], "checksum differs"))
                continue
            target = dst / row["class"] / row["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            ok += 1
    print(f"copied {ok} files to {dst}")
    for cls, name, why in bad[:20]:
        print(f"  problem: {cls}/{name}: {why}")
    if bad:
        print(f"{len(bad)} files could not be reproduced")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
