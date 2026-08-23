"""Extract the unique SPRSound recordings from the downloaded ZIP.

The archive stores each recording several times over, once per BioCAS
challenge split (2022/2023/2024/2025, Classification, Detection,
Compression) - 20,457 entries for 6,567 distinct recordings. Extracting
everything would waste ~2.4 GB, which matters on a nearly full disk.

We therefore keep one copy of each recording, flattened into audio/, plus
its JSON annotation and the patient-summary CSVs.
"""

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = Path.home() / "Downloads" / "SPRSound-main.zip"
DEST = ROOT / "data" / "raw" / "sprsound"
AUDIO = DEST / "audio"
ANNOT = DEST / "annotations"
SUMMARY = DEST / "Patient Summary"

MIN_FREE_BYTES = 400 * 1024**2  # keep headroom so we never fill the disk


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def main() -> None:
    if not ZIP.exists():
        sys.exit(f"missing {ZIP}")

    for d in (AUDIO, ANNOT, SUMMARY):
        d.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(ZIP)
    infos = zf.infolist()

    # Pick one source entry per basename, preferring the shortest path so the
    # choice is deterministic across runs.
    best: dict[str, zipfile.ZipInfo] = {}
    summaries: list[zipfile.ZipInfo] = []
    for info in infos:
        if info.is_dir():
            continue
        name = Path(info.filename).name
        low = info.filename.lower()
        if low.endswith(".csv") and "patient summary" in low:
            summaries.append(info)
            continue
        if not (low.endswith(".wav") or low.endswith(".json")):
            continue
        prev = best.get(name)
        if prev is None or len(info.filename) < len(prev.filename):
            best[name] = info

    wavs = {k: v for k, v in best.items() if k.lower().endswith(".wav")}
    jsons = {k: v for k, v in best.items() if k.lower().endswith(".json")}
    planned = sum(v.file_size for v in best.values())

    print(f"unique recordings : {len(wavs):,}")
    print(f"unique annotations: {len(jsons):,}")
    print(f"bytes to write    : {planned / 1024**3:.2f} GB")
    print(f"free space        : {free_bytes(DEST) / 1024**3:.2f} GB")

    if free_bytes(DEST) - planned < MIN_FREE_BYTES:
        sys.exit("not enough free disk space - aborting before writing anything")

    written = skipped = 0
    for n, (name, info) in enumerate(sorted(best.items()), start=1):
        target = (AUDIO if name.lower().endswith(".wav") else ANNOT) / name
        if target.exists() and target.stat().st_size == info.file_size:
            skipped += 1
            continue
        if n % 500 == 0:
            if free_bytes(DEST) < MIN_FREE_BYTES:
                sys.exit(f"\nstopped at {n:,}: disk space ran low")
            print(f"  {n:,}/{len(best):,}")
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 256)
        written += 1

    for info in summaries:
        target = SUMMARY / Path(info.filename).name
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)

    n_wav = len(list(AUDIO.glob("*.wav")))
    n_json = len(list(ANNOT.glob("*.json")))
    n_csv = len(list(SUMMARY.glob("*.csv")))
    print(f"\nwrote {written:,} files ({skipped:,} already present)")
    print(f"audio/       {n_wav:,} wav")
    print(f"annotations/ {n_json:,} json")
    print(f"summaries    {n_csv}")
    print(f"free space remaining: {free_bytes(DEST) / 1024**3:.2f} GB")
    print(f"\ndest: {DEST}")


if __name__ == "__main__":
    main()
