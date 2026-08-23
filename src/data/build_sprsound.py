"""Build a patient-level manifest for the SPRSound audio agent.

Joins every .wav recording to its patient-level diagnosis from the three
patient-summary files. Deduplicates recordings by basename because the
BioCAS20xx/ and Classification/ directories share identical git trees
(the same recordings appear under several challenge splits).

Filename convention: {patient_id}_{age}_{sex}_{position}_{record_id}.wav
    sex: 0 = female, 1 = male   (per SPRSound README)
    position: p1..p4 chest auscultation site

Output: data/processed/sprsound_manifest.csv
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "sprsound"
SUMMARY_DIR = RAW / "Patient Summary"
OUT = ROOT / "data" / "processed" / "sprsound_manifest.csv"

FNAME_RE = re.compile(
    r"^(?P<patient_id>\d+)_(?P<age>[\d.]+)_(?P<sex>[01])_p(?P<position>\d)_(?P<record_id>\d+)$"
)

# Collapse the raw diagnosis strings into the analysis classes.
DISEASE_MAP = {
    "asthma": "asthma",
    "control group": "control",
    "pneumonia (non-severe)": "pneumonia",
    "pneumonia (severe)": "pneumonia",
    "bronchitis": "bronchitis",
}


def load_patient_labels() -> pd.DataFrame:
    """Union the three patient-summary files into one patient -> diagnosis table."""
    frames = []
    for path in sorted(SUMMARY_DIR.glob("*.csv")):
        df = pd.read_csv(path, dtype=str)
        # SPRSound_patient_summary.csv carries an unnamed index column.
        df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
        missing = {"patient_num", "disease"} - set(df.columns)
        if missing:
            print(f"  ! {path.name}: missing {sorted(missing)}, skipped")
            continue
        df["source"] = path.stem
        frames.append(df)

    if not frames:
        sys.exit("no patient summary files found")

    allp = pd.concat(frames, ignore_index=True)
    allp["patient_id"] = allp["patient_num"].str.strip().str.lstrip("0")
    allp["disease_raw"] = allp["disease"].str.strip()
    allp["disease"] = (
        allp["disease_raw"].str.lower().map(DISEASE_MAP).fillna("other")
    )
    allp = allp[allp["disease_raw"] != "-"]

    # A patient may appear in several challenge years; keep one row, but flag
    # any patient whose diagnosis is inconsistent across files.
    nunique = allp.groupby("patient_id")["disease"].nunique()
    conflicts = nunique[nunique > 1].index.tolist()
    if conflicts:
        print(f"  ! {len(conflicts)} patients have conflicting labels across files; dropped")
        allp = allp[~allp["patient_id"].isin(conflicts)]

    return allp.drop_duplicates(subset="patient_id")[
        ["patient_id", "disease", "disease_raw"]
    ]


def collect_recordings() -> pd.DataFrame:
    """Find every .wav under the dataset, deduplicated by basename."""
    rows = {}
    unparsed = 0
    for wav in RAW.rglob("*.wav"):
        stem = wav.stem
        m = FNAME_RE.match(stem)
        if not m:
            unparsed += 1
            continue
        if stem in rows:  # same recording under another challenge split
            continue
        d = m.groupdict()
        # Annotations are extracted into a sibling annotations/ directory, but
        # fall back to a JSON alongside the wav for other layouts.
        cand = [RAW / "annotations" / f"{stem}.json", wav.with_suffix(".json")]
        js = next((p for p in cand if p.exists()), None)
        rows[stem] = {
            "record_stem": stem,
            "patient_id": d["patient_id"].lstrip("0"),
            "age_years": float(d["age"]),
            "sex": "male" if d["sex"] == "1" else "female",
            "position": f"p{d['position']}",
            "wav_path": str(wav.relative_to(ROOT)),
            "json_path": str(js.relative_to(ROOT)) if js else "",
        }
    if unparsed:
        print(f"  ! {unparsed} wav files did not match the filename convention")
    return pd.DataFrame(rows.values())


def attach_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the record-level annotation (Normal/CAS/DAS/...) from each JSON."""
    record_ann, has_wheeze = [], []
    for rel in df["json_path"]:
        if not rel:
            record_ann.append("")
            has_wheeze.append(pd.NA)
            continue
        try:
            ann = json.loads((ROOT / rel).read_text())
        except (OSError, json.JSONDecodeError):
            record_ann.append("")
            has_wheeze.append(pd.NA)
            continue
        record_ann.append(ann.get("record_annotation", ""))
        events = ann.get("event_annotation", []) or []
        has_wheeze.append(any("wheeze" in e.get("type", "").lower() for e in events))
    df["record_annotation"] = record_ann
    df["has_wheeze_event"] = has_wheeze
    return df


def main() -> None:
    if not SUMMARY_DIR.exists():
        sys.exit(f"missing {SUMMARY_DIR} - is the dataset downloaded?")

    print("loading patient labels...")
    labels = load_patient_labels()
    print(f"  patients with a diagnosis: {len(labels):,}")

    print("collecting recordings...")
    recs = collect_recordings()
    print(f"  unique recordings: {len(recs):,}")
    if recs.empty:
        sys.exit("no recordings found - the audio checkout has not completed")

    print("reading event annotations...")
    recs = attach_event_labels(recs)

    man = recs.merge(labels, on="patient_id", how="left")
    unlabelled = man["disease"].isna().sum()
    if unlabelled:
        print(f"  ! {unlabelled:,} recordings have no patient-level diagnosis")

    # Binary target for the audio agent: asthma vs non-asthma (labelled only).
    man["is_asthma"] = (man["disease"] == "asthma").astype("Int64")
    man.loc[man["disease"].isna(), "is_asthma"] = pd.NA

    OUT.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT, index=False)

    labelled = man.dropna(subset=["disease"])
    print("\nrecordings per diagnosis:")
    print(labelled["disease"].value_counts().to_string())
    print("\npatients per diagnosis:")
    print(
        labelled.drop_duplicates("patient_id")["disease"].value_counts().to_string()
    )
    print("\nrecordings per patient:")
    per = labelled.groupby("patient_id").size()
    print(f"  median {per.median():.0f}, min {per.min()}, max {per.max()}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
