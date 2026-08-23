"""Extract interpretable acoustic features from SPRSound recordings.

Design notes
------------
SPRSound audio is 8 kHz mono (Nyquist 4 kHz), which is adequate: the
clinically relevant adventitious sounds sit well below that. Wheeze is a
continuous musical sound, conventionally >100 ms, with dominant energy
roughly in the 100-1000 Hz band and a tonal (peaked) spectrum. Crackles are
short, broadband and transient. The feature set below is built to separate
those two regimes, and every feature is nameable in a clinical explanation -
which matters because these features are what SHAP will attribute over in
the explanation-fusion layer.

Deep embeddings would likely score higher, but with ~71 asthma patients they
would overfit, they need a GPU, and they cannot be named in an explanation.
Handcrafted features are the defensible choice at this sample size.

Output: data/processed/audio_features.csv (one row per recording)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "sprsound_manifest.csv"
OUT = ROOT / "data" / "processed" / "audio_features.csv"

SR = 8000
N_FFT = 512
HOP = 128

# Physiologically motivated bands (Hz). Wheeze energy concentrates in the
# mid bands; the lowest band is dominated by heart sounds and handling noise.
BANDS = {
    "b_60_150": (60, 150),
    "b_150_300": (150, 300),
    "b_300_600": (300, 600),
    "b_600_1200": (600, 1200),
    "b_1200_2400": (1200, 2400),
    "b_2400_4000": (2400, 4000),
}


def _stats(name, x):
    """Summarise a per-frame descriptor into a few named scalars."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or not np.isfinite(x).any():
        return {f"{name}_{s}": np.nan for s in ("mean", "std", "p90")}
    return {
        f"{name}_mean": float(np.nanmean(x)),
        f"{name}_std": float(np.nanstd(x)),
        f"{name}_p90": float(np.nanpercentile(x, 90)),
    }


def tonality(S_pow, freqs):
    """Wheeze proxy: how peaked the spectrum is inside the wheeze band.

    For each frame, the ratio of the single strongest bin to the band mean.
    A pure tone (wheeze) gives a high ratio; broadband noise gives ~1.
    Returned per frame so the caller can summarise over time.
    """
    band = (freqs >= 100) & (freqs <= 1600)
    if not band.any():
        return np.array([])
    Sb = S_pow[band, :]
    mean = Sb.mean(axis=0)
    peak = Sb.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean > 0, peak / mean, np.nan)


def extract_one(path: Path) -> dict:
    import librosa

    y, sr = librosa.load(path, sr=SR, mono=True)
    if y.size < N_FFT:
        return {}
    # Remove DC offset and normalise level so features reflect spectral shape
    # rather than how hard the stethoscope was pressed.
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    S_pow = S**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    feats = {"duration_s": float(len(y) / sr)}

    # --- band energy ratios (scale-free, so comparable across recordings)
    total = S_pow.sum() + 1e-12
    for name, (lo, hi) in BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        feats[f"{name}_ratio"] = float(S_pow[sel, :].sum() / total)

    # --- spectral shape descriptors
    feats.update(_stats("centroid", librosa.feature.spectral_centroid(S=S, sr=sr)))
    feats.update(_stats("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    feats.update(_stats("rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)))
    feats.update(_stats("flatness", librosa.feature.spectral_flatness(S=S)))
    feats.update(_stats("zcr", librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)))
    feats.update(_stats("rms", librosa.feature.rms(S=S, frame_length=N_FFT)))

    # --- wheeze tonality
    feats.update(_stats("tonality", tonality(S_pow, freqs)))

    # --- MFCCs: compact timbre summary
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))

    # --- temporal dynamics: crackles are transient, wheeze is sustained
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(S_pow), sr=sr, hop_length=HOP)
    feats.update(_stats("onset_strength", onset))
    feats["onset_rate_per_s"] = float(
        len(librosa.onset.onset_detect(onset_envelope=onset, sr=sr, hop_length=HOP))
        / max(feats["duration_s"], 1e-6)
    )

    return feats


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST} - run src/data/build_sprsound.py first")

    man = pd.read_csv(MANIFEST)
    man = man.dropna(subset=["disease"])
    print(f"extracting features for {len(man):,} labelled recordings...")

    rows, failed = [], 0
    for n, rec in enumerate(man.itertuples(index=False), start=1):
        path = ROOT / rec.wav_path
        try:
            f = extract_one(path)
        except Exception as exc:  # noqa: BLE001 - keep going, report at end
            failed += 1
            if failed <= 3:
                print(f"  ! {path.name}: {exc}")
            continue
        if not f:
            failed += 1
            continue
        f["record_stem"] = rec.record_stem
        rows.append(f)
        if n % 250 == 0:
            print(f"  {n:,}/{len(man):,}")

    if not rows:
        sys.exit("no features extracted")

    feats = pd.DataFrame(rows)
    out = man.merge(feats, on="record_stem", how="inner")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    n_feat = len(feats.columns) - 1
    print(f"\nextracted {n_feat} features for {len(out):,} recordings ({failed} failed)")
    print(f"patients: {out['patient_id'].nunique():,}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
