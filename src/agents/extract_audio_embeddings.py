"""Extract deep audio embeddings with AST (Audio Spectrogram Transformer).

AST is pretrained on AudioSet, so it brings general acoustic priors that our
~66 asthma patients could never teach a model from scratch. We use it frozen
as a feature extractor - no fine-tuning, which would overfit at this sample
size - and train a light classifier head on the embeddings.

Two details that matter:

* Sample rate. SPRSound is 8 kHz; AST expects 16 kHz. We upsample. This
  invents no information (the 4 kHz Nyquist ceiling stands) but it puts the
  audio in the domain the pretrained filters expect.

* Clip length. AST's input window is 10.24 s and our recordings are 15.36 s.
  Truncating would discard a third of every recording, so we embed
  overlapping windows and mean-pool them into one vector per recording.

Output: data/processed/audio_embeddings.npz (embeddings + record_stem order)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "sprsound_manifest.csv"
OUT = ROOT / "data" / "processed" / "audio_embeddings.npz"

# Prefer the locally downloaded weights; the hub id is the fallback.
_LOCAL = ROOT / "models" / "ast"
MODEL_ID = str(_LOCAL) if (_LOCAL / "model.safetensors").exists() else (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
TARGET_SR = 16000
WINDOW_S = 10.24
HOP_S = 5.12
BATCH = 8


def windows(y, sr):
    """Overlapping fixed-length windows covering the whole recording."""
    n = int(WINDOW_S * sr)
    hop = int(HOP_S * sr)
    if len(y) <= n:
        return [np.pad(y, (0, n - len(y)))]
    out = []
    for start in range(0, len(y) - n + 1, hop):
        out.append(y[start:start + n])
    tail = y[-n:]
    if not np.array_equal(tail, out[-1]):
        out.append(tail)
    return out


def main() -> None:
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, ASTModel

    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST}")

    man = pd.read_csv(MANIFEST).dropna(subset=["disease"]).reset_index(drop=True)
    print(f"recordings to embed: {len(man):,}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTModel.from_pretrained(MODEL_ID).to(device).eval()
    dim = model.config.hidden_size
    print(f"model: {MODEL_ID} (hidden size {dim})")

    # Resume support: this takes ~1 h, longer than any single run may last, so
    # progress is checkpointed and completed rows are skipped on restart.
    ckpt = OUT.with_suffix(".partial.npz")
    embeddings = np.zeros((len(man), dim), dtype=np.float32)
    done = np.zeros(len(man), dtype=bool)
    if ckpt.exists():
        prev = np.load(ckpt, allow_pickle=True)
        if prev["embeddings"].shape == embeddings.shape:
            embeddings, done = prev["embeddings"], prev["done"]
            print(f"resuming: {done.sum():,}/{len(man):,} already embedded")

    stems, failed = list(man["record_stem"]), 0
    t0 = time.time()
    start_done = int(done.sum())

    for i, rec in enumerate(man.itertuples(index=False)):
        if done[i]:
            continue
        try:
            y, _ = librosa.load(ROOT / rec.wav_path, sr=TARGET_SR, mono=True)
            chunks = windows(y, TARGET_SR)
            vecs = []
            for j in range(0, len(chunks), BATCH):
                batch = chunks[j:j + BATCH]
                inputs = fe(batch, sampling_rate=TARGET_SR, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs).last_hidden_state.mean(dim=1)
                vecs.append(out.cpu().numpy())
            embeddings[i] = np.concatenate(vecs, axis=0).mean(axis=0)
            done[i] = True
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"  ! {rec.record_stem}: {exc}")

        n_done = int(done.sum())
        if n_done % 100 == 0 and done[i]:
            np.savez(ckpt, embeddings=embeddings, done=done)
            el = time.time() - t0
            rate = max(n_done - start_done, 1) / max(el, 1e-6)
            eta = (len(man) - n_done) / rate / 60
            print(f"  {n_done:,}/{len(man):,}  {rate:.1f} rec/s  ETA {eta:.1f} min")

    np.savez(ckpt, embeddings=embeddings, done=done)
    if done.all():
        np.savez_compressed(OUT, embeddings=embeddings, record_stem=np.array(stems))
        ckpt.unlink(missing_ok=True)
        print(f"\ncomplete: {done.sum():,} recordings, wrote {OUT} shape={embeddings.shape}")
    else:
        print(f"\npartial: {done.sum():,}/{len(man):,} embedded ({failed} failed) - "
              f"rerun to continue")


if __name__ == "__main__":
    main()
