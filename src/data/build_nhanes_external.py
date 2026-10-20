"""
Build an independent pediatric cohort from NHANES 2015-2016 (I) and 2017-2018 (J)
for TEMPORAL EXTERNAL VALIDATION of the Clinical Agent.

Why these cycles and not 2013-2014
----------------------------------
NHANES retired the RDQ respiratory module and the SPX spirometry component after
the 2011-2012 cycle. No later cycle carries wheeze, nocturnal cough or FEV1/FVC,
so the full 21-feature agent CANNOT be tested out-of-era. What can be tested is
the subset of variables collected identically in both eras. That subset is what
this script assembles; the development cohort is re-fitted on the same subset so
that the comparison is like-for-like.

Smoking-item recoding
---------------------
SMD410/SMD415 (2007-2012) were replaced by SMD460/SMD470 in 2015+. SMD470
("number of people who smoke inside the house") is the direct analogue of
SMD415, and household_smoker is derived from it rather than from SMD460, which
counts household smokers regardless of where they smoke.

Label, leakage and skip-pattern handling follow build_nhanes.py exactly.
"""
import pandas as pd, numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "nhanes"
OUT = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES = {"I": "2015-2016", "J": "2017-2018"}

KEEP = {
    "DEMO": ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDHHSIZ"],
    "MCQ":  ["SEQN", "MCQ010", "MCQ035", "MCQ300B"],
    "SMQFAM": ["SEQN", "SMD460", "SMD470"],
    "BMX":  ["SEQN", "BMXBMI", "BMXHT"],
    "COT":  ["SEQN", "LBXCOT"],
}

MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}

# Features present and identically defined in BOTH eras.
TRANSPORTABLE = ["sex", "age", "ethnicity", "income_poverty_ratio",
                 "household_size", "family_hx_asthma", "household_smoker",
                 "n_household_smokers", "bmi", "height", "log_cotinine"]


def load(stub, sfx):
    f = RAW / f"{stub}_{sfx}.xpt"
    if not f.exists():
        print(f"  ! missing {f.name}")
        return None
    df = pd.read_sas(f, format="xport")
    cols = [c for c in KEEP[stub] if c in df.columns]
    absent = set(KEEP[stub]) - set(cols)
    if absent:
        print(f"  ! {stub}_{sfx}: absent columns {sorted(absent)}")
    return df[cols]


def build():
    frames = []
    for sfx, label in CYCLES.items():
        print(f"[{label}]")
        d = load("DEMO", sfx)
        if d is None:
            continue
        for stub in ["MCQ", "SMQFAM", "BMX", "COT"]:
            o = load(stub, sfx)
            if o is not None:
                d = d.merge(o, on="SEQN", how="left")
        d["cycle"] = label
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    print(f"\npooled participants: {len(df):,}")

    df = df[df["RIDAGEYR"].between(6, 17)].copy()
    print(f"aged 6-17:           {len(df):,}")

    for c in ["MCQ010", "MCQ035", "MCQ300B", "SMD460", "SMD470"]:
        df[c] = df[c].where(~df[c].isin(MISSING_CODES), np.nan)

    df["asthma_status"] = np.select(
        [df.MCQ010.eq(1) & df.MCQ035.eq(1),
         df.MCQ010.eq(1) & df.MCQ035.eq(2),
         df.MCQ010.eq(2)],
        ["current", "former", "never"], default="unknown")

    model = df[df.asthma_status.isin(["current", "never"])].copy()
    model["current_asthma"] = (model.asthma_status == "current").astype(int)
    model["MCQ300B"] = model.MCQ300B.map({1: 1, 2: 0})

    # SMD470 counts people who smoke INSIDE the house -> the SMD415 analogue.
    model["household_smoker"] = (model.SMD470.fillna(0) > 0).astype(float)
    model.loc[model.SMD470.isna() & model.SMD460.isna(), "household_smoker"] = np.nan
    model["n_household_smokers"] = model.SMD470

    model = model.rename(columns={
        "RIAGENDR": "sex", "RIDAGEYR": "age", "RIDRETH1": "ethnicity",
        "INDFMPIR": "income_poverty_ratio", "DMDHHSIZ": "household_size",
        "MCQ300B": "family_hx_asthma", "BMXBMI": "bmi", "BMXHT": "height",
    })
    model["log_cotinine"] = np.log1p(model.LBXCOT)
    # Same skip-pattern rule as development: no smoker inside -> a known zero.
    model.loc[model.household_smoker.eq(0), "n_household_smokers"] = 0

    model = model[["SEQN", "cycle"] + TRANSPORTABLE + ["current_asthma"]]
    model.to_csv(OUT / "clinical_agent_external.csv", index=False)

    n, pos = len(model), int(model.current_asthma.sum())
    print(f"\nexternal cohort:     {n:,}")
    print(f"  current asthma:    {pos:,} ({pos/n:.1%})")
    print(f"  never asthma:      {n-pos:,}")
    print(f"  former (excluded): {(df.asthma_status=='former').sum():,}")
    print("\nby cycle:")
    print(model.groupby("cycle").current_asthma.agg(["size", "sum", "mean"]).to_string())
    print("\nmissingness (%):")
    print(model[TRANSPORTABLE].isna().mean().mul(100).round(1).sort_values(ascending=False).to_string())
    print(f"\nwrote {OUT/'clinical_agent_external.csv'}")
    return model


if __name__ == "__main__":
    build()
