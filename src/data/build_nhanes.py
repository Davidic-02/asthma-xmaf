"""
Build the pediatric clinical cohort for the Clinical Agent from NHANES.

Cohort: children aged 6-17 (lower bound set by MCQ300B family-history eligibility).
Cycles: 2007-2008 (E), 2009-2010 (F), 2011-2012 (G) -- the last cycles carrying
the RDQ respiratory module (wheeze/nocturnal-cough), which was discontinued after 2011-2012.

Label
-----
current_asthma = 1  if MCQ010 == 1 (ever dx) and MCQ035 == 1 (still has)
               = 0  if MCQ010 == 2 (never dx)
Former asthma (MCQ010==1 & MCQ035==2) is written out but EXCLUDED from the
primary binary task; it is kept for a sensitivity analysis.

Leakage
-------
MCQ025/MCQ040/MCQ050 are consequences of an existing diagnosis, not predictors of
it. They are deliberately NOT features. Including them inflates accuracy and is a
common flaw in published asthma-ML papers.
"""
import pandas as pd, numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "nhanes"
OUT = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CYCLES = {"E": "2007-2008", "F": "2009-2010", "G": "2011-2012"}

KEEP = {
    "DEMO": ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "INDFMPIR", "DMDHHSIZ"],
    "MCQ":  ["SEQN", "MCQ010", "MCQ035", "MCQ300B", "MCQ080"],
    "RDQ":  ["SEQN", "RDQ070", "RDQ080", "RDQ100", "RDQ140"],
    "SMQFAM": ["SEQN", "SMD410", "SMD415"],
    "BMX":  ["SEQN", "BMXBMI", "BMDBMIC", "BMXHT"],
    # Spirometry: the physiological gold-standard test for airway obstruction.
    # Ages 6-79, so it covers the whole cohort. SPDBRONC (selected for
    # bronchodilator) is deliberately NOT taken -- it is a derived clinical
    # decision flag, and post-bronchodilator reversibility forms part of the
    # diagnostic criterion itself.
    "SPX":  ["SEQN", "SPXNFEV1", "SPXNFVC", "SPXNPEF", "SPXNSTAT", "SPDNACC"],
    # Serum cotinine: objective biomarker of tobacco-smoke exposure.
    "COTNAL": ["SEQN", "LBXCOT"],
}

# NHANES codes 7/9 (and 77/99, 777/999) as Refused/Don't know -> missing.
MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}


def load(stub: str, sfx: str) -> pd.DataFrame | None:
    f = RAW / f"{stub}_{sfx}.xpt"
    if not f.exists():
        print(f"  ! missing {f.name}")
        return None
    df = pd.read_sas(f, format="xport")
    cols = [c for c in KEEP[stub] if c in df.columns]
    missing = set(KEEP[stub]) - set(cols)
    if missing:
        print(f"  ! {stub}_{sfx}: absent columns {sorted(missing)}")
    return df[cols]


def clean_codes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].where(~df[c].isin(MISSING_CODES), np.nan)
    return df


def build() -> pd.DataFrame:
    frames = []
    for sfx, label in CYCLES.items():
        print(f"[{label}]")
        d = load("DEMO", sfx)
        if d is None:
            continue
        for stub in ["MCQ", "RDQ", "SMQFAM", "BMX", "SPX", "COTNAL"]:
            o = load(stub, sfx)
            if o is not None:
                d = d.merge(o, on="SEQN", how="left")
        d["cycle"] = label
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    print(f"\npooled participants: {len(df):,}")

    df = df[df["RIDAGEYR"].between(6, 17)].copy()
    print(f"aged 6-17:           {len(df):,}")

    coded = ["MCQ010", "MCQ035", "MCQ300B", "MCQ080",
             "RDQ070", "RDQ100", "RDQ140", "SMD410"]
    df = clean_codes(df, coded)

    df["asthma_status"] = np.select(
        [df.MCQ010.eq(1) & df.MCQ035.eq(1),
         df.MCQ010.eq(1) & df.MCQ035.eq(2),
         df.MCQ010.eq(2)],
        ["current", "former", "never"], default="unknown")

    df.to_csv(OUT / "nhanes_pediatric_full.csv", index=False)

    model = df[df.asthma_status.isin(["current", "never"])].copy()
    model["current_asthma"] = (model.asthma_status == "current").astype(int)

    # 1=Yes / 2=No  ->  1/0
    for c in ["MCQ300B", "MCQ080", "RDQ070", "RDQ100", "RDQ140", "SMD410"]:
        if c in model.columns:
            model[c] = model[c].map({1: 1, 2: 0})

    model = model.rename(columns={
        "RIAGENDR": "sex", "RIDAGEYR": "age", "RIDRETH1": "ethnicity",
        "INDFMPIR": "income_poverty_ratio", "DMDHHSIZ": "household_size",
        "MCQ300B": "family_hx_asthma", "MCQ080": "told_overweight",
        "RDQ070": "wheeze_12mo", "RDQ080": "wheeze_attacks",
        "RDQ100": "exercise_wheeze", "RDQ140": "nocturnal_dry_cough",
        "SMD410": "household_smoker", "SMD415": "n_household_smokers",
        "BMXBMI": "bmi", "BMDBMIC": "bmi_category", "BMXHT": "height",
        "SPXNFEV1": "fev1", "SPXNFVC": "fvc", "SPXNPEF": "pef",
    })

    # --- Spirometry derivation ---------------------------------------------
    # Keep only manoeuvres meeting ATS acceptability; otherwise the values are
    # noise from a child who could not perform the test properly.
    if "SPXNSTAT" in model.columns:
        bad = model.SPXNSTAT.ne(1) | model.SPDNACC.fillna(0).lt(1)
        model.loc[bad, ["fev1", "fvc", "pef"]] = np.nan
        model = model.drop(columns=["SPXNSTAT", "SPDNACC"])
    model["fev1_fvc_ratio"] = model.fev1 / model.fvc
    # Height-normalised flows: lung volume scales with stature in children, so
    # raw mL is dominated by growth rather than by airway obstruction.
    model["fev1_per_ht"] = model.fev1 / model.height
    model["pef_per_ht"] = model.pef / model.height
    model["log_cotinine"] = np.log1p(model.LBXCOT)

    # --- Skip-pattern resolution -------------------------------------------
    # RDQ080/RDQ100 are only asked when RDQ070 (wheeze in past 12 mo) == yes;
    # SMD415 only when SMD410 == yes. A blank there is a KNOWN zero, not a
    # missing value. Imputing these as missing would corrupt the strongest
    # features in the model.
    model.loc[model.wheeze_12mo.eq(0), ["wheeze_attacks", "exercise_wheeze"]] = 0
    model.loc[model.household_smoker.eq(0), "n_household_smokers"] = 0

    # Dropped: told_overweight (MCQ080, asked 16+ only -> 86% structurally
    # absent) and bmi_category (BMDBMIC, present only in 2011-2012).
    # Continuous BMI plus age carries the same signal across all three cycles.
    model = model.drop(columns=[c for c in ["told_overweight", "bmi_category"]
                                if c in model.columns])

    feats = ["sex", "age", "ethnicity", "income_poverty_ratio", "household_size",
             "family_hx_asthma", "wheeze_12mo", "wheeze_attacks",
             "exercise_wheeze", "nocturnal_dry_cough", "household_smoker",
             "n_household_smokers", "bmi", "height",
             "fev1", "fvc", "pef", "fev1_fvc_ratio", "fev1_per_ht",
             "pef_per_ht", "log_cotinine"]
    feats = [f for f in feats if f in model.columns]
    model = model[["SEQN", "cycle"] + feats + ["current_asthma"]]
    model.to_csv(OUT / "clinical_agent_cohort.csv", index=False)

    n, pos = len(model), int(model.current_asthma.sum())
    print(f"\nmodelling cohort:    {n:,}")
    print(f"  current asthma:    {pos:,} ({pos/n:.1%})")
    print(f"  never asthma:      {n-pos:,}")
    print(f"  former (excluded): {(df.asthma_status=='former').sum():,}")
    print("\nmissingness:")
    print(model[feats].isna().mean().sort_values(ascending=False).mul(100).round(1).to_string())
    print(f"\nwrote {OUT/'clinical_agent_cohort.csv'}")
    return model


if __name__ == "__main__":
    build()
