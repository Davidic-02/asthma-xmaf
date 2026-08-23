"""SHAP attributions for the Clinical Agent + logistic odds ratios."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.clinical_agent import load, preprocessor, SEED

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

X, y = load()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)

prep = preprocessor(X)
Atr, Ate = prep.fit_transform(Xtr), prep.transform(Xte)
names = [n.split("__")[-1] for n in prep.get_feature_names_out()]

rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                            class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Atr, ytr)
sv = shap.TreeExplainer(rf).shap_values(Ate)
sv = sv[:, :, 1] if sv.ndim == 3 else sv

imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
imp.to_csv(RES / "clinical_shap_importance.csv", index=False)
print("=== SHAP importance (Random Forest) ===")
print(imp.head(14).to_string(index=False))

plt.figure()
shap.summary_plot(sv, Ate, feature_names=names, show=False, max_display=14)
plt.tight_layout(); plt.savefig(RES / "fig_shap_beeswarm.png", dpi=300); plt.close()
plt.figure()
shap.summary_plot(sv, Ate, feature_names=names, plot_type="bar", show=False, max_display=14)
plt.tight_layout(); plt.savefig(RES / "fig_shap_bar.png", dpi=300); plt.close()

lr = Pipeline([("prep", preprocessor(X)),
               ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=SEED))]).fit(Xtr, ytr)
co = lr.named_steps["clf"].coef_[0]
odds = pd.DataFrame({"feature": names, "coef": co, "odds_ratio": np.exp(co)}) \
         .sort_values("odds_ratio", ascending=False)
odds.to_csv(RES / "clinical_odds_ratios.csv", index=False)
print("\n=== Logistic odds ratios ===")
print(odds.round(3).to_string(index=False))
print(f"\nfigures -> {RES}")
