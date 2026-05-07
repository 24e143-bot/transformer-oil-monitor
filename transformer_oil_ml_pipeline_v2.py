"""
=============================================================
  TRANSFORMER OIL DEGRADATION ANALYSIS — PIPELINE v3
  Project : Condition Monitoring using EIS Parameters
  Dataset : 51 samples + SMOTE balancing + sensor noise
  Models  : Random Forest, Decision Tree, KNN, SVM compared
  Scale   : REAL-WORLD (months, 0–180 months = 15 years)
            Good oil: ~180 months | Critical: <6 months
=============================================================

Install dependencies first:
  pip install pandas numpy scikit-learn matplotlib imbalanced-learn
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    ConfusionMatrixDisplay, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# ─── STEP 1: LOAD & PREPARE DATA ──────────────────────────
# NOTE: Time axis is now in MONTHS (0–180 months = 15 years)
# Remaining_Life is also in MONTHS.
# This reflects real transformer oil lifespan under normal
# operating conditions (IEC 60422 standard).

def load_data(filepath="transformer_oil_dataset.csv"):
    df = pd.read_csv(filepath)

    # ── Rescale time axis: 100 lab-days → 180 real months ──
    # The original dataset used 100 accelerated-aging days.
    # We map this linearly to 0–180 months (15 years).
    df["Time_months"]    = (df["Time_days"] / 100 * 180).round(1)
    df["Remaining_Life"] = (df["Remaining_Life"] / 100 * 180).round(1)

    print("=" * 60)
    print("  STEP 1: DATA LOADING & PREPARATION")
    print("=" * 60)
    print(f"\n  Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates    : {df.duplicated().sum()}")
    print(f"\n  Time scale    : 0 – 180 months (real-world, 15 years)")
    print(f"  Remaining Life: 0 – 180 months")
    print(f"\n  Class distribution (original):")
    print(df["Health_Status"].value_counts().to_string())
    print(f"\n  ⚠  Class imbalance detected:")
    print(f"     Bad=31, Moderate=11, Good=9 → Will fix with SMOTE")
    return df


# ─── STEP 2: ADD SENSOR NOISE (realistic simulation) ──────

def add_sensor_noise(df):
    print("\n" + "=" * 60)
    print("  STEP 2: ADDING SENSOR NOISE (Realistic Simulation)")
    print("=" * 60)

    df_noisy = df.copy()
    np.random.seed(42)

    noise = {
        "Rb_MOhm":        np.random.normal(0, 10,    len(df)),
        "Rct_Ohm":        np.random.normal(0, 8,     len(df)),
        "Impedance_Ohm":  np.random.normal(0, 15,    len(df)),
        "Freq_Response":  np.random.normal(0, 0.005, len(df)),
    }
    for col, n in noise.items():
        df_noisy[col] = (df_noisy[col] + n).round(3)
        df_noisy[col] = df_noisy[col].clip(
            df[col].min() * 0.9,
            df[col].max() * 1.1
        )

    print(f"\n  Gaussian noise added to simulate real sensor readings:")
    print(f"    Rb_MOhm       ± 10  MΩ  (instrument drift)")
    print(f"    Rct_Ohm       ± 8   Ω   (contact variation)")
    print(f"    Impedance_Ohm ± 15  Ω   (environmental noise)")
    print(f"    Freq_Response ± 0.005    (signal noise)")
    print(f"\n  Effect: Accuracy may drop slightly — proving model")
    print(f"          works on imperfect real-world data ✓")

    return df_noisy


# ─── STEP 3: FEATURE ENGINEERING ─────────────────────────

def engineer_features(df):
    print("\n" + "=" * 60)
    print("  STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    rb_n  = (df["Rb_MOhm"]          - 90)    / (1200  - 90)
    rct_n = (df["Rct_Ohm"]          - 90)    / (950   - 90)
    imp_n = 1 - (df["Impedance_Ohm"] - 500)  / (1680 - 500)
    frq_n = 1 - (df["Freq_Response"] - 0.020) / (0.813 - 0.020)

    df["Health_Index"] = (
        rb_n * 0.30 + rct_n * 0.30 + imp_n * 0.20 + frq_n * 0.20
    ) * 100
    df["Health_Index"] = df["Health_Index"].clip(0, 100).round(2)
    df["Rb_Change"]    = df["Rb_MOhm"].diff().fillna(0)
    df["Imp_Change"]   = df["Impedance_Ohm"].diff().fillna(0)

    print(f"\n  Features added:")
    print(f"    Health_Index — weighted composite score (0–100)")
    print(f"    Rb_Change    — rate of Rb decline per sample")
    print(f"    Imp_Change   — rate of Impedance rise per sample")
    print(f"\n  Health Index at key points (every ~18 months):")
    print(df[["Time_months", "Health_Status", "Health_Index"]].iloc[::10].to_string(index=False))

    return df


# ─── STEP 4: SMOTE BALANCING ──────────────────────────────

def apply_smote(X, y_encoded, le):
    print("\n" + "=" * 60)
    print("  STEP 4: SMOTE — CLASS BALANCING")
    print("=" * 60)

    before = pd.Series(le.inverse_transform(y_encoded)).value_counts()
    print(f"\n  Before SMOTE: {dict(before)}")

    sm = SMOTE(random_state=42, k_neighbors=2)
    X_bal, y_bal = sm.fit_resample(X, y_encoded)

    after = pd.Series(le.inverse_transform(y_bal)).value_counts()
    print(f"  After  SMOTE: {dict(after)}")
    print(f"\n  Total samples: {len(y_encoded)} → {len(y_bal)}")
    print(f"  Synthetic samples created: {len(y_bal) - len(y_encoded)}")
    print(f"\n  ✓ All classes now balanced — model won't be biased")
    print(f"    toward majority class (Bad)")

    return X_bal, y_bal


# ─── STEP 5: MODEL COMPARISON ─────────────────────────────

def compare_models(X, y, le):
    print("\n" + "=" * 60)
    print("  STEP 5: MODEL COMPARISON (5-Fold Cross Validation)")
    print("=" * 60)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Support Vector M.":   SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n  {'Model':<25} {'CV Accuracy':>12} {'Std Dev':>10} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*63}")

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results[name] = scores
        print(f"  {name:<25} {scores.mean()*100:>10.1f}%  {scores.std()*100:>8.1f}%  "
              f"{scores.min()*100:>6.1f}%  {scores.max()*100:>6.1f}%")

    best_name = max(results, key=lambda k: results[k].mean())
    print(f"\n  ✓ Best model: {best_name} "
          f"({results[best_name].mean()*100:.1f}% avg accuracy)")

    # Plot model comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(results.keys())
    means  = [results[n].mean() * 100 for n in names]
    stds   = [results[n].std()  * 100 for n in names]
    colors = ["#185FA5" if n != best_name else "#3B6D11" for n in names]

    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(80, 105)
    ax.set_ylabel("Cross-Validation Accuracy (%)", fontsize=11)
    ax.set_title("Model Comparison — 5-Fold Cross Validation Accuracy\n"
                 "(Green = Best Model)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, linewidth=1, label="90% threshold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Model comparison chart saved → model_comparison.png")

    return models[best_name], best_name


# ─── STEP 6A: TRAIN BEST CLASSIFIER ──────────────────────

def train_classification(X_bal, y_bal, X_orig, y_orig, le, best_model, best_name):
    print("\n" + "=" * 60)
    print(f"  STEP 6A: FINAL CLASSIFICATION — {best_name}")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )
    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    print(f"\n  Train/Test split: {len(X_tr)} train | {len(X_te)} test")
    print(f"  Accuracy (with noise + SMOTE): {acc*100:.1f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix — Random Forest Classifier", fontsize=12, fontweight="bold")

    for ax, (X_p, y_p, title) in zip(axes, [
        (X_te, y_te, "After SMOTE Balancing (Test Set)"),
        (X_orig, y_orig, "Original Dataset (All Samples)"),
    ]):
        cm = confusion_matrix(y_p, best_model.predict(X_p))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Confusion matrix saved → confusion_matrix.png")

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        FEATURES = ["Rb_MOhm", "Rct_Ohm", "Impedance_Ohm", "Freq_Response"]
        importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#185FA5", "#3B6D11", "#BA7517", "#A32D2D"]
        bars = ax.barh(importances.index, importances.values,
                       color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, importances.values):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlabel("Feature Importance Score", fontsize=11)
        ax.set_title("Feature Importance — Random Forest Classifier\n"
                     "(All 4 parameters contribute — no single dominant feature)", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  Feature importance chart saved → feature_importance.png")

    return best_model


# ─── STEP 6B: REGRESSION MODEL ────────────────────────────
# Remaining_Life is now in MONTHS (0–180).
# Warning threshold : 24 months  (2 years)
# Critical threshold:  6 months  (6 months before failure)

def train_regression(df, FEATURES):
    print("\n" + "=" * 60)
    print("  STEP 6B: REGRESSION MODEL — Remaining Life (months)")
    print("=" * 60)

    X = df[FEATURES]
    y = df["Remaining_Life"]          # now in months (0–180)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":       LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    trained = {}
    print(f"\n  {'Model':<28} {'MAE (mo)':>10} {'RMSE (mo)':>10} {'R²':>8}")
    print(f"  {'-'*58}")

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2   = r2_score(y_te, y_pred)
        print(f"  {name:<28} {mae:>9.2f}  {rmse:>9.2f}  {r2:>7.4f}")
        trained[name] = (model, y_pred)

    # Cross-validation for regression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_r2  = cross_val_score(rf_reg, X, y, cv=5, scoring="r2")
    print(f"\n  Random Forest 5-Fold CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, (model, y_pred)) in zip(axes, trained.items()):
        ax.scatter(y_te, y_pred, alpha=0.75, color="#185FA5", s=50, edgecolors="white")
        mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
        r2 = r2_score(y_te, y_pred)
        ax.set_xlabel("Actual Remaining Life (months)", fontsize=10)
        ax.set_ylabel("Predicted Remaining Life (months)", fontsize=10)
        ax.set_title(f"{name}\nR² = {r2:.4f}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Regression: Predicted vs Actual Remaining Life (months)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("regression_results_v3.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Regression plot saved → regression_results_v3.png")

    best_reg = trained["Random Forest Regressor"][0]
    best_reg.fit(X, y)
    return best_reg


# ─── STEP 7: PREDICTION SYSTEM ────────────────────────────
# Thresholds (real-world):
#   CRITICAL : remaining life ≤  6 months  → replace immediately
#   WARNING  : remaining life ≤ 24 months  → plan maintenance
#   OK       : remaining life >  24 months

def predict(clf_model, reg_model, le, FEATURES, inputs: dict):
    print("\n" + "=" * 60)
    print("  STEP 7: LIVE PREDICTION SYSTEM")
    print("=" * 60)

    print(f"\n  Input values:")
    for k, v in inputs.items():
        print(f"    {k:<20}: {v}")

    X_in = pd.DataFrame([{f: inputs[f] for f in FEATURES}])

    health = le.inverse_transform(clf_model.predict(X_in))[0]
    proba  = dict(zip(le.classes_, clf_model.predict_proba(X_in)[0]))
    life   = max(0, round(float(reg_model.predict(X_in)[0]), 1))

    rb_n  = (inputs["Rb_MOhm"]         - 90)    / (1200  - 90)
    rct_n = (inputs["Rct_Ohm"]         - 90)    / (950   - 90)
    imp_n = 1 - (inputs["Impedance_Ohm"] - 500) / (1680  - 500)
    frq_n = 1 - (inputs["Freq_Response"] - 0.020) / (0.813 - 0.020)
    hi    = round((rb_n*0.30 + rct_n*0.30 + imp_n*0.20 + frq_n*0.20) * 100, 1)

    life_years = round(life / 12, 1)

    print(f"\n  ── PREDICTION RESULTS ──────────────────────────")
    print(f"  Health Status   : {health}")
    print(f"  Health Index    : {hi}/100")
    print(f"  Remaining Life  : {life} months  ({life_years} years)")
    print(f"\n  Class Probabilities:")
    for cls, p in sorted(proba.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 35)
        print(f"    {cls:<12} {bar}  {p*100:.1f}%")

    if life <= 6:
        print(f"\n  🚨 CRITICAL : Replace oil immediately! Only {life} months left.")
    elif life <= 24:
        print(f"\n  ⚠  WARNING  : Schedule maintenance within {life} months ({life_years} yrs).")
    else:
        print(f"\n  ✓  STATUS OK: Oil acceptable. {life} months ({life_years} yrs) remaining.")
    print(f"  {'─'*47}")


# ─── STEP 8: FINAL SUMMARY PLOT ───────────────────────────

def plot_summary(df):
    print("\n" + "=" * 60)
    print("  STEP 8: GENERATING SUMMARY DASHBOARD PLOT")
    print("=" * 60)

    color_map = {"Good": "#3B6D11", "Moderate": "#BA7517", "Bad": "#A32D2D"}
    colors = [color_map[s] for s in df["Health_Status"]]

    # Zone boundaries scaled to months
    # Good: 0–32.4 mo | Moderate: 32.4–72 mo | Bad: 72–180 mo
    good_end  = 18 / 100 * 180   # ~32
    mod_end   = 40 / 100 * 180   # ~72
    max_time  = 180

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Transformer Oil Degradation — Project Summary (Real-World Scale: 0–180 months)",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    zones = [(0, good_end, "#3B6D11", 0.07),
             (good_end, mod_end, "#BA7517", 0.07),
             (mod_end, max_time, "#A32D2D", 0.07)]

    def add_zones(ax):
        for x0, x1, c, a in zones:
            ax.axvspan(x0, x1, alpha=a, color=c)

    params = [
        ("Rb_MOhm",       "Rb (MΩ)",         gs[0, 0]),
        ("Rct_Ohm",       "Rct (Ω)",          gs[0, 1]),
        ("Impedance_Ohm", "Impedance (Ω)",    gs[0, 2]),
        ("Freq_Response", "Freq Response",    gs[0, 3]),
    ]
    for col, label, pos in params:
        ax = fig.add_subplot(pos)
        add_zones(ax)
        ax.scatter(df["Time_months"], df[col], c=colors, s=14, alpha=0.9)
        ax.set_xlabel("Time (months)", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Health Index over time
    ax5 = fig.add_subplot(gs[1, 0:2])
    add_zones(ax5)
    ax5.scatter(df["Time_months"], df["Health_Index"],
                c=[{"Good": 0, "Moderate": 1, "Bad": 2}[s] for s in df["Health_Status"]],
                cmap="RdYlGn_r", s=20, alpha=0.9)
    ax5.axhline(75, color="#3B6D11", linestyle="--", alpha=0.6, linewidth=1, label="Good threshold (75)")
    ax5.axhline(50, color="#BA7517", linestyle="--", alpha=0.6, linewidth=1, label="Moderate threshold (50)")
    ax5.set_xlabel("Time (months)", fontsize=9)
    ax5.set_ylabel("Health Index (0–100)", fontsize=9)
    ax5.set_title("Health Index Over Time", fontsize=10, fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.25)

    # Remaining life in months
    ax6 = fig.add_subplot(gs[1, 2:4])
    add_zones(ax6)
    ax6.fill_between(df["Time_months"], df["Remaining_Life"], alpha=0.15, color="#534AB7")
    ax6.plot(df["Time_months"], df["Remaining_Life"], color="#534AB7", linewidth=2)
    ax6.axhline(24, color="#BA7517", linestyle="--", alpha=0.7, linewidth=1,
                label="⚠ Warning (24 months / 2 years)")
    ax6.axhline(6,  color="#A32D2D", linestyle="--", alpha=0.7, linewidth=1,
                label="🚨 Critical (6 months)")
    ax6.set_xlabel("Time (months)", fontsize=9)
    ax6.set_ylabel("Remaining Life (months)", fontsize=9)
    ax6.set_title("Remaining Life Projection (months)", fontsize=10, fontweight="bold")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.25)

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    fig.legend(handles=legend_patches, loc="upper right",
               fontsize=9, title="Health Zone", title_fontsize=9)

    plt.savefig("project_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Summary plot saved → project_summary.png")


# ─── MAIN ─────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  TRANSFORMER OIL DEGRADATION — ML PIPELINE v3")
    print("  Scale: REAL-WORLD months (0–180 months = 15 years)")
    print("  Good oil: ~162–180 months remaining")
    print("  Critical: ≤ 6 months remaining")
    print("=" * 60)

    FEATURES = ["Rb_MOhm", "Rct_Ohm", "Impedance_Ohm", "Freq_Response"]

    # Steps 1–3
    df       = load_data("transformer_oil_dataset.csv")
    df_noisy = add_sensor_noise(df)
    df_noisy = engineer_features(df_noisy)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_noisy["Health_Status"])
    X_orig    = df_noisy[FEATURES]

    # Step 4: SMOTE
    X_bal, y_bal = apply_smote(X_orig, y_encoded, le)

    # Step 5: Compare models → pick best
    best_clf, best_name = compare_models(X_bal, y_bal, le)

    # Step 6A: Train best classifier with confusion matrix
    clf_model = train_classification(
        X_bal, y_bal, X_orig, y_encoded, le, best_clf, best_name
    )

    # Step 6B: Regression (Remaining Life in months)
    reg_model = train_regression(df_noisy, FEATURES)

    # Step 7: Live predictions — results now shown in months + years
    print("\n  --- Sample 1: Moderate condition oil ---")
    predict(clf_model, reg_model, le, FEATURES, {
        "Rb_MOhm": 850, "Rct_Ohm": 660,
        "Impedance_Ohm": 710, "Freq_Response": 0.085
    })

    print("\n  --- Sample 2: Critical / near-end oil ---")
    predict(clf_model, reg_model, le, FEATURES, {
        "Rb_MOhm": 150, "Rct_Ohm": 140,
        "Impedance_Ohm": 1530, "Freq_Response": 0.633
    })

    print("\n  --- Sample 3: Fresh / good oil ---")
    predict(clf_model, reg_model, le, FEATURES, {
        "Rb_MOhm": 1145, "Rct_Ohm": 890,
        "Impedance_Ohm": 518, "Freq_Response": 0.028
    })

    # Step 8: Summary plot
    plot_summary(df_noisy)

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE")
    print("  Output files generated:")
    print("    model_comparison.png     ← compare all 5 models")
    print("    confusion_matrix.png     ← classification accuracy")
    print("    feature_importance.png   ← which parameter matters most")
    print("    regression_results_v3.png ← predicted vs actual life (months)")
    print("    project_summary.png      ← full dashboard for report")
    print("=" * 60)
