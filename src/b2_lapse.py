"""Part B tasks 4-8 - lapse problem, leak-free preprocessing, tuning, class balance, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLEAN, OUT

CUTOFF, WINDOW_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")

orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"]).set_index("customer_id")
products = pd.read_csv(CLEAN / "products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id")

# ------------------------------------------------------------------ task 4: cohort, label, features
before = orders[orders["order_ts"] < CUTOFF]               # ONLY these rows may feed features
before_done = before[before["status"] == "completed"]
cohort = sorted(before_done["customer_id"].unique())
sept_active = set(orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF)
                         & (orders["order_ts"] < WINDOW_END)]["customer_id"])
y = pd.Series([0 if cid in sept_active else 1 for cid in cohort], index=cohort, name="lapsed")

g = before_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount_pct": g["discount_pct"].mean(),
    "avg_delivery_minutes": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.mode().sort_values().iloc[0]),
}).loc[cohort]
feat["cancel_rate"] = before.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean()).loc[cohort]
feat["city"] = customers.loc[cohort, "city"]
feat["age"] = customers.loc[cohort, "age"]
feat["gender"] = customers.loc[cohort, "gender"]
feat["tenure_days"] = (CUTOFF - customers.loc[cohort, "signup_date"]).dt.days
NOMINAL = ["city", "gender", "fav_category"]
NUMERIC = [c for c in feat.columns if c not in NOMINAL]
assert before["order_ts"].max() < CUTOFF, "an order on/after the cutoff entered the features"
print(f"cohort {len(cohort):,}; lapsed {int(y.sum())} ({y.mean():.2%}); features: {list(feat.columns)}")

X_train, X_test, y_train, y_test = train_test_split(feat, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(X_train)} test {len(X_test)}; train lapsed {int(y_train.sum())} / not {int((1 - y_train).sum())}")
base_pred = np.zeros(len(y_test), dtype=int)
print(f"baseline (always 'not lapsed') test accuracy {accuracy_score(y_test, base_pred):.1%}, "
      f"lapsed F1 {f1_score(y_test, base_pred, zero_division=0):.2f}")

# ------------------------------------------------------------------ task 5: leak-free preprocessing
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUMERIC),
                              ("cat", OneHotEncoder(drop="first", sparse_output=False), NOMINAL)])

assert feat.isna().sum().sum() == 0, "a feature column has missing values"
print("no feature column contains a missing value: confirmed")

def evaluate(model, name):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    p, r, f = (m(y_test, pred, zero_division=0) for m in (precision_score, recall_score, f1_score))
    print(f"{name:34s} P {p:.3f} R {r:.3f} F1 {f:.3f}  confusion [[TN {cm[0,0]} FP {cm[0,1]}] [FN {cm[1,0]} TP {cm[1,1]}]]")
    return p, r, f

# ------------------------------------------------------------------ task 6: tuning
lr_search = GridSearchCV(Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))]),
                         {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_search = GridSearchCV(Pipeline([("pre", make_pre()),
                                   ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                         cv=5, scoring="f1").fit(X_train, y_train)
best_lr, best_rf = lr_search.best_estimator_, rf_search.best_estimator_
print(f"\nLogReg  best {lr_search.best_params_} CV F1 {lr_search.best_score_:.3f}")
print(f"Forest  best {rf_search.best_params_} CV F1 {rf_search.best_score_:.3f}")

# the scaler inside the fitted pipeline was fit on the training rows only
scaler_mean = best_lr.named_steps["pre"].named_transformers_["num"].mean_[NUMERIC.index("frequency")]
print(f"scaler mean of frequency (train rows only) {scaler_mean:.3f} vs full-cohort mean {feat['frequency'].mean():.3f}")

print("\nTest-set evaluation (lapsed = positive class):")
evaluate(best_lr, "tuned LogisticRegression")
rf_p, rf_r, rf_f = evaluate(best_rf, "tuned RandomForest")

# selection by cross-validated F1 (never by the test set)
selected, sel_name = (best_rf, "RandomForest") if rf_search.best_score_ >= lr_search.best_score_ else (best_lr, "LogisticRegression")
print(f"selected model (higher CV F1): {sel_name}")
sel_p, sel_r, sel_f = evaluate(selected, "selected model")
assert sel_f >= 0.55, "selected model test F1 below 0.55"

# ------------------------------------------------------------------ task 7: class balance (training rows only)
n_extra = int((y_train == 0).sum() - (y_train == 1).sum())
extra = X_train[y_train == 1].sample(n=n_extra, replace=True, random_state=42)
X_bal = pd.concat([X_train, extra])
y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
print(f"\nbalanced training set: {int((y_bal == 0).sum())} not-lapsed / {int((y_bal == 1).sum())} lapsed "
      f"(+{n_extra} duplicated lapsed rows)")
lr_bal = clone(best_lr).fit(X_bal, y_bal)
evaluate(best_lr, "LogReg on original train")
evaluate(lr_bal, "LogReg on balanced train")

# ------------------------------------------------------------------ task 8: interpret, persist, use
names = [n.split("__", 1)[1] for n in best_rf.named_steps["pre"].get_feature_names_out()]
imp = pd.Series(best_rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop-5 random-forest feature importances:\n" + imp.head(5).round(3).to_string())
coef = pd.Series(best_lr.named_steps["clf"].coef_[0], index=names)
print("LogReg coefficients (standardised):", {k: round(float(coef[k]), 3) for k in ("recency", "frequency", "monetary")})

joblib.dump(selected, OUT / "lapse_model.joblib")
reloaded = joblib.load(OUT / "lapse_model.joblib")
same = np.array_equal(selected.predict(X_test), reloaded.predict(X_test)) and \
    np.allclose(selected.predict_proba(X_test), reloaded.predict_proba(X_test))
print(f"\nreloaded model reproduces test predictions exactly: {same}")
assert same

seg = pd.read_csv(OUT / "customer_segments.csv").set_index("customer_id")
proba = pd.Series(selected.predict_proba(X_test)[:, 1], index=X_test.index)
premium_test = [c for c in X_test.index if bool(seg.loc[c, "is_premium"])]
top = proba.loc[premium_test].sort_values(ascending=False).head(10)
ret = pd.DataFrame({"customer_id": top.index, "lapse_probability": top.values.round(4),
                    "actually_lapsed": y_test.loc[top.index].values})
ret.to_csv(OUT / "retention_list.csv", index=False)
print(f"premium customers in test set: {len(premium_test)}; list size {len(ret)}; truly lapsed: {int(ret.actually_lapsed.sum())}")
print(ret.to_string(index=False))
print(f"of all {len(premium_test)} premium test customers, {int(y_test.loc[premium_test].sum())} truly lapsed "
      f"(premium test-set lapse rate {y_test.loc[premium_test].mean():.1%} vs {y_test.mean():.1%} overall)")
