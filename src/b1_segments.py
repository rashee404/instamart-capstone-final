"""Part B tasks 1-3 - RFM table, K-means sweep, named segments, PCA; writes outputs/customer_segments.csv."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import CLEAN, OUT

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ---------------------------------------------------------------- task 1: RFM
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
}).reset_index()
print(f"RFM customers: {len(rfm):,}")
print(f"recency  median {rfm.recency.median():.0f} min {rfm.recency.min()} max {rfm.recency.max()}")
print(f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()}")
print(f"monetary  mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}")

# ---------------------------------------------------------------- task 2: K-means sweep
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1), "silhouette": round(silhouette_score(X, km.labels_), 3)})
sweep = pd.DataFrame(rows)
sweep["inertia_drop_pct"] = (-sweep["inertia"].pct_change() * 100).round(1)
print("\nK-means sweep:\n" + sweep.to_string(index=False))

# ---------------------------------------------------------------- task 3: profile, name, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)

# Names come from the centroids, never from the (arbitrary) cluster numbers.
premium_id = prof["monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["frequency"].idxmax()
lapsing_id = rest.drop(index=loyal_id)["recency"].idxmax()
occasional_id = [i for i in rest.index if i not in (loyal_id, lapsing_id)][0]
NAMES = {
    premium_id: ("Premium Big Spenders", "Early access to electronics launches and a dedicated priority desk."),
    loyal_id: ("Frequent Loyalists", "Reward streaks and bundle offers to keep the weekly habit going."),
    occasional_id: ("Occasional Shoppers", "Nudge with category coupons to raise order frequency."),
    lapsing_id: ("Fading Customers", "Win-back offer now; they have been silent the longest."),
}
rfm["segment_name"] = rfm["segment_id"].map(lambda i: NAMES[i][0])
rfm["is_premium"] = rfm["segment_id"] == premium_id
prof["segment_name"] = [NAMES[i][0] for i in prof.index]
prof["action"] = [NAMES[i][1] for i in prof.index]
print(f"\nSegment profile (K={FINAL_K}); cluster numbers are labels only, with no order:")
print(prof.to_string())
print(f"Premium segment id = {premium_id} (highest mean monetary)")

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: {r[0]:.3f} {r[1]:.3f} total {r.sum():.3f}")

cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
rfm[cols].to_csv(OUT / "customer_segments.csv", index=False)
print(f"\nwrote {OUT / 'customer_segments.csv'} ({len(rfm)} rows); premium customers: {int(rfm.is_premium.sum())}")
