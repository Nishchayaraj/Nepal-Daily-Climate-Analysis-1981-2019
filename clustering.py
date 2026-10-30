import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = ["Temp_mean", "Annual_rain_mm", "Monsoon_rain_share", "Seasonal_temp_swing"]
FEATURE_LABELS = {
    "Temp_mean": "Mean temp",
    "Annual_rain_mm": "Annual rainfall",
    "Monsoon_rain_share": "Monsoon rain share",
    "Seasonal_temp_swing": "Seasonal temp swing",
}
K_RANGE = range(2, 9)       
K_CHOSEN = 3                
RANDOM_STATE = 42
N_INIT = 50                 
HIGHLIGHT = ["Banke", "Kathmandu", "Humla"]   

ZONE_NAMES_K3 = ["Lowlands & low hills", "Hills", "High mountains"]
# warmest -> coldest;
ZONE_COLORS = ["#D55E00", "#009E73", "#0072B2", "#CC79A7", "#E69F00",
               "#56B4E9", "#7f8c8d", "#b7950b"]
SOURCE = "Source: Nepal Daily Climate Dataset (Kaggle), 1981-2019, 62 districts"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

REPORT = []

def say(text=""):
    """Print and remember a line for the text report."""
    print(text)
    REPORT.append(str(text))


def save_fig(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"  saved {name}")


def zone_names(k):
    if k == 3:
        return ZONE_NAMES_K3
    names = []
    for i in range(k):
        tag = " (warmest)" if i == 0 else " (coldest)" if i == k - 1 else ""
        names.append(f"Zone {i + 1}{tag}")
    return names


# 1-2. Load and standardize

def load_features(data_dir):
    path = data_dir / "district_features.csv"
    if not path.exists():
        sys.exit(f"Cannot find {path}. Run prepare_data.py first.")
    feats = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in feats.columns]
    if missing:
        sys.exit(f"district_features.csv is missing columns: {missing}")
    return feats


def standardize(df, cols):
    """Scale each column to mean 0 and standard deviation 1, so that rainfall
    in mm and temperature in C count equally."""
    return StandardScaler().fit_transform(df[cols])


# 3. Try k = 2..8

def evaluate_k(X):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def plot_choose_k(ks, out_dir):
    best_k = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.9))

    a.plot(ks["k"], ks["inertia"], marker="o", lw=2.4, color="#34495e", zorder=3)
    a.axvline(K_CHOSEN, color="#c0392b", ls="--", lw=1.8, zorder=2)
    a.text(K_CHOSEN + 0.1, ks["inertia"].max() * 0.93, f"chosen k = {K_CHOSEN}",
           color="#c0392b", fontsize=10)
    a.set_xlabel("Number of clusters (k)")
    a.set_ylabel("Inertia (lower = tighter clusters)")
    a.set_title("Elbow method: look for where the gains slow down", loc="left")
    a.grid(alpha=0.3)

    b.plot(ks["k"], ks["silhouette"], marker="o", lw=2.4, color="#117a65", zorder=3)
    b.scatter([best_k], [ks.loc[ks["k"] == best_k, "silhouette"]], s=260, marker="*",
              color="#f39c12", edgecolor="black", zorder=4,
              label=f"highest score (k = {best_k})")
    b.axvline(K_CHOSEN, color="#c0392b", ls="--", lw=1.8, zorder=2,
              label=f"chosen k = {K_CHOSEN}")
    b.set_xlabel("Number of clusters (k)")
    b.set_ylabel("Silhouette score (higher = better separated)")
    b.set_title("Silhouette score: how clearly separated are the groups?", loc="left")
    b.legend(frameon=True, fontsize=9)
    b.grid(alpha=0.3)

    for ax in (a, b):
        ax.set_xticks(list(K_RANGE))
    fig.suptitle("Choosing the number of climate zones", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    save_fig(fig, out_dir, "7_choose_k.png")
    return best_k


# 4-5. Final clustering, naming, representatives

def fit_final(feats, X):
    km = KMeans(n_clusters=K_CHOSEN, n_init=N_INIT, random_state=RANDOM_STATE).fit(X)
    raw = km.labels_

    
    temp_by_cluster = feats.groupby(raw)["Temp_mean"].mean().sort_values(ascending=False)
    remap = {old: new for new, old in enumerate(temp_by_cluster.index)}
    rank = np.array([remap[c] for c in raw])

    names = zone_names(K_CHOSEN)
    out = feats.copy()
    out["Cluster"] = rank
    out["Zone"] = [names[r] for r in rank]

    centres = np.vstack([X[rank == c].mean(axis=0) for c in range(K_CHOSEN)])
    out["Distance_to_centre"] = np.linalg.norm(X - centres[rank], axis=1)
    out["Silhouette"] = silhouette_samples(X, rank)

    out["Representative"] = False
    for c in range(K_CHOSEN):
        idx = out.loc[out["Cluster"] == c, "Distance_to_centre"].idxmin()
        out.loc[idx, "Representative"] = True
    return out, centres


def profile_table(out):
    cols = FEATURES + ["Pressure_mean", "Winter_MinTemp"]
    g = out.groupby("Zone", sort=False)
    prof = g[cols].mean().round(2)
    prof.insert(0, "Districts", g.size())
    prof["Pressure_min"] = g["Pressure_mean"].min().round(1)
    prof["Pressure_max"] = g["Pressure_mean"].max().round(1)
    order = [zone for zone in zone_names(K_CHOSEN) if zone in prof.index]
    return prof.loc[order]


# Figures: map, profiles

def draw_outline(ax, path):
    if not Path(path).exists():
        return
    gj = json.loads(Path(path).read_text())
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            for ring in poly:
                xs, ys = zip(*[(pt[0], pt[1]) for pt in ring])
                ax.plot(xs, ys, color="#34495e", lw=1.4, zorder=1)
                ax.fill(xs, ys, color="#f4f6f7", zorder=0)


def plot_map(out, outline, out_dir):
    names = zone_names(K_CHOSEN)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    draw_outline(ax, outline)
    for c, name in enumerate(names):
        s = out[out["Cluster"] == c]
        ax.scatter(s["Longitude"], s["Latitude"], s=190, color=ZONE_COLORS[c],
                   edgecolor="#2c3e50", linewidth=0.7, zorder=3,
                   label=f"{name} ({len(s)} districts)")
    
    label_pos = {"Dolpa": (83.0, 29.45), "Lamjung": (85.2, 28.65),
                 "Nawalparasi": (84.0, 27.25)}
    for _, r in out[out["Representative"]].iterrows():
        ax.scatter(r["Longitude"], r["Latitude"], s=700, marker="*", color="#f1c40f",
                   edgecolor="black", linewidth=1.3, zorder=5)
        box = dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.95)
        if r["District"] in label_pos:
            ax.annotate(r["District"], (r["Longitude"], r["Latitude"]),
                        xytext=label_pos[r["District"]], textcoords="data", ha="center",
                        fontsize=10, fontweight="bold", zorder=6, bbox=box,
                        arrowprops=dict(arrowstyle="-", color="black", lw=1))
        else:
            ax.annotate(r["District"], (r["Longitude"], r["Latitude"]), xytext=(0, 20),
                        textcoords="offset points", ha="center", fontsize=10,
                        fontweight="bold", zorder=6, bbox=box)
    ax.scatter([], [], s=260, marker="*", color="#f1c40f", edgecolor="black",
               label="Representative district (closest to zone centre)")
    ax.set_xlim(79.9, 88.4)
    ax.set_ylim(26.1, 30.6)
    ax.set_aspect(1 / np.cos(np.radians(28)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax.set_title(f"Climate zones found by K-Means (k = {K_CHOSEN})", loc="left")
    fig.tight_layout()
    save_fig(fig, out_dir, "8_cluster_map.png")


def plot_profiles(out, X, out_dir):
    names = zone_names(K_CHOSEN)
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})

    mean_all = out[FEATURES].mean()
    std_all = out[FEATURES].std(ddof=0)
    w = 0.8 / K_CHOSEN
    for c, name in enumerate(names):
        z = (out.loc[out["Cluster"] == c, FEATURES].mean() - mean_all) / std_all
        a.bar(np.arange(len(FEATURES)) + (c - (K_CHOSEN - 1) / 2) * w, z.values, width=w,
              color=ZONE_COLORS[c], label=name, zorder=3)
    a.axhline(0, color="black", lw=1)
    a.set_xticks(range(len(FEATURES)))
    a.set_xticklabels([FEATURE_LABELS[f].replace(" ", "\n", 1) for f in FEATURES])
    a.set_ylabel("Zone average vs Nepal average\n(in standard deviations)")
    a.set_title("What makes each zone different", loc="left")
    a.legend(frameon=True, fontsize=9)
    a.grid(axis="y", alpha=0.3, zorder=0)

    pca = PCA(n_components=2).fit(X)
    P = pca.transform(X)
    for c, name in enumerate(names):
        m = (out["Cluster"] == c).values
        b.scatter(P[m, 0], P[m, 1], s=90, color=ZONE_COLORS[c], edgecolor="#2c3e50",
                  linewidth=0.6, label=name, zorder=3)
    for i, r in out.reset_index(drop=True).iterrows():
        if r["Representative"]:
            b.scatter(P[i, 0], P[i, 1], s=420, marker="*", color="#f1c40f",
                      edgecolor="black", linewidth=1.1, zorder=5)
            b.annotate(r["District"], (P[i, 0], P[i, 1]), xytext=(0, 16),
                       textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ev = pca.explained_variance_ratio_
    b.set_xlabel(f"Main climate gradient (PC1, {ev[0]:.0%} of variation)")
    b.set_ylabel(f"Second gradient (PC2, {ev[1]:.0%})")
    b.set_title("The zones blend into each other: no sharp gaps", loc="left")
    b.margins(x=0.1, y=0.1)
    b.grid(alpha=0.3)
    fig.suptitle("Describing the climate zones", fontsize=15, fontweight="bold",
                 x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    save_fig(fig, out_dir, "9_cluster_profiles.png")
    return ev


# 6. How sensitive is the result to the feature choice?

def sensitivity(feats, labels):
    num_cols = [c for c in feats.columns if c != "District"]
    res = {}

    Xall = StandardScaler().fit_transform(feats[num_cols])
    lab_all = KMeans(K_CHOSEN, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(Xall)
    res["All 13 numeric columns (incl. lat/lon)"] = adjusted_rand_score(labels, lab_all)

    pca = PCA(n_components=0.90).fit(Xall)
    scores = pca.transform(Xall)
    lab_pca = KMeans(K_CHOSEN, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(scores)
    res[f"PCA on all columns ({pca.n_components_} components, 90% of variation)"] = \
        adjusted_rand_score(labels, lab_pca)

    for drop in FEATURES:
        cols = [f for f in FEATURES if f != drop]
        lab = KMeans(K_CHOSEN, n_init=N_INIT, random_state=RANDOM_STATE).fit_predict(
            standardize(feats, cols))
        res[f"Chosen features without {FEATURE_LABELS[drop]}"] = adjusted_rand_score(labels, lab)
    return res


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python clustering.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    outline = data_dir / "nepal_outline.geojson"

    feats = load_features(data_dir)
    X = standardize(feats, FEATURES)

    say("=" * 64)
    say("CLUSTERING REPORT - Nepal climate zones")
    say("=" * 64)
    say(f"Districts: {len(feats)}")
    say("Features used (standardized): " + ", ".join(FEATURES))
    say("Reminder: rainfall features use the full 1981-2019 period (2004 jump included).")

    # choose k
    say("\nStep A: trying k = 2..8")
    ks = evaluate_k(X)
    say(ks.round(3).to_string(index=False))
    best_k = plot_choose_k(ks, out_dir)
    chosen_sil = ks.loc[ks["k"] == K_CHOSEN, "silhouette"].iloc[0]
    say(f"Highest silhouette: k = {best_k}. Chosen k = {K_CHOSEN} "
        f"(silhouette {chosen_sil:.3f}).")
    if best_k != K_CHOSEN:
        say(f"NOTE: k = {best_k} scores higher statistically, and the elbow curve is smooth "
            f"with no sharp bend. k = {K_CHOSEN} is chosen mainly because it gives zones that are "
            f"easy to interpret. Say this openly in your presentation.")

    # final model
    say(f"\nStep B: final K-Means with k = {K_CHOSEN}")
    out, _ = fit_final(feats, X)
    prof = profile_table(out)
    say("\nZone profiles (average of each feature):")
    say(prof.to_string())

    say("\nNote: zone names are descriptive labels based on the temperature ranking, "
        "not official regions. Check the member lists below.")
    say("\nDistricts in each zone:")
    for name in zone_names(K_CHOSEN):
        members = sorted(out.loc[out["Zone"] == name, "District"])
        say(f"  {name} ({len(members)}): " + ", ".join(members))

    overlap = []
    zs = prof.index.tolist()
    for i in range(len(zs) - 1):  
        if prof.loc[zs[i], "Pressure_min"] < prof.loc[zs[i + 1], "Pressure_max"]:
            overlap.append(f"{zs[i]} and {zs[i + 1]}")
    say("\nAltitude check (pressure: lower = higher altitude):")
    say("  Pressure ranges of the zones do NOT overlap - zones are clean altitude bands."
        if not overlap else
        "  Pressure ranges overlap for: " + "; ".join(overlap) +
        " - some districts are borderline between zones.")

    # representatives
    say("\nStep C: representative districts (use these for forecasting in Step 5)")
    rep_rows = []
    for name in zone_names(K_CHOSEN):
        s = out[out["Zone"] == name].sort_values("Distance_to_centre")
        top3 = ", ".join(f"{r.District} ({r.Distance_to_centre:.2f})" for r in s.head(3).itertuples())
        say(f"  {name}: {s.iloc[0]['District']}   [closest three: {top3}]")
        rep_rows.append(s.iloc[0][["Zone", "District", "Distance_to_centre",
                                   "Temp_mean", "Annual_rain_mm", "Pressure_mean"]])
    reps = pd.DataFrame(rep_rows)

    say("\n  How spread out is each zone? (average distance to the zone centre; "
        "a representative far from 0 is less typical)")
    for name in zone_names(K_CHOSEN):
        s_ = out[out["Zone"] == name]
        say(f"    {name:<22} average {s_['Distance_to_centre'].mean():.2f}, "
            f"representative {s_['Distance_to_centre'].min():.2f}")

    say("\nWhere the three EDA districts fall:")
    for d in HIGHLIGHT:
        if d in set(out["District"]):
            say(f"  {d}: {out.loc[out['District'] == d, 'Zone'].iloc[0]}")

    low = out.sort_values("Silhouette").head(5)
    say("\nMost borderline districts (lowest silhouette; negative = may fit another zone better):")
    for r in low.itertuples():
        say(f"  {r.District:<14} {r.Zone:<18} silhouette {r.Silhouette:+.2f}")

    # sensitivity
    say("\nStep D: how much does the result depend on the feature choice?")
    say("  (agreement with the main result: 1.00 = identical groups, 0 = unrelated)")
    for label, ari in sensitivity(feats, out["Cluster"].values).items():
        say(f"  {label:<62} {ari:.2f}")

    # figures + files
    say("\nSaving figures and tables:")
    plot_map(out, outline, out_dir)
    ev = plot_profiles(out, X, out_dir)
    say(f"  (the first two PCA axes capture {ev.sum():.0%} of the variation in the 4 features)")

    out.drop(columns=["Cluster"]).assign(Cluster=out["Cluster"]).to_csv(
        out_dir / "district_clusters.csv", index=False)
    reps.to_csv(out_dir / "representative_districts.csv", index=False)
    prof.to_csv(out_dir / "cluster_profiles.csv")
    ks.to_csv(out_dir / "k_selection.csv", index=False)
    (out_dir / "clustering_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
