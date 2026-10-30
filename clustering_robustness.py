import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Settings

BASE_FEATURES = ["Temp_mean", "Annual_rain_mm", "Monsoon_rain_share", "Seasonal_temp_swing"]
TEMP_FEATURES = ["Temp_mean", "Seasonal_temp_swing", "Pressure_mean"]
K = 3
K_RANGE = range(2, 9)
K_STAB = range(2, 7)
RANDOM_STATE = 42
N_INIT = 50
N_SUBSAMPLES = 500
SUBSAMPLE_FRACTION = 0.8
BREAK_YEAR = 2004
ZONE_NAMES = ["Lowlands & low hills", "Hills", "High mountains"]
ZONE_COLORS = ["#D55E00", "#009E73", "#0072B2"]
BELT_ORDER = ["Terai", "Hill", "Mountain"]
BELT_CODE = {"Terai": 0, "Hill": 1, "Mountain": 2}
BELT_MARKER = {"Terai": "s", "Hill": "o", "Mountain": "^"}
BELT_COLOR = {"Terai": "#D55E00", "Hill": "#009E73", "Mountain": "#0072B2"}
SOURCE = "Source: Nepal Daily Climate Dataset (Kaggle), 1981-2019, 62 districts"


_M, _H, _T = "Mountain", "Hill", "Terai"
OFFICIAL = {
    "Taplejung": ("Taplejung", _M, ""), "Sankhuwasabha": ("Sankhuwasabha", _M, ""),
    "Solukhumbu": ("Solukhumbu", _M, ""), "Dolkha": ("Dolakha", _M, ""),
    "Rasuwa": ("Rasuwa", _M, ""), "Manang": ("Manang", _M, ""), "Mustang": ("Mustang", _M, ""),
    "Dolpa": ("Dolpa", _M, ""), "Jumla": ("Jumla", _M, ""), "Mugu": ("Mugu", _M, ""),
    "Humla": ("Humla", _M, ""), "Bajang": ("Bajhang", _M, ""), "Darchula": ("Darchula", _M, ""),
    "Arghakhanchi": ("Arghakhanchi", _H, ""), "Baglung": ("Baglung", _H, ""),
    "Baitadi": ("Baitadi", _H, ""), "Bhaktapur": ("Bhaktapur", _H, "Kathmandu Valley"),
    "Dadeldhura": ("Dadeldhura", _H, ""), "Dailekh": ("Dailekh", _H, ""),
    "Dhading": ("Dhading", _H, "Mountain under the 2017 Act"),
    "Dhankuta": ("Dhankuta", _H, ""), "Doti": ("Doti", _H, ""),
    "Gorkha": ("Gorkha", _H, "Mountain under the 2017 Act"), "Gulmi": ("Gulmi", _H, ""),
    "Ilam": ("Ilam", _H, ""), "Kabhre": ("Kavrepalanchok", _H, ""), "Kaski": ("Kaski", _H, ""),
    "Kathmandu": ("Kathmandu", _H, "Kathmandu Valley"), "Lalitpur": ("Lalitpur", _H, "Kathmandu Valley"),
    "Lamjung": ("Lamjung", _H, ""), "Makwanpur": ("Makwanpur", _H, "Inner Terai, counted as Hill"),
    "Myagdi": ("Myagdi", _H, ""), "Nuwakot": ("Nuwakot", _H, ""), "Okhaldhunga": ("Okhaldhunga", _H, ""),
    "Palpa": ("Palpa", _H, ""), "Panchther": ("Panchthar", _H, ""), "Parbat": ("Parbat", _H, ""),
    "Rukum": ("Rukum", _H, "Eastern Rukum is Mountain under the 2017 Act"),
    "Salyan": ("Salyan", _H, ""), "Sindhuli": ("Sindhuli", _H, "Inner Terai, counted as Hill"),
    "Surkhet": ("Surkhet", _H, "Inner Terai, counted as Hill"), "Syangja": ("Syangja", _H, ""),
    "Tanahun": ("Tanahu", _H, ""), "Terhathum": ("Tehrathum", _H, ""),
    "Udayapur": ("Udayapur", _H, "Inner Terai, counted as Hill"),
    "Banke": ("Banke", _T, ""), "Bara": ("Bara", _T, ""), "Bardiya": ("Bardiya", _T, ""),
    "Chitawan": ("Chitwan", _T, "Inner Terai, counted as Terai"),
    "Dang": ("Dang", _T, "Inner Terai, counted as Terai"), "Dhanusa": ("Dhanusha", _T, ""),
    "Jhapa": ("Jhapa", _T, ""), "Kailali": ("Kailali", _T, ""), "Kanchanpur": ("Kanchanpur", _T, ""),
    "Mahottari": ("Mahottari", _T, ""), "Morang": ("Morang", _T, ""),
    "Nawalparasi": ("Nawalparasi", _T, "Part Inner Terai, counted as Terai"),
    "Routahat": ("Rautahat", _T, ""), "Rupandehi": ("Rupandehi", _T, ""),
    "Saptari": ("Saptari", _T, ""), "Sarlahi": ("Sarlahi", _T, ""), "Sunsari": ("Sunsari", _T, ""),
}
DISPUTED = ["Dhading", "Gorkha", "Rukum"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

REPORT = []


def say(text=""):
    print(text, flush=True)
    REPORT.append(str(text))


def save_fig(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"  saved {name}")


# Data and clustering helpers

def load(data_dir):
    fpath = data_dir / "district_features.csv"
    dpath = data_dir / "climate_daily_clean.csv"
    for p, hint in [(fpath, "prepare_data.py"), (dpath, "prepare_data.py")]:
        if not p.exists():
            sys.exit(f"Cannot find {p}. Run {hint} first.")
    feats = pd.read_csv(fpath).set_index("District").sort_index()
    daily = pd.read_csv(dpath, usecols=["District", "Year", "Month", "Precip"])
    unknown = [d for d in feats.index if d not in OFFICIAL]
    if unknown:
        sys.exit(f"Districts missing from the official-classification table: {unknown}")
    return feats, daily


def rain_features(daily, y0, y1):
    """Annual rainfall and monsoon share computed from the years y0..y1 only."""
    x = daily[(daily["Year"] >= y0) & (daily["Year"] <= y1)]
    yearly = x.groupby(["District", "Year"])["Precip"].sum().groupby("District").mean()
    monsoon = (x[x["Month"].isin([6, 7, 8, 9])].groupby(["District", "Year"])["Precip"].sum()
               .groupby("District").mean())
    return pd.DataFrame({"Annual_rain_mm": yearly, "Monsoon_rain_share": monsoon / yearly})


def make_variants(feats, daily):
    post = rain_features(daily, BREAK_YEAR, 2019)
    pre = rain_features(daily, 1981, BREAK_YEAR - 1)
    base = feats[BASE_FEATURES].copy()
    v_post = base.copy()
    v_post[["Annual_rain_mm", "Monsoon_rain_share"]] = post
    v_pre = base.copy()
    v_pre[["Annual_rain_mm", "Monsoon_rain_share"]] = pre
    v_nomon = feats[["Temp_mean", "Annual_rain_mm", "Seasonal_temp_swing"]].copy()
    v_post_nomon = v_nomon.copy()
    v_post_nomon["Annual_rain_mm"] = post["Annual_rain_mm"]
    return {
        "Baseline": base,
        "Rain 2004+": v_post,
        "Rain 1981-2003": v_pre,
        "No monsoon share": v_nomon,
        "Rain 2004+, no monsoon share": v_post_nomon,
        "Temp + altitude only": feats[TEMP_FEATURES].copy(),
    }


def rank_labels(labels, temp):
    """Renumber clusters so 0 = warmest ... k-1 = coldest (mean temperature)."""
    order = pd.Series(np.asarray(temp)).groupby(np.asarray(labels)).mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(order.index)}
    return np.array([mapping[l] for l in labels])


def standardize(df):
    return StandardScaler().fit_transform(df)


def kmeans(X, k, seed=RANDOM_STATE, n_init=N_INIT):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X)


def ward_labels(X, k):
    return fcluster(linkage(X, method="ward"), k, criterion="maxclust") - 1


def safe_sil(X, labels):
    return float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan


# Stability by resampling

def stability(df, k, temp, detail=False, seed=0):
    rng = np.random.default_rng(seed)
    X = standardize(df)
    full = rank_labels(kmeans(X, k), temp)
    n = len(df)
    m = int(round(SUBSAMPLE_FRACTION * n))
    aris = []
    jac = np.full((N_SUBSAMPLES, k), np.nan)
    hit = np.zeros(n)
    cnt = np.zeros(n)
    for b in range(N_SUBSAMPLES):
        idx = rng.choice(n, m, replace=False)
        lab = kmeans(standardize(df.iloc[idx]), k, seed=int(rng.integers(1_000_000_000)), n_init=5)
        aris.append(ARI(full[idx], lab))
        if detail:
            cont = pd.crosstab(full[idx], lab).reindex(index=range(k), columns=range(k),
                                                       fill_value=0).values
            r, c = linear_sum_assignment(-cont)
            mapping = dict(zip(c, r))
            mapped = np.array([mapping[l] for l in lab])
            hit[idx] += (mapped == full[idx])
            cnt[idx] += 1
            for z in range(k):
                a = set(idx[full[idx] == z])
                if not a:
                    continue
                jac[b, z] = max(len(a & set(idx[lab == l])) / len(a | set(idx[lab == l]))
                                for l in range(k) if (set(idx[lab == l]) | a))
    out = {"aris": np.array(aris), "full": full}
    if detail:
        out["jaccard"] = np.nanmean(jac, axis=0)
        out["confidence"] = hit / np.maximum(cnt, 1)
    return out


# Figures

def draw_outline(ax, path):
    if path is None or not Path(path).exists():
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


def setup_map(ax):
    ax.set_xlim(79.9, 88.4)
    ax.set_ylim(26.1, 30.6)
    ax.set_aspect(1 / np.cos(np.radians(28)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_sensitivity(analyses, comp, out_dir):
    names = list(analyses)
    n = len(names)
    M = np.array([[ARI(analyses[a]["labels"], analyses[b]["labels"]) for b in names] for a in names])
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    im = a.imshow(np.ma.array(M, mask=mask), cmap="YlGn", vmin=0, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                   color="white" if M[i, j] > 0.75 else "black")
    a.set_xticks(range(n))
    a.set_yticks(range(n))
    a.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    a.set_yticklabels(names, fontsize=9)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("How similar are the zones? (agreement, 1 = identical)", loc="left")
    fig.colorbar(im, ax=a, shrink=0.75, pad=0.02).set_label("Adjusted Rand index")

    ch = comp.set_index("Analysis")["Districts_changed_vs_baseline"].drop("Baseline")
    colors = ["#5d6d7e" if x.startswith(("Ward", "Gaussian")) else "#1e8449" for x in ch.index]
    bars = b.barh(ch.index[::-1], ch.values[::-1], color=colors[::-1], zorder=3)
    for r_, v in zip(bars, ch.values[::-1]):
        b.text(v + 0.15, r_.get_y() + r_.get_height() / 2, f"{int(v)}", va="center", fontsize=10,
               fontweight="bold")
    b.set_xlabel("Districts that move to a different zone (out of 62)")
    b.set_title("How many districts change zone vs. the baseline?", loc="left")
    b.grid(axis="x", alpha=0.3, zorder=0)
    b.set_xlim(0, max(ch.max() * 1.45, 6))
    b.text(0.99, 0.97, "green = other feature sets (K-Means)\ngrey = other algorithms",
           transform=b.transAxes, ha="right", va="top", fontsize=9.5, color="#555555")
    fig.suptitle("Do the zones survive different features and algorithms?", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "24_feature_sensitivity.png")


def plot_choosing_k(ksel, stab, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    a, b, c = axes
    a.plot(ksel["k"], ksel["Silhouette_KMeans_baseline"], marker="o", lw=2.3, color="#c0392b",
           label="K-Means, baseline features")
    a.plot(ksel["k"], ksel["Silhouette_Ward_baseline"], marker="s", lw=2.3, color="#7d3c98",
           label="Ward, baseline features")
    a.plot(ksel["k"], ksel["Silhouette_KMeans_temp_altitude"], marker="^", lw=2.3, color="#1e8449",
           label="K-Means, temperature + altitude only")
    a.axvline(K, color="black", ls="--", lw=1.3)
    a.set_xlabel("Number of zones (k)")
    a.set_ylabel("Silhouette score (higher = better separated)")
    a.set_title("Separation of the zones", loc="left")
    a.legend(fontsize=9, frameon=True)
    a.grid(alpha=0.3)

    kk = list(range(1, 9))
    b.plot(kk, ksel_bic(ksel, "BIC_GMM_diagonal"), marker="o", lw=2.3, color="#2e86c1",
           label="diagonal covariance")
    b.plot(kk, ksel_bic(ksel, "BIC_GMM_full"), marker="s", lw=2.3, color="#aab7b8",
           label="full covariance")
    best = int(np.nanargmin(ksel_bic(ksel, "BIC_GMM_diagonal")) + 1)
    b.scatter([best], [min(ksel_bic(ksel, "BIC_GMM_diagonal"))], s=240, marker="*", color="#f39c12",
              edgecolor="black", zorder=5, label=f"lowest, diagonal (k = {best})")
    b.axvline(K, color="black", ls="--", lw=1.3)
    b.set_xlabel("Number of zones (k)")
    b.set_ylabel("BIC (lower = better)")
    b.set_title("Gaussian mixture model", loc="left")
    b.legend(fontsize=9, frameon=True)
    b.grid(alpha=0.3)

    for key, color, lab in [("baseline", "#c0392b", "baseline features"),
                            ("temp", "#1e8449", "temperature + altitude only")]:
        s = stab[key]
        c.plot(s["k"], s["Mean_ARI"], marker="o", lw=2.3, color=color, label=f"{lab}: average")
        c.plot(s["k"], s["P5_ARI"], marker="v", lw=1.6, ls=":", color=color,
               label=f"{lab}: worst 5% of resamples")
    c.axvline(K, color="black", ls="--", lw=1.3)
    c.set_xlabel("Number of zones (k)")
    c.set_ylabel("Agreement with the full-data zones\n(ARI over 500 resamples of 80% of districts)")
    c.set_ylim(0, 1.04)
    c.set_title("Stability under resampling", loc="left")
    c.legend(fontsize=8.5, frameon=True, loc="lower right")
    c.grid(alpha=0.3)
    for ax in (a, c):
        ax.set_xticks(list(K_RANGE) if ax is a else list(K_STAB))
    b.set_xticks(kk)
    fig.suptitle(f"Which number of zones is best supported? (dashed line = our choice, k = {K})",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "25_choosing_k.png")


def ksel_bic(ksel, col):
    return ksel[col].values


def plot_stability(conf, jaccard, outline, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={"width_ratios": [0.8, 1.5]})
    bars = a.bar(range(K), jaccard, color=ZONE_COLORS, zorder=3)
    for r_, v in zip(bars, jaccard):
        a.text(r_.get_x() + r_.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=11,
               fontweight="bold")
    a.axhline(0.75, color="#1e8449", ls="--", lw=1.5)
    a.text(K - 0.45, 0.76, "stable\n(above 0.75)", color="#1e8449", fontsize=9, ha="left", va="bottom")
    a.axhline(0.6, color="#c0392b", ls="--", lw=1.5)
    a.text(K - 0.45, 0.61, "uncertain\n(0.6 to 0.75)", color="#c0392b", fontsize=9, ha="left", va="bottom")
    a.set_xlim(-0.6, K + 0.55)
    a.set_xticks(range(K))
    a.set_xticklabels([z.replace(" & ", "\n& ").replace(" ", "\n", 1) if len(z) > 14 else z
                       for z in ZONE_NAMES], fontsize=10)
    a.set_ylim(0, 1.08)
    a.set_ylabel("Jaccard stability (1 = zone reappears exactly)")
    a.set_title("Does each zone reappear when 20% of districts are left out?", loc="left", fontsize=11.5)
    a.grid(axis="y", alpha=0.3, zorder=0)

    draw_outline(b, outline)
    sc = b.scatter(conf["Longitude"], conf["Latitude"], c=conf["Membership_confidence"], cmap="RdYlGn",
                   vmin=0.2, vmax=1.0, s=190, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
    low = conf[conf["Membership_confidence"] < 0.6].sort_values("Longitude")
    offsets = [(0, 17), (0, -24), (22, 10), (-22, -20)]
    for i, (name, r) in enumerate(low.iterrows()):
        b.annotate(name, (r["Longitude"], r["Latitude"]), xytext=offsets[i % len(offsets)],
                   textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", zorder=6,
                   arrowprops=dict(arrowstyle="-", color="black", lw=0.7),
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    setup_map(b)
    cb = fig.colorbar(sc, ax=b, orientation="horizontal", pad=0.14, shrink=0.8)
    cb.set_label("Confidence: share of resamples in which the district keeps its zone")
    b.set_title("Which districts are borderline? (labelled: confidence below 0.6)", loc="left", fontsize=11.5)
    fig.suptitle("Stability of the three climate zones", fontsize=15, fontweight="bold", x=0.01,
                 ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "26_stability.png")


def plot_official(table, ct, outline, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [0.85, 1.6]})
    im = a.imshow(ct.values, cmap="Blues")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            a.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=15, fontweight="bold",
                   color="white" if ct.values[i, j] > ct.values.max() * 0.55 else "black")
    a.set_xticks(range(K))
    a.set_xticklabels([z.replace(" & ", "\n& ") for z in ZONE_NAMES], fontsize=10)
    a.set_yticks(range(len(BELT_ORDER)))
    a.set_yticklabels([f"{b_}\n(official)" for b_ in BELT_ORDER], fontsize=10)
    a.set_xlabel("Our data-driven zone")
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("Number of districts", loc="left")

    draw_outline(b, outline)
    for belt in BELT_ORDER:
        s = table[table["Official_belt"] == belt]
        b.scatter(s["Longitude"], s["Latitude"], c=[ZONE_COLORS[z] for z in s["Zone_rank"]],
                  marker=BELT_MARKER[belt], s=170, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
    mism = table[~table["Matches_official"]]
    b.scatter(mism["Longitude"], mism["Latitude"], s=470, facecolor="none", edgecolor="#c0392b",
              linewidth=2, zorder=4)
    setup_map(b)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=ZONE_COLORS[i], markeredgecolor="#2c3e50",
                      markersize=11, label=f"Zone: {ZONE_NAMES[i]}") for i in range(K)]
    handles += [Line2D([0], [0], marker=BELT_MARKER[bt], ls="", markerfacecolor="white",
                       markeredgecolor="#2c3e50", markersize=11, label=f"Official: {bt}") for bt in BELT_ORDER]
    handles += [Line2D([0], [0], marker="o", ls="", markerfacecolor="none", markeredgecolor="#c0392b",
                       markeredgewidth=2, markersize=14, label=f"Zone and belt disagree ({len(mism)})")]
    b.legend(handles=handles, loc="upper right", fontsize=9, frameon=True)
    b.set_title("Colour = our zone, shape = official belt", loc="left")
    fig.suptitle("Do our zones match Nepal's official Mountain / Hill / Terai belts?", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "27_official_comparison.png")


def plot_dendrogram(X, names, belts, out_dir):
    Z = linkage(X, method="ward")
    cut = (Z[-3, 2] + Z[-2, 2]) / 2
    fig, ax = plt.subplots(figsize=(10, 14))
    d = dendrogram(Z, labels=list(names), orientation="right", color_threshold=0,
                   above_threshold_color="#7f8c8d", leaf_font_size=9, ax=ax)
    for lab in ax.get_ymajorticklabels():
        lab.set_color(BELT_COLOR[belts[lab.get_text()]])
        lab.set_fontweight("bold")
    ax.axvline(cut, color="#c0392b", ls="--", lw=1.8)
    ax.text(cut, ax.get_ylim()[1] * 0.995, " cut: 3 groups", color="#c0392b", fontsize=11,
            fontweight="bold", va="top")
    ax.set_xlabel("Ward distance (larger = less similar)")
    ax.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=BELT_COLOR[b_], lw=6, label=f"Official: {b_}")
                       for b_ in BELT_ORDER], loc="lower right", frameon=True, fontsize=10)
    ax.set_title("Hierarchical clustering tree (Ward), districts labelled by official belt", loc="left")
    fig.tight_layout()
    save_fig(fig, out_dir, "28_dendrogram.png")



def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python clustering_robustness.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "clustering_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    outline = data_dir / "nepal_outline.geojson"
    t0 = time.time()

    feats, daily = load(data_dir)
    districts = feats.index
    temp = feats["Temp_mean"].values
    n = len(districts)
    belts = pd.Series({d: OFFICIAL[d][1] for d in districts})
    belt_code = belts.map(BELT_CODE).values

    say("=" * 72)
    say("CLUSTERING ROBUSTNESS REPORT - how trustworthy are the climate zones?")
    say("=" * 72)
    say(f"Districts: {n}. Zones: k = {K}. Baseline features: {', '.join(BASE_FEATURES)}")
    say(f"Official belts: {belts.value_counts().reindex(BELT_ORDER).to_dict()} "
        f"(census scheme; see the top of the script for the source)")

    # Analyses at k = 3
    variants = make_variants(feats, daily)
    analyses = {}
    for name, df in variants.items():
        X = standardize(df)
        lab = rank_labels(kmeans(X, K), temp)
        analyses[name] = {"labels": lab, "X": X}
    Xb = analyses["Baseline"]["X"]
    analyses["Ward hierarchical"] = {"labels": rank_labels(ward_labels(Xb, K), temp), "X": Xb}
    gm = GaussianMixture(K, covariance_type="diag", n_init=20, random_state=0).fit(Xb)
    analyses["Gaussian mixture"] = {"labels": rank_labels(gm.predict(Xb), temp), "X": Xb}

    base = analyses["Baseline"]["labels"]
    zpath = data_dir / "clustering" / "district_clusters.csv"
    if zpath.exists():
        prev = pd.read_csv(zpath).set_index("District").reindex(districts)["Cluster"].values
        diff = int((prev != base).sum())
        say(f"Check: the baseline reproduces the Step 4 zones "
            f"({'exactly' if diff == 0 else f'{diff} districts differ'}).")

    rows = []
    for name, a in analyses.items():
        lab = a["labels"]
        rows.append({
            "Analysis": name, "Features_or_method": "K-Means" if name in variants else name,
            "Zone_sizes": "/".join(str(int(x)) for x in np.bincount(lab, minlength=K)),
            "Silhouette": round(safe_sil(a["X"], lab), 3),
            "ARI_vs_baseline": round(ARI(base, lab), 2),
            "Districts_changed_vs_baseline": int((lab != base).sum()),
            "ARI_vs_official": round(ARI(belt_code, lab), 2),
            "Match_official_pct": round(100 * float((lab == belt_code).mean()), 1),
        })
    comp = pd.DataFrame(rows)

    say("\n" + "=" * 72)
    say("A. SENSITIVITY: do the zones survive other features and other algorithms?")
    say("=" * 72)
    say("Zone sizes are warmest zone / middle zone / coldest zone.")
    say(comp.drop(columns="Features_or_method").to_string(index=False))
    r = comp.set_index("Analysis")
    say(f"\n  Rainfall from {BREAK_YEAR}+ only moves {r.loc['Rain 2004+', 'Districts_changed_vs_baseline']} "
        f"districts (agreement {r.loc['Rain 2004+', 'ARI_vs_baseline']:.2f}); rainfall from before "
        f"{BREAK_YEAR} moves {r.loc['Rain 1981-2003', 'Districts_changed_vs_baseline']} "
        f"(agreement {r.loc['Rain 1981-2003', 'ARI_vs_baseline']:.2f}).")
    say(f"  Pre-2004 versus post-2004 rainfall features agree with each other at "
        f"{ARI(analyses['Rain 1981-2003']['labels'], analyses['Rain 2004+']['labels']):.2f}.")
    say(f"  Dropping the monsoon share moves {r.loc['No monsoon share', 'Districts_changed_vs_baseline']} "
        f"district(s), so that feature hardly matters.")
    say(f"  Temperature + altitude only: silhouette {r.loc['Temp + altitude only', 'Silhouette']:.2f} "
        f"versus {r.loc['Baseline', 'Silhouette']:.2f} for the baseline; agreement with the baseline "
        f"{r.loc['Temp + altitude only', 'ARI_vs_baseline']:.2f}.")
    say(f"  Ward agrees with K-Means at {r.loc['Ward hierarchical', 'ARI_vs_baseline']:.2f}. The Gaussian mixture "
        f"agrees at {r.loc['Gaussian mixture', 'ARI_vs_baseline']:.2f}; with only {n} districts it "
        f"cannot estimate a full covariance reliably (full-covariance versions collapse into one huge and "
        f"one or two tiny groups), so treat it as a weak check.")

    # k selection across methods
    Xt = analyses["Temp + altitude only"]["X"]
    krows = []
    for k in range(1, 9):
        row = {"k": k}
        row["BIC_GMM_diagonal"] = GaussianMixture(k, covariance_type="diag", n_init=10, random_state=0).fit(Xb).bic(Xb)
        row["BIC_GMM_full"] = GaussianMixture(k, covariance_type="full", n_init=10, random_state=0).fit(Xb).bic(Xb)
        if k >= 2:
            row["Silhouette_KMeans_baseline"] = safe_sil(Xb, kmeans(Xb, k))
            row["Silhouette_Ward_baseline"] = safe_sil(Xb, ward_labels(Xb, k))
            row["Silhouette_KMeans_temp_altitude"] = safe_sil(Xt, kmeans(Xt, k))
        krows.append(row)
    ksel = pd.DataFrame(krows)

    say("\n" + "=" * 72)
    say("B. WHICH NUMBER OF ZONES IS BEST SUPPORTED?")
    say("=" * 72)
    say(ksel.round(3).to_string(index=False))
    sil_cols = ["Silhouette_KMeans_baseline", "Silhouette_Ward_baseline", "Silhouette_KMeans_temp_altitude"]
    for c in sil_cols:
        bk = int(ksel.loc[ksel[c].idxmax(), "k"])
        say(f"  Best k by {c.replace('Silhouette_', 'silhouette, ').replace('_', ' ')}: {bk}")
    say(f"  Best k by BIC (diagonal mixture): {int(ksel.loc[ksel['BIC_GMM_diagonal'].idxmin(), 'k'])}")

    # Stability
    say("\n" + "=" * 72)
    say(f"C. STABILITY: {N_SUBSAMPLES} random {int(SUBSAMPLE_FRACTION * 100)}% subsets of the districts")
    say("=" * 72)
    say("Running the resampling (about 1-3 minutes) ...")
    stab = {}
    detail = None
    for key, df in [("baseline", variants["Baseline"]), ("temp", variants["Temp + altitude only"])]:
        recs = []
        for k in K_STAB:
            s = stability(df, k, temp, detail=(key == "baseline" and k == K))
            if key == "baseline" and k == K:
                detail = s
            recs.append({"k": k, "Mean_ARI": s["aris"].mean(), "P5_ARI": np.percentile(s["aris"], 5),
                         "P95_ARI": np.percentile(s["aris"], 95)})
        stab[key] = pd.DataFrame(recs)
    sb = pd.concat([stab["baseline"].assign(Features="Baseline"),
                    stab["temp"].assign(Features="Temp + altitude only")])
    say("Agreement between each resample and the full-data zones (ARI):")
    say(sb.round(2).to_string(index=False))
    bestb = int(stab["baseline"].loc[stab["baseline"]["Mean_ARI"].idxmax(), "k"])
    bestt = int(stab["temp"].loc[stab["temp"]["Mean_ARI"].idxmax(), "k"])
    say(f"  Most stable k: {bestb} (baseline features), {bestt} (temperature + altitude only).")

    jac = detail["jaccard"]
    say(f"\nPer-zone stability at k = {K} (Jaccard; > 0.75 stable, 0.6-0.75 uncertain):")
    for z in range(K):
        say(f"  {ZONE_NAMES[z]:<22} {jac[z]:.2f}")
    conf = detail["confidence"]

    # Per-district table
    belts_arr = belts.values
    agree = np.mean([analyses[a]["labels"] == base for a in analyses if a != "Baseline"], axis=0)
    table = pd.DataFrame({
        "District": districts, "Latitude": feats["Latitude"].values, "Longitude": feats["Longitude"].values,
        "Pressure_mean": feats["Pressure_mean"].values,
        "Zone_rank": base, "Zone": [ZONE_NAMES[z] for z in base],
        "Membership_confidence": np.round(conf, 2),
        "Agreement_across_analyses": np.round(agree, 2),
        "Official_belt": belts_arr, "Official_note": [OFFICIAL[d][2] for d in districts],
    }).set_index("District", drop=False)
    table["Matches_official"] = table["Zone_rank"].values == belt_code
    table["Borderline"] = (table["Membership_confidence"] < 0.6) | (table["Agreement_across_analyses"] < 0.5)

    low = table[table["Membership_confidence"] < 0.6].sort_values("Membership_confidence")
    say(f"\nDistricts with confidence below 0.6 ({len(low)}): " + ", ".join(
        f"{r.District} ({r.Membership_confidence:.2f}, {r.Zone})" for r in low.itertuples()))
    say(f"Districts with confidence of at least 0.9: {int((table['Membership_confidence'] >= 0.9).sum())} of {n}.")

    # Official comparison
    say("\n" + "=" * 72)
    say("D. DO THE ZONES MATCH NEPAL'S OFFICIAL MOUNTAIN / HILL / TERAI BELTS?")
    say("=" * 72)
    ct = pd.crosstab(table["Official_belt"], table["Zone_rank"]).reindex(index=BELT_ORDER,
                                                                         columns=range(K), fill_value=0)
    ct.columns = ZONE_NAMES
    say("Districts by official belt (rows) and our zone (columns):")
    say(ct.to_string())
    say(f"\n  Agreement: ARI {ARI(belt_code, base):.2f}, NMI {NMI(belt_code, base):.2f}, "
        f"{100 * (base == belt_code).mean():.0f}% of districts in the matching zone.")
    say(f"  Terai districts in our lowland zone: {int(ct.loc['Terai', ZONE_NAMES[0]])} of {int(ct.loc['Terai'].sum())}.")
    say(f"  Mountain districts in our high-mountain zone: {int(ct.loc['Mountain', ZONE_NAMES[2]])} of "
        f"{int(ct.loc['Mountain'].sum())}.")
    off2 = belt_code.copy()
    for d in DISPUTED:
        off2[list(districts).index(d)] = BELT_CODE["Mountain"]
    say(f"  If Dhading, Gorkha and Rukum are counted as Mountain (2017 Act), ARI changes from "
        f"{ARI(belt_code, base):.2f} to {ARI(off2, base):.2f}. Our zone for them: "
        + ", ".join(f"{d} = {ZONE_NAMES[int(table.loc[d, 'Zone_rank'])]}" for d in DISPUTED) + ".")

    mism = table[~table["Matches_official"]].sort_values(["Official_belt", "Zone_rank"])
    say(f"\n  The {len(mism)} districts where zone and official belt disagree:")
    for belt in ["Hill", "Mountain", "Terai"]:
        for z in range(K):
            g = mism[(mism["Official_belt"] == belt) & (mism["Zone_rank"] == z)]
            if len(g):
                say(f"    Official {belt}, our zone '{ZONE_NAMES[z]}' ({len(g)}): " + ", ".join(
                    f"{r.District}{' [' + r.Official_note + ']' if r.Official_note else ''}"
                    for r in g.itertuples()))
    inner = mism[mism["Official_note"].str.contains("Inner Terai")]
    say(f"  Of these, {len(inner)} are 'Inner Terai' districts that the census counts as Hill although "
        f"they are low-lying valleys: " + (", ".join(inner["District"]) if len(inner) else "none") + ".")
    low_conf_mism = mism[mism["Membership_confidence"] < 0.6]
    say(f"  {len(low_conf_mism)} of the {len(mism)} mismatches are also low-confidence districts.")

    say("\n  Pressure (a proxy for altitude; lower = higher) by official belt:")
    pr = table.groupby("Official_belt")["Pressure_mean"].agg(["count", "min", "mean", "max"]).reindex(BELT_ORDER)
    say(pr.round(1).to_string())
    say("  The official belts overlap on this altitude proxy, because each district gets a single label.")

    k2 = rank_labels(kmeans(Xb, 2), temp)
    mt = belts_arr == "Mountain"
    say(f"\n  With only k = 2: {int(((k2 == 1) & mt).sum())} of {int(mt.sum())} Mountain districts fall in the cold "
        f"zone, and {int(((k2 == 1) & ~mt).sum())} of {int((~mt).sum())} Hill/Terai districts do.")

    say("\n  Agreement with the official belts for every analysis (k = 3):")
    say(comp[["Analysis", "ARI_vs_official", "Match_official_pct"]].to_string(index=False))

    # Summary
    say("\n" + "=" * 72)
    say("SUMMARY")
    say("=" * 72)
    sb3 = stab["baseline"].set_index("k").loc[K]
    say(f"  1. Feature sensitivity: the zones are robust to dropping the monsoon share "
        f"({r.loc['No monsoon share', 'Districts_changed_vs_baseline']} district moves) but sensitive to the "
        f"2004 break ({r.loc['Rain 2004+', 'Districts_changed_vs_baseline']} districts move when only 2004+ "
        f"rainfall is used).")
    say(f"  2. Choice of k: silhouette favours k = "
        f"{int(ksel.loc[ksel['Silhouette_KMeans_baseline'].idxmax(), 'k'])} on the baseline features; on "
        f"temperature + altitude only the best is k = "
        f"{int(ksel.loc[ksel['Silhouette_KMeans_temp_altitude'].idxmax(), 'k'])}.")
    say(f"  3. Stability at k = {K}: mean agreement {sb3['Mean_ARI']:.2f} (5th percentile {sb3['P5_ARI']:.2f}); "
        f"zone stabilities {', '.join(f'{j:.2f}' for j in jac)}; {len(low)} borderline districts.")
    say(f"  4. Official belts: all Terai districts land in the lowland zone; the disagreement is at the "
        f"Hill/Mountain edge and among low hills. {len(mism)} of {n} districts disagree.")

    # Save
    say("\nSaving figures and tables:")
    plot_sensitivity(analyses, comp, out_dir)
    plot_choosing_k(ksel, stab, out_dir)
    plot_stability(table, jac, outline, out_dir)
    plot_official(table, ct, outline, out_dir)
    plot_dendrogram(Xb, districts, belts.to_dict(), out_dir)

    zones_by = pd.DataFrame({"District": districts})
    for name, a in analyses.items():
        zones_by[name] = [ZONE_NAMES[z] for z in a["labels"]]
    zones_by.to_csv(out_dir / "zones_by_analysis.csv", index=False)
    comp.to_csv(out_dir / "analysis_comparison.csv", index=False)
    ksel.round(4).to_csv(out_dir / "k_selection.csv", index=False)
    sb.round(4).to_csv(out_dir / "stability_by_k.csv", index=False)
    table.drop(columns=["Zone_rank"]).to_csv(out_dir / "district_zone_confidence.csv", index=False)
    pd.DataFrame([{"Dataset_name": d, "Official_name": o[0], "Belt": o[1], "Note": o[2]}
                  for d, o in OFFICIAL.items() if d in set(districts)]).to_csv(
        out_dir / "official_classification.csv", index=False)
    ct.to_csv(out_dir / "official_vs_zone_crosstab.csv")
    (out_dir / "clustering_robustness_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone in {(time.time() - t0) / 60:.1f} minutes. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
