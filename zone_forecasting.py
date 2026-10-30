
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Settings

SPLIT_DATE = pd.Timestamp("2015-01-01")  
VARIABLES = {
    "Temp_mean": ("Temperature", "RMSE (C)"),
    "Precip_total": ("Rainfall", "RMSE (% of average monthly rainfall)"),
}
SARIMA_SPEC = {
    "Temp_mean": ((1, 0, 0), (0, 1, 1, 12)),
    "Precip_total": ((2, 0, 0), (1, 1, 1, 12)),
}
FULL, POST = "Full history (1981-2014)", "Post-2004 only (2004-2014)"
WINDOWS = {FULL: 1981, POST: 2004}
M_SN, M_CL, M_SA = "Seasonal-naive", "Climatology", "SARIMA"
ZONE_COLORS = ["#D55E00", "#009E73", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9"]
SOURCE = "Source: Nepal Daily Climate Dataset (Kaggle), 1981-2019, 62 districts"

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
    print(text)
    REPORT.append(str(text))


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def save_fig(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"  saved {name}")


# Models

def seasonal_naive(train, index):
    last = train.iloc[-12:]
    lookup = {d.month: v for d, v in zip(last.index, last.values)}
    return pd.Series([lookup[d.month] for d in index], index=index)


def climatology(train, index):
    avg = train.groupby(train.index.month).mean()
    return pd.Series(avg.reindex([d.month for d in index]).values, index=index)


def sarima(train, index, var):
    order, sorder = SARIMA_SPEC[var]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(train, order=order, seasonal_order=sorder,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False, maxiter=200)
            f = res.get_forecast(steps=len(index)).predicted_mean.values
    except Exception:
        return pd.Series(np.nan, index=index)
    f = pd.Series(f, index=index)
    return f.clip(lower=0.0) if var == "Precip_total" else f


def metrics(actual, pred, var):
    err = np.asarray(pred, float) - np.asarray(actual, float)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    out = {"RMSE": rmse, "MAE": float(np.mean(np.abs(err))), "Bias": float(np.mean(err))}
    if var == "Precip_total":
        out["Error"] = 100 * rmse / float(np.mean(actual))    # % of average monthly rainfall
        out["Bias_pct"] = 100 * out["Bias"] / float(np.mean(actual))
    else:
        out["Error"] = rmse
        out["Bias_pct"] = np.nan
    return out


def run_district(y, var):
    """All models for one district and one variable. Returns a list of result dicts."""
    train_full = y[y.index < SPLIT_DATE]
    test = y[y.index >= SPLIT_DATE]
    rows = []

    def add(model, window, pred):
        rows.append({"Model": model, "Window": window, **metrics(test, pred, var)})

    add(M_SN, "Last training year", seasonal_naive(train_full, test.index))
    for wname, start in WINDOWS.items():
        train = train_full[train_full.index.year >= start]
        add(M_CL, wname, climatology(train, test.index))
        add(M_SA, wname, sarima(train, test.index, var))
    return rows


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


def plot_by_zone(res, zones, reps, kw, out_dir):
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, (var, (vname, ylab)) in zip(axes, VARIABLES.items()):
        sub = res[(res["Variable"] == var) & (res["Model"] == M_SA) & (res["Window"] == FULL)]
        data = [sub.loc[sub["Zone"] == z, "Error"].values for z in zones]
        bp = ax.boxplot(data, positions=range(len(zones)), widths=0.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=2.2), zorder=2)
        for patch, c in zip(bp["boxes"], ZONE_COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for i, z in enumerate(zones):
            s = sub[sub["Zone"] == z]
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(s)), s["Error"], s=48, color=ZONE_COLORS[i],
                       edgecolor="#2c3e50", linewidth=0.6, zorder=3)
            for _, r in s[s["District"].isin(reps)].iterrows():
                ax.scatter(i, r["Error"], s=520, marker="*", color="#f1c40f", edgecolor="black",
                           linewidth=1.2, zorder=5)
                ax.annotate(r["District"], (i, r["Error"]), xytext=(18, 4), textcoords="offset points",
                            fontsize=9.5, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xticks(range(len(zones)))
        ax.set_xticklabels([f"{z}\n(n = {len(d)})" for z, d in zip(zones, data)])
        ax.set_ylabel(ylab + ", lower is better")
        p = kw[var]
        ax.set_title(f"{vname}: zones differ ({pfmt(p)})" if p < 0.05 else
                     f"{vname}: no clear difference between zones ({pfmt(p)})", loc="left")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    fig.suptitle("Is forecasting harder in some climate zones? (SARIMA, full history, "
                 "62 districts, 2015-2019)", fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.text(0.01, 0.955, "Each dot is a district; stars are the three districts forecast in Step 5.",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "17_errors_by_zone.png")


def plot_full_vs_post(res, zones, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    zone_color = dict(zip(zones, ZONE_COLORS))
    for r_i, (var, (vname, ylab)) in enumerate(VARIABLES.items()):
        for c_i, model in enumerate([M_CL, M_SA]):
            ax = axes[r_i, c_i]
            a = res[(res["Variable"] == var) & (res["Model"] == model) & (res["Window"] == FULL)]
            b = res[(res["Variable"] == var) & (res["Model"] == model) & (res["Window"] == POST)]
            m = a.merge(b, on=["District", "Zone"], suffixes=("_full", "_post")).dropna(
                subset=["Error_full", "Error_post"])
            for z in zones:
                s = m[m["Zone"] == z]
                ax.scatter(s["Error_full"], s["Error_post"], s=55, color=zone_color[z],
                           edgecolor="#2c3e50", linewidth=0.6, label=z, zorder=3)
            lo = min(m["Error_full"].min(), m["Error_post"].min()) * 0.93
            hi = max(m["Error_full"].max(), m["Error_post"].max()) * 1.07
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.3, zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            better = int((m["Error_post"] < m["Error_full"]).sum())
            chg = 100 * (m["Error_post"] / m["Error_full"] - 1).median()
            ax.text(0.04, 0.96, f"Post-2004 training is better in\n{better} of {len(m)} districts "
                    f"(median change {chg:+.0f}%)", transform=ax.transAxes, va="top", fontsize=10,
                    fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
            ax.set_title(f"{vname}, {model}", loc="left")
            ax.set_xlabel("Error when trained on the full history")
            ax.set_ylabel("Error when trained on 2004-2014 only")
            ax.grid(alpha=0.3)
            if r_i == 0 and c_i == 0:
                ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("What does the 2004 break cost? Same test months, two training windows",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.text(0.01, 0.975, "Points below the dashed line: training on 2004 onward gave the smaller "
             "error. Rainfall error is in % of average monthly rainfall; temperature in C.",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_fig(fig, out_dir, "18_full_vs_post2004.png")


def plot_maps(res, info, outline, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    for ax, (var, (vname, ylab)) in zip(axes, VARIABLES.items()):
        sub = res[(res["Variable"] == var) & (res["Model"] == M_SA) & (res["Window"] == FULL)]
        sub = sub.merge(info[["District", "Latitude", "Longitude"]], on="District")
        draw_outline(ax, outline)
        sc = ax.scatter(sub["Longitude"], sub["Latitude"], c=sub["Error"], cmap="YlOrRd", s=170,
                        edgecolor="#2c3e50", linewidth=0.7, zorder=3)
        ax.set_xlim(79.9, 88.4)
        ax.set_ylim(26.1, 30.6)
        ax.set_aspect(1 / np.cos(np.radians(28)))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{vname} forecast error (SARIMA, full history)", loc="left")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.17, shrink=0.85)
        cb.set_label(ylab)
    fig.suptitle("Where is forecasting easy or hard?", fontsize=15, fontweight="bold",
                 x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "19_error_maps.png")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python zone_forecasting.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "zone_forecasting"
    out_dir.mkdir(parents=True, exist_ok=True)

    mpath = data_dir / "climate_monthly.csv"
    cpath = data_dir / "clustering" / "district_clusters.csv"
    for p, hint in [(mpath, "prepare_data.py"), (cpath, "clustering.py")]:
        if not p.exists():
            sys.exit(f"Cannot find {p}. Run {hint} first.")
    monthly = pd.read_csv(mpath, parse_dates=["Date"])
    info = pd.read_csv(cpath)
    zone_order = info.sort_values("Cluster")["Zone"].drop_duplicates().tolist()   # warmest first
    zone_of = dict(zip(info["District"], info["Zone"]))
    rep_path = data_dir / "clustering" / "representative_districts.csv"
    reps = pd.read_csv(rep_path)["District"].tolist() if rep_path.exists() else []
    outline = data_dir / "nepal_outline.geojson"

    say("=" * 70)
    say("ZONE FORECASTING REPORT - all districts, with and without the 2004 break")
    say("=" * 70)
    say(f"Districts: {info['District'].nunique()}   Zones: {', '.join(zone_order)}")
    say("Training: full history 1981-2014 (main) and 2004-2014 only (comparison)")
    say("Testing : 2015-01 to 2019-12 (60 months), forecast all at once")
    say(f"Representative districts from Step 5: {', '.join(reps) if reps else '(none found)'}")

    rows = []
    districts = sorted(info["District"])
    say(f"\nFitting models for {len(districts)} districts (about 2-5 minutes) ...")
    for i, d in enumerate(districts, 1):
        s = monthly[monthly["District"] == d].set_index("Date").sort_index().asfreq("MS")
        for var in VARIABLES:
            for r in run_district(s[var], var):
                rows.append({"District": d, "Zone": zone_of[d], "Variable": var, **r})
        if i % 10 == 0 or i == len(districts):
            say(f"  {i}/{len(districts)} districts done")
    res = pd.DataFrame(rows)

    failed = res[(res["Model"] == M_SA) & res["Error"].isna()]
    if len(failed):
        say(f"WARNING: SARIMA failed for {len(failed)} district/variable/window combinations "
            f"(left out of the comparisons).")

    # Q1: difficulty by zone
    say("\n" + "=" * 70)
    say("Q1. DOES FORECASTING DIFFICULTY DIFFER BETWEEN ZONES?")
    say("=" * 70)
    kw = {}
    zsum = []
    for var, (vname, ylab) in VARIABLES.items():
        say(f"\n--- {vname.upper()}: median error per zone ({ylab}) ---")
        sub = res[res["Variable"] == var]
        tab = sub.groupby(["Zone", "Model", "Window"])["Error"].median().reset_index()
        tab["Column"] = tab["Model"].where(tab["Model"] == M_SN, tab["Model"] + " | " + tab["Window"].str.split(" ").str[0])
        wide = tab.pivot(index="Zone", columns="Column", values="Error").reindex(zone_order)
        say(wide.round(2).to_string())
        sa = sub[(sub["Model"] == M_SA) & (sub["Window"] == FULL)]
        groups = [sa.loc[sa["Zone"] == z, "Error"].dropna().values for z in zone_order]
        h, p = stats.kruskal(*groups)
        kw[var] = p
        say(f"Kruskal-Wallis test (SARIMA, full history) across zones: {pfmt(p)}  "
            f"({'zones differ' if p < 0.05 else 'no clear difference'}; "
            "zones were built from climate features, so treat this as descriptive)")
        for (z, m, w), g in sub.groupby(["Zone", "Model", "Window"]):
            zsum.append({"Variable": vname, "Zone": z, "Model": m, "Window": w, "Districts": g["Error"].notna().sum(),
                         "Median_error": g["Error"].median(), "Q1": g["Error"].quantile(0.25),
                         "Q3": g["Error"].quantile(0.75)})

    # Q2: representatives
    say("\n" + "=" * 70)
    say("Q2. WERE THE THREE REPRESENTATIVE DISTRICTS TYPICAL?")
    say("=" * 70)
    say("Position of each representative within its zone (SARIMA, full history).")
    say("'Percentile' = share of districts in the same zone with a SMALLER error (50% = typical).")
    for d in reps:
        for var, (vname, _) in VARIABLES.items():
            sa = res[(res["Variable"] == var) & (res["Model"] == M_SA) & (res["Window"] == FULL)]
            z = zone_of[d]
            zs = sa[sa["Zone"] == z]
            mine = zs.loc[zs["District"] == d, "Error"].iloc[0]
            pct = 100 * (zs["Error"] < mine).sum() / max(len(zs) - 1, 1)
            say(f"  {d:<12} {vname:<12} error {mine:6.2f}   zone median {zs['Error'].median():6.2f}   "
                f"percentile {pct:3.0f}%   ({z})")

    # Q3: full vs post-2004
    say("\n" + "=" * 70)
    say("Q3. WHAT DOES THE 2004 BREAK COST? (full history vs 2004-2014 only)")
    say("=" * 70)
    q3 = []
    for var, (vname, _) in VARIABLES.items():
        for model in [M_CL, M_SA]:
            a = res[(res["Variable"] == var) & (res["Model"] == model) & (res["Window"] == FULL)]
            b = res[(res["Variable"] == var) & (res["Model"] == model) & (res["Window"] == POST)]
            m = a.merge(b, on="District", suffixes=("_f", "_p")).dropna(subset=["Error_f", "Error_p"])
            chg = 100 * (m["Error_p"] / m["Error_f"] - 1)
            row = {"Variable": vname, "Model": model, "Districts": len(m),
                   "Post2004_better_in": int((m["Error_p"] < m["Error_f"]).sum()),
                   "Median_error_full": m["Error_f"].median(), "Median_error_post2004": m["Error_p"].median(),
                   "Median_change_pct": chg.median()}
            if var == "Precip_total":
                row["Median_bias_full_pct"] = m["Bias_pct_f"].median()
                row["Median_bias_post2004_pct"] = m["Bias_pct_p"].median()
            q3.append(row)
    q3 = pd.DataFrame(q3)
    say(q3.round(1).to_string(index=False))
    say("Bias columns: forecast minus actual as % of average monthly rainfall (negative = too low).")

    sn = res[(res["Model"] == M_SN) & (res["Variable"] == "Precip_total")]["Error"].median()
    say(f"\nFor reference, the seasonal-naive baseline's median rainfall error is {sn:.1f}%.")

    # Q4: do the models beat the baselines?
    say("\n" + "=" * 70)
    say("Q4. DOES SARIMA BEAT THE SIMPLE BASELINES? (full-history training, all districts)")
    say("=" * 70)
    for var, (vname, ylab) in VARIABLES.items():
        pv = res[res["Variable"] == var].pivot_table(index="District", columns=["Model", "Window"],
                                                      values="Error")
        sa, cl, sn = pv[(M_SA, FULL)], pv[(M_CL, FULL)], pv[(M_SN, "Last training year")]
        n = int(sa.notna().sum())
        say(f"{vname}: median error  seasonal-naive {sn.median():.2f} | climatology {cl.median():.2f} "
            f"| SARIMA {sa.median():.2f}")
        say(f"   SARIMA beats climatology in {int((sa < cl).sum())} of {n} districts, "
            f"and seasonal-naive in {int((sa < sn).sum())} of {n}.")

    # save and plot
    say("\nSaving figures and tables:")
    plot_by_zone(res, zone_order, reps, kw, out_dir)
    plot_full_vs_post(res, zone_order, out_dir)
    plot_maps(res, info, outline, out_dir)
    res.round(4).to_csv(out_dir / "zone_forecast_metrics.csv", index=False)
    pd.DataFrame(zsum).round(3).to_csv(out_dir / "zone_summary.csv", index=False)
    (out_dir / "zone_forecasting_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
