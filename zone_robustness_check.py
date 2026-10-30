import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# Settings

ZONE_NAMES = ["Lowlands & low hills", "Hills", "High mountains"]
ZONE_COLORS = ["#D55E00", "#009E73", "#0072B2"]
FULL = "Full history (1981-2014)"
POST = "Post-2004 only (2004-2014)"
VARIABLES = {"Temp_mean": ("Temperature", "RMSE (C)"),
             "Precip_total": ("Rainfall", "RMSE (% of average monthly rainfall)")}
ALT_COLUMN = "Temp + altitude only"       
MIN_CONFIDENCE = 0.6
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
    print(text, flush=True)
    REPORT.append(str(text))


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def save_fig(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"  saved {name}")


# Loading

def need(path, hint):
    if not path.exists():
        sys.exit(f"Cannot find {path}. Run {hint} first.")
    return path


def load(data_dir):
    metrics = pd.read_csv(need(data_dir / "zone_forecasting" / "zone_forecast_metrics.csv",
                               "zone_forecasting.py"))
    base = pd.read_csv(need(data_dir / "clustering" / "district_clusters.csv", "clustering.py"))
    alt = pd.read_csv(need(data_dir / "clustering_robustness" / "zones_by_analysis.csv",
                           "clustering_robustness.py"))
    conf = pd.read_csv(need(data_dir / "clustering_robustness" / "district_zone_confidence.csv",
                            "clustering_robustness.py"))
    monthly = pd.read_csv(need(data_dir / "climate_monthly.csv", "prepare_data.py"),
                          parse_dates=["Date"])
    reps_path = data_dir / "clustering" / "representative_districts.csv"
    reps = pd.read_csv(reps_path)["District"].tolist() if reps_path.exists() else []
    return (metrics, base.set_index("District"), alt.set_index("District"),
            conf.set_index("District"), monthly, reps)


def zone_systems(base, alt, conf):
    """The three ways of assigning districts to zones (District -> zone name)."""
    b = base["Zone"]
    a = alt[ALT_COLUMN].reindex(b.index)
    core = (conf["Membership_confidence"].reindex(b.index) >= MIN_CONFIDENCE) & (a == b)
    systems = {
        "Baseline zones (Step 4)": b,
        "Temperature + altitude zones": a,
        "Stable core (baseline zones)": b[core],
    }
    return systems, core


# Analysis pieces

def error_tables(metrics, monthly):
    """Main error (as in zone_forecasting.py) and error relative to variability."""
    sa = metrics[(metrics["Model"] == "SARIMA") & (metrics["Window"] == FULL)]
    main = sa.pivot(index="District", columns="Variable", values="Error")
    rmse = sa.pivot(index="District", columns="Variable", values="RMSE")
    test = monthly[(monthly["Date"] >= "2015-01-01") & (monthly["Date"] < "2020-01-01")]
    sd = test.groupby("District")[list(VARIABLES)].std()
    rel = rmse / sd.reindex(rmse.index)
    return main, rel, test.groupby("District")["Precip_total"].mean()


def zone_stats(err, zones):
    """Median error per zone and a Kruskal-Wallis test, for one variable and one zone system."""
    df = pd.DataFrame({"Error": err, "Zone": zones}).dropna()
    groups = [df.loc[df["Zone"] == z, "Error"].values for z in ZONE_NAMES]
    groups = [g for g in groups if len(g)]
    h, p = stats.kruskal(*groups)
    n = len(df)
    summary = (df.groupby("Zone")["Error"].agg(["count", "median",
                                                lambda s_: s_.quantile(0.25),
                                                lambda s_: s_.quantile(0.75)])
               .reindex(ZONE_NAMES))
    summary.columns = ["Districts", "Median_error", "Q1", "Q3"]
    return summary, h, p, h / (n - 1), df


def ordering(summary):
    s_ = summary["Median_error"].dropna().sort_values()
    return " < ".join(s_.index)


def geography_control(err, zones, lon):
    """Do zones add anything beyond west-to-east position (and vice versa)?"""
    df = pd.DataFrame({"err": err, "zone": zones, "lon": lon}).dropna()
    m_lon = smf.ols("err ~ lon", df).fit()
    m_zone = smf.ols("err ~ C(zone)", df).fit()
    m_both = smf.ols("err ~ lon + C(zone)", df).fit()
    return {"R2_zone_only": m_zone.rsquared, "R2_longitude_only": m_lon.rsquared,
            "R2_both": m_both.rsquared,
            "p_zone_adds_to_longitude": anova_lm(m_lon, m_both)["Pr(>F)"].iloc[1],
            "p_longitude_adds_to_zone": anova_lm(m_zone, m_both)["Pr(>F)"].iloc[1]}


def spear(x, y):
    ok = x.notna() & y.notna()
    rho, p = stats.spearmanr(x[ok], y[ok])
    return float(rho), float(p)


# Figures

def plot_definitions(err, systems, reps, out_dir, tests, ylabels, title, subtitle, fname, show_reps=True):
    names = list(systems)
    fig, axes = plt.subplots(2, len(names), figsize=(6.4 * len(names), 10.2), sharey="row")
    rng = np.random.default_rng(1)
    for r_i, (var, (vname, _)) in enumerate(VARIABLES.items()):
        for c_i, sname in enumerate(names):
            ax = axes[r_i, c_i]
            summary, h, p, eps, df = tests[(sname, var)]
            data = [df.loc[df["Zone"] == z, "Error"].values for z in ZONE_NAMES]
            bp = ax.boxplot(data, positions=range(3), widths=0.55, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=2.2), zorder=2)
            for patch, col in zip(bp["boxes"], ZONE_COLORS):
                patch.set_facecolor(col)
                patch.set_alpha(0.35)
            for i, z in enumerate(ZONE_NAMES):
                s_ = df[df["Zone"] == z]
                ax.scatter(i + rng.uniform(-0.17, 0.17, len(s_)), s_["Error"], s=40, color=ZONE_COLORS[i],
                           edgecolor="#2c3e50", linewidth=0.5, zorder=3)
                if show_reps:
                    for d in reps:
                        if d in s_.index:
                            ax.scatter(i, s_.loc[d, "Error"], s=420, marker="*", color="#f1c40f",
                                       edgecolor="black", linewidth=1.1, zorder=5)
                            ax.annotate(d, (i, s_.loc[d, "Error"]), xytext=(15, 3),
                                        textcoords="offset points", fontsize=8.5, fontweight="bold",
                                        zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                                            ec="none", alpha=0.85))
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"{z.replace(' & ', ' &' + chr(10))}\n(n = {len(d_)})"
                                for z, d_ in zip(ZONE_NAMES, data)], fontsize=9)
            if c_i == 0:
                ax.set_ylabel(ylabels[var])
            ax.set_title(f"{sname}", loc="left", fontsize=11.5)
            ax.text(0.97, 0.97, f"{pfmt(p)}\neffect size {eps:.2f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=9.5, bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
            ax.grid(axis="y", alpha=0.3, zorder=0)
            top = max(tests[(sn, var)][4]["Error"].max() for sn in names)
            ax.set_ylim(0, top * 1.32)          # headroom so the label box never covers points
        axes[r_i, 0].text(-0.2, 0.5, vname, transform=axes[r_i, 0].transAxes, rotation=90, va="center",
                          fontsize=14, fontweight="bold")
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.text(0.01, 0.97, subtitle, fontsize=10, color="#555555")
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    save_fig(fig, out_dir, fname)


def plot_geography(err, base, reps, corr, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for r_i, (var, (vname, ylab)) in enumerate(VARIABLES.items()):
        for c_i, (pred, xlab, flip) in enumerate([
                ("Pressure_mean", "Surface pressure (kPa); reversed so that higher altitude is to the right", True),
                ("Longitude", "Longitude (west to east)", False)]):
            ax = axes[r_i, c_i]
            df = pd.DataFrame({"Error": err[var], "X": base[pred], "Zone": base["Zone"]}).dropna()
            for i, z in enumerate(ZONE_NAMES):
                s_ = df[df["Zone"] == z]
                ax.scatter(s_["X"], s_["Error"], s=55, color=ZONE_COLORS[i], edgecolor="#2c3e50",
                           linewidth=0.6, label=z, zorder=3)
            for d in reps:
                if d in df.index:
                    ax.scatter(df.loc[d, "X"], df.loc[d, "Error"], s=360, marker="*", color="#f1c40f",
                               edgecolor="black", linewidth=1.1, zorder=5)
                    ax.annotate(d, (df.loc[d, "X"], df.loc[d, "Error"]), xytext=(9, 6),
                                textcoords="offset points", fontsize=9, fontweight="bold", zorder=6)
            df["bin"] = pd.qcut(df["X"], 4, labels=False)
            med = df.groupby("bin").agg(X=("X", "median"), E=("Error", "median"))
            ax.plot(med["X"], med["E"], color="black", lw=2.4, marker="D", ms=8, zorder=4,
                    label="median of each quarter of districts")
            c = corr[(corr["Variable"] == vname) & (corr["Predictor"] == pred)].iloc[0]
            rho = -c["Spearman_rho"] if flip else c["Spearman_rho"]
            what = "altitude" if flip else "longitude"
            ax.text(0.03, 0.96, f"Spearman correlation with {what}: {rho:+.2f} ({pfmt(c['p_value'])})",
                    transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
            if flip:
                ax.invert_xaxis()
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(f"{vname}: error vs {'altitude proxy' if flip else 'longitude'}", loc="left")
            ax.grid(alpha=0.3)
            if r_i == 0 and c_i == 0:
                ax.legend(fontsize=8.5, frameon=True, loc="lower right")
    fig.suptitle("Without zones: error rises strongly toward the west; altitude shows no straight-line trend",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "30_altitude_and_longitude.png")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python zone_robustness_check.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "zone_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, base, alt, conf, monthly, reps = load(data_dir)
    err, rel, mean_rain = error_tables(metrics, monthly)
    err, rel = err.reindex(base.index), rel.reindex(base.index)
    systems, core = zone_systems(base, alt, conf)

    say("=" * 72)
    say("ZONE ROBUSTNESS CHECK - do the forecasting conclusions depend on the zones?")
    say("=" * 72)
    say("District-level forecast errors are reused (they do not depend on the zones); only the")
    say("grouping of districts changes.\n")
    changed = base.index[(alt[ALT_COLUMN].reindex(base.index) != base["Zone"])]
    say(f"Baseline versus temperature + altitude zones: {len(changed)} districts differ:")
    for d in changed:
        say(f"  {d:<10} {base.loc[d, 'Zone']}  ->  {alt.loc[d, ALT_COLUMN]}")
    excluded = base.index[~core]
    say(f"Stable core: {int(core.sum())} districts. Left out ({len(excluded)}): {', '.join(excluded)}")
    say("")

    # Checks 1 + 2: zone definitions, main yardstick
    tests, rows, trows = {}, [], []
    for var, (vname, ylab) in VARIABLES.items():
        say("=" * 72)
        say(f"CHECKS 1-2  {vname.upper()}: SARIMA error by zone ({ylab})")
        say("=" * 72)
        orders = {}
        for sname, zones in systems.items():
            summary, h, p, eps, df = zone_stats(err[var], zones)
            tests[(sname, var)] = (summary, h, p, eps, df)
            orders[sname] = ordering(summary)
            say(f"\n{sname}")
            say(summary.round(2).to_string())
            say(f"  Kruskal-Wallis: {pfmt(p)}, effect size {eps:.2f}")
            say(f"  Easiest to hardest: {orders[sname]}")
            for z, r in summary.iterrows():
                rows.append({"Yardstick": "main", "Zone_definition": sname, "Variable": vname, "Zone": z,
                             "Districts": int(r["Districts"]), "Median_error": round(r["Median_error"], 3),
                             "Q1": round(r["Q1"], 3), "Q3": round(r["Q3"], 3)})
            trows.append({"Yardstick": "main", "Zone_definition": sname, "Variable": vname,
                          "Kruskal_H": round(h, 2), "p_value": round(p, 5),
                          "Effect_size_eps2": round(eps, 3), "Easiest_to_hardest": orders[sname]})
        same = len(set(orders.values())) == 1
        say(f"\n  {vname}: all zone definitions give the same ordering: {'YES' if same else 'NO'}; "
            f"significant (p < 0.05) in "
            f"{sum(1 for s_ in systems if tests[(s_, var)][2] < 0.05)} of {len(systems)}.\n")

    # Check 5: relative yardstick (done here so the report reads in order)
    say("=" * 72)
    say("CHECK 5  A DIFFERENT YARDSTICK: error relative to the variability of the months")
    say("=" * 72)
    say("Error = RMSE divided by the standard deviation of the district's monthly values in")
    say("2015-2019. 0 = nearly perfect, 1 = no better than guessing the average month.")
    rho_m, p_m = spear(err["Precip_total"], mean_rain.reindex(err.index))
    say(f"Why: the rainfall error used so far is a % of average rainfall, so it is inflated in dry")
    say(f"districts (Spearman correlation with mean monthly rainfall: {rho_m:+.2f}, {pfmt(p_m)}).")
    tests_rel = {}
    for var, (vname, _) in VARIABLES.items():
        say(f"\n{vname.upper()}")
        orders = {}
        for sname, zones in systems.items():
            summary, h, p, eps, df = zone_stats(rel[var], zones)
            tests_rel[(sname, var)] = (summary, h, p, eps, df)
            orders[sname] = ordering(summary)
            med = ", ".join(f"{z.split(' ')[0]} {summary.loc[z, 'Median_error']:.3f}" for z in ZONE_NAMES)
            say(f"  {sname:<32} medians: {med}   {pfmt(p)}   {orders[sname]}")
            for z, r in summary.iterrows():
                rows.append({"Yardstick": "relative to variability", "Zone_definition": sname,
                             "Variable": vname, "Zone": z, "Districts": int(r["Districts"]),
                             "Median_error": round(r["Median_error"], 3), "Q1": round(r["Q1"], 3),
                             "Q3": round(r["Q3"], 3)})
            trows.append({"Yardstick": "relative to variability", "Zone_definition": sname,
                          "Variable": vname, "Kruskal_H": round(h, 2), "p_value": round(p, 5),
                          "Effect_size_eps2": round(eps, 3), "Easiest_to_hardest": orders[sname]})
        rl, pl = spear(rel[var], base["Longitude"])
        say(f"  Correlation of this relative error with longitude: {rl:+.2f} ({pfmt(pl)})")

    # Representatives
    say("\n" + "=" * 72)
    say("THE THREE REPRESENTATIVE DISTRICTS UNDER EACH ZONE DEFINITION (main yardstick)")
    say("=" * 72)
    say("Percentile = share of districts in the same zone with a SMALLER error (50% = typical).")
    reps_rows = []
    for d in reps:
        for var, (vname, _) in VARIABLES.items():
            line = f"  {d:<12} {vname:<12}"
            for sname in ["Baseline zones (Step 4)", "Temperature + altitude zones"]:
                zs = systems[sname]
                z = zs[d]
                same = zs[zs == z].index
                e = err.loc[same, var]
                pct = 100 * (e < err.loc[d, var]).sum() / max(len(e) - 1, 1)
                short = "baseline" if sname.startswith("Base") else "temp+alt"
                line += f" | {short}: {z} (percentile {pct:.0f}%)"
                reps_rows.append({"District": d, "Variable": vname, "Zone_definition": sname, "Zone": z,
                                  "Error": round(err.loc[d, var], 3), "Zone_median": round(e.median(), 3),
                                  "Percentile_in_zone": round(pct, 0)})
            say(line)
    alt_zones = {alt.loc[d, ALT_COLUMN] for d in reps}
    say(f"\n  Under the temperature + altitude zones the representatives fall in {len(alt_zones)} different zone(s).")

    # Break cost by zone
    say("\n" + "=" * 72)
    say("COST OF THE 2004 BREAK BY ZONE (training on 2004-2014 only vs the full history)")
    say("=" * 72)
    say("Median change in rainfall error when the training window starts in 2004; negative = better.")
    brows = []
    rain = metrics[metrics["Variable"] == "Precip_total"]
    for model in ["Climatology", "SARIMA"]:
        a_ = rain[(rain["Model"] == model) & (rain["Window"] == FULL)].set_index("District")["Error"]
        b_ = rain[(rain["Model"] == model) & (rain["Window"] == POST)].set_index("District")["Error"]
        chg = (100 * (b_ / a_ - 1)).reindex(base.index)
        for sname, zones in systems.items():
            for z in ZONE_NAMES:
                idx = zones[zones == z].index
                v = chg.reindex(idx).dropna()
                brows.append({"Model": model, "Zone_definition": sname, "Zone": z, "Districts": len(v),
                              "Median_change_pct": round(v.median(), 1),
                              "Post2004_better_in": int((v < 0).sum())})
    brk = pd.DataFrame(brows)
    for model in ["Climatology", "SARIMA"]:
        t = brk[brk["Model"] == model].pivot(index="Zone", columns="Zone_definition",
                                              values="Median_change_pct").reindex(ZONE_NAMES)
        say(f"\n{model}:")
        say(t.to_string())

    # Check 3: no zones
    say("\n" + "=" * 72)
    say("CHECK 3  NO ZONES: correlation of error with altitude proxy, latitude and longitude")
    say("=" * 72)
    say("Spearman rank correlation across the 62 districts (main yardstick). Pressure falls as")
    say("altitude rises, so a NEGATIVE correlation with pressure means error grows with altitude.")
    crow = []
    for var, (vname, _) in VARIABLES.items():
        for pred in ["Pressure_mean", "Temp_mean", "Latitude", "Longitude"]:
            rho, p = spear(base[pred].reindex(err.index), err[var])
            crow.append({"Variable": vname, "Predictor": pred, "Spearman_rho": round(rho, 3),
                         "p_value": round(p, 5)})
    corr = pd.DataFrame(crow)
    say(corr.to_string(index=False))

    # Check 4: zones or geography
    say("\n" + "=" * 72)
    say("CHECK 4  ZONES OR GEOGRAPHY? Fit error ~ longitude + zone")
    say("=" * 72)
    say("R2 = share of the variation in error explained. 'p zone adds' tests whether the zones")
    say("explain anything beyond longitude; 'p longitude adds' tests the reverse.")
    grows = []
    for yard, table in [("main", err), ("relative to variability", rel)]:
        for var, (vname, _) in VARIABLES.items():
            for sname, zones in systems.items():
                g = geography_control(table[var], zones, base["Longitude"])
                grows.append({"Yardstick": yard, "Variable": vname, "Zone_definition": sname,
                              **{k: round(float(v), 4) for k, v in g.items()}})
    geo = pd.DataFrame(grows)
    say(geo.rename(columns={"R2_zone_only": "R2 zone", "R2_longitude_only": "R2 lon", "R2_both": "R2 both",
                            "p_zone_adds_to_longitude": "p zone adds",
                            "p_longitude_adds_to_zone": "p lon adds"}).drop(
        columns=["Zone_definition"]).assign(Zones=geo["Zone_definition"].str.split(" ").str[0]).to_string(index=False))

    # Verdict
    say("\n" + "=" * 72)
    say("VERDICT")
    say("=" * 72)
    for var, (vname, _) in VARIABLES.items():
        sig = sum(1 for s_ in systems if tests[(s_, var)][2] < 0.05)
        ords = {ordering(tests[(s_, var)][0]) for s_ in systems}
        hard = sorted({ordering(tests[(s_, var)][0]).split(" < ")[-1] for s_ in systems})
        easy = sorted({ordering(tests[(s_, var)][0]).split(" < ")[0] for s_ in systems})
        sig2 = sum(1 for s_ in systems if tests_rel[(s_, var)][2] < 0.05)
        hard2 = sorted({ordering(tests_rel[(s_, var)][0]).split(" < ")[-1] for s_ in systems})
        ra, pa = spear(base["Pressure_mean"].reindex(err.index), err[var])
        rl, pl = spear(base["Longitude"], err[var])
        gb = geo[(geo["Variable"] == vname) & (geo["Yardstick"] == "main") &
                 (geo["Zone_definition"] == "Baseline zones (Step 4)")].iloc[0]
        say(f"  {vname}")
        say(f"    main yardstick: zones differ in {sig} of {len(systems)} definitions; easiest {', '.join(easy)}; "
            f"hardest {', '.join(hard)}; {'same ordering throughout' if len(ords) == 1 else 'ordering changes'}.")
        say(f"    relative yardstick: zones differ in {sig2} of {len(systems)} definitions; hardest "
            f"{', '.join(hard2)}.")
        say(f"    no zones: error vs altitude rho = {-ra:+.2f} ({pfmt(pa)}); error vs longitude rho = "
            f"{rl:+.2f} ({pfmt(pl)}).")
        say(f"    zones or geography: longitude alone R2 = {gb['R2_longitude_only']:.2f}, zones alone "
            f"{gb['R2_zone_only']:.2f}; zones still add beyond longitude: {pfmt(gb['p_zone_adds_to_longitude'])}.")

    say("\nSaving figures and tables:")
    plot_definitions(err, systems, reps, out_dir, tests,
                     {v: VARIABLES[v][1] + ", lower is better" for v in VARIABLES},
                     "Does 'high mountains are hardest to forecast' depend on how the zones are drawn?",
                     "Same district errors (SARIMA, full history, 2015-2019), grouped three ways. "
                     "Stars = the three districts forecast in Step 5.",
                     "29_zone_definitions.png")
    plot_geography(err, base, reps, corr, out_dir)
    plot_definitions(rel, systems, reps, out_dir, tests_rel,
                     {v: "RMSE / SD of the monthly values, lower is better" for v in VARIABLES},
                     "A fairer yardstick: error relative to how variable each district's months are",
                     "RMSE divided by the standard deviation of the 2015-2019 monthly values. "
                     "Removes the inflation of percentage errors in dry districts.",
                     "31_relative_yardstick.png", show_reps=False)
    pd.DataFrame(rows).to_csv(out_dir / "robustness_zone_summary.csv", index=False)
    pd.DataFrame(trows).to_csv(out_dir / "robustness_tests.csv", index=False)
    corr.to_csv(out_dir / "zone_free_correlations.csv", index=False)
    geo.to_csv(out_dir / "geography_control.csv", index=False)
    pd.DataFrame(reps_rows).to_csv(out_dir / "representatives_check.csv", index=False)
    brk.to_csv(out_dir / "break_cost_by_zone.csv", index=False)
    (out_dir / "zone_robustness_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
