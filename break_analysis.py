
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Settings

BREAK_YEAR = 2004          
N_PERM = 1000              
SEED = 0
NEAR = (2003, 2004, 2005) 
MIN_EXPLAINED = 0.40   

VARS = { 
    "Precip": ("Rainfall", "mm/day"),
    "Humidity_2m": ("Specific humidity", "g/kg"),
    "RH_2m": ("Relative humidity", "%"),
    "TempRange_2m": ("Daily temperature range", "C"),
    "WindSpeed_10m": ("Wind speed at 10 m", "m/s"),
    "WindSpeed_50m": ("Wind speed at 50 m", "m/s"),
    "WetBulbTemp_2m": ("Wet-bulb temperature", "C"),
    "MaxTemp_2m": ("Daily maximum temperature", "C"),
    "MinTemp_2m": ("Daily minimum temperature", "C"),
    "EarthSkinTemp": ("Earth skin temperature", "C"),
    "Temp_2m": ("Mean temperature", "C"),
    "Pressure": ("Surface pressure", "kPa"),
}
PANELS = ["Precip", "Humidity_2m", "RH_2m", "TempRange_2m",
          "WindSpeed_10m", "EarthSkinTemp", "Temp_2m", "Pressure"]
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


def save_fig(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"  saved {name}")


# Change-point detection

def best_break(y, min_seg=5):
    """Find the split point that best divides y into two flat (constant-mean) pieces.
    Returns (k, explained): the second piece starts at index k; 'explained' is the share of
    the series' variation accounted for by the step (0 to 1)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    c1, c2 = np.cumsum(y), np.cumsum(y ** 2)
    tot, tot2 = c1[-1], c2[-1]
    best_k, best_sse = None, np.inf
    for k in range(min_seg, n - min_seg + 1):
        s1, q1 = c1[k - 1], c2[k - 1]
        s2, q2 = tot - s1, tot2 - q1
        sse = (q1 - s1 ** 2 / k) + (q2 - s2 ** 2 / (n - k))
        if sse < best_sse:
            best_k, best_sse = k, sse
    sse0 = tot2 - tot ** 2 / n
    return best_k, 1 - best_sse / sse0


def perm_p(y, explained, rng, min_seg=5):
    """How often does a shuffled (structure-free) series show a step this big by chance?"""
    count = 0
    for _ in range(N_PERM):
        if best_break(rng.permutation(y), min_seg)[1] >= explained:
            count += 1
    return (count + 1) / (N_PERM + 1)


# Analysis

def analyse(d):
    rng = np.random.default_rng(SEED)
    cols = list(VARS)
    annual = d.groupby("Year")[cols].mean()
    years = annual.index.values

    monthly = d.groupby(["Year", "Month"])[cols].mean()
    monthly.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in monthly.index])
    anomaly = monthly - monthly.groupby(monthly.index.month).transform("mean")

    rows = []
    for v in cols:
        y = annual[v].values
        k, expl = best_break(y)
        pre = annual.loc[:BREAK_YEAR - 1, v]
        post = annual.loc[BREAK_YEAR:, v]
        row = {
            "Variable": v, "Label": VARS[v][0], "Unit": VARS[v][1],
            "Best_break_year": int(years[k]),
            "Variation_explained": round(expl, 3),
            "Permutation_p": round(perm_p(y, expl, rng), 3),
            f"Average_before_{BREAK_YEAR}": round(pre.mean(), 3),
            f"Average_from_{BREAK_YEAR}": round(post.mean(), 3),
            f"Shift_at_{BREAK_YEAR}_in_SD": round((post.mean() - pre.mean()) / annual[v].std(ddof=1), 2),
            f"Change_at_{BREAK_YEAR}_in_units": round(post.mean() - pre.mean(), 2),
            f"Change_at_{BREAK_YEAR}_pct": round(100 * (post.mean() / pre.mean() - 1), 1),
        }

        if VARS[v][1] == "C":
            row["Change"] = f"{post.mean() - pre.mean():+.1f} C"
        else:
            row["Change"] = f"{100 * (post.mean() / pre.mean() - 1):+.0f}%"
        near = row["Best_break_year"] in NEAR and expl >= MIN_EXPLAINED
        row["Breaks_near_2004"] = near
        if near:
            km, _ = best_break(anomaly[v].values, min_seg=24)
            row["Break_month"] = f"{anomaly.index[km]:%Y-%m}"
        else:
            row["Break_month"] = ""
        rows.append(row)
    summary = pd.DataFrame(rows)


    ar = d.groupby(["District", "Year"])["Precip"].sum().unstack()
    yrs = ar.columns.values
    dist_break = pd.Series({dist: int(yrs[best_break(ar.loc[dist].values)[0]]) for dist in ar.index})
    return annual, anomaly, summary, dist_break


def physics_check(d):
    out = {}
    for lab, x in [("1981-2003", d[d["Year"] <= BREAK_YEAR - 1]),
                   (f"{BREAK_YEAR + 1}-2019", d[d["Year"] >= BREAK_YEAR + 1])]:
        mons = x[x["Month"].isin([6, 7, 8, 9])]
        out[lab] = {
            "corr_rain_RH": x["Precip"].corr(x["RH_2m"]),
            "monsoon_rain": mons["Precip"].mean(),
            "monsoon_RH": mons["RH_2m"].mean(),
            "monsoon_humidity": mons["Humidity_2m"].mean(),
        }
    return pd.DataFrame(out)


# Figures

def plot_variables(annual, summary, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.6))
    s = summary.set_index("Variable")
    for ax, v in zip(axes.ravel(), PANELS):
        label, unit = VARS[v]
        r = s.loc[v]
        jumped = bool(r["Breaks_near_2004"])
        color = "#c0392b" if jumped else "#566573"
        y = annual[v]
        pre, post = y.loc[:BREAK_YEAR - 1], y.loc[BREAK_YEAR:]
        ax.plot(y.index, y.values, color=color, marker="o", ms=3.5, lw=1.5, zorder=3)
        ax.hlines(pre.mean(), pre.index.min(), pre.index.max(), color="black", ls="--", lw=1.8, zorder=4)
        ax.hlines(post.mean(), post.index.min(), post.index.max(), color="black", ls="--", lw=1.8, zorder=4)
        ax.axvline(BREAK_YEAR - 0.5, color="#c0392b", ls=":", lw=1.5, zorder=2)
        ax.set_title(f"{label} ({unit})", loc="left", pad=22)
        if jumped:
            verdict = "jumps near 2004"
        elif int(r["Best_break_year"]) in NEAR:
            verdict = "weak step near 2004"
        else:
            verdict = "no jump at 2004"
        ax.text(0, 1.02, f"{verdict}: {r[f'Shift_at_{BREAK_YEAR}_in_SD']:+.1f} SD, "
                f"best break {int(r['Best_break_year'])}", transform=ax.transAxes, fontsize=9.5,
                color=color, va="bottom", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Year")
    fig.suptitle("Did only rainfall jump in 2004? No: moisture-related variables jumped together, "
                 "temperature and pressure did not", fontsize=15, fontweight="bold",
                 x=0.01, ha="left", y=1.005)
    fig.text(0.01, 0.955, "National average of all districts per year. Dashed lines = average "
             f"before / from {BREAK_YEAR}. SD = standard deviations of the yearly values "
             "(about 1.5 or more is a big step).", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    save_fig(fig, out_dir, "14_variables_around_2004.png")


def plot_timing(anomaly, summary, dist_break, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})

    # A: monthly national rainfall anomaly around the break
    an = anomaly["Precip"]
    win = an["2000-01-01":"2008-12-01"]
    smooth = an.rolling(12, center=True).mean()["2000-01-01":"2008-12-01"]
    a.axhline(0, color="grey", lw=1)
    a.bar(win.index, win.values, width=25, color="#aed6f1", zorder=2, label="Monthly (vs. usual for that month)")
    a.plot(smooth.index, smooth.values, color="#1b4f72", lw=2.8, zorder=4, label="12-month average")
    months = pd.to_datetime(summary.loc[summary["Break_month"] != "", "Break_month"] + "-01")
    if len(months):
        a.axvspan(months.min(), months.max() + pd.offsets.MonthEnd(0), color="#f5b7b1", alpha=0.6,
                  zorder=1, label=f"Break month of {len(months)} variables\n"
                  f"({months.min():%b %Y} to {months.max():%b %Y})")
    a.set_ylabel("National rainfall anomaly (mm/day)")
    a.set_title("The step happens within a few months, not gradually", loc="left")
    a.legend(loc="upper left", frameon=True, fontsize=9)
    a.grid(alpha=0.3)

    # B: per-district break years
    counts = dist_break.value_counts().sort_index()
    yrs = np.arange(counts.index.min() - 1, counts.index.max() + 2)
    vals = [counts.get(y, 0) for y in yrs]
    colors = ["#c0392b" if y in NEAR else "#aab7b8" for y in yrs]
    b.bar(yrs, vals, color=colors, zorder=3)
    near = int(dist_break.isin(NEAR).sum())
    exact = int((dist_break == BREAK_YEAR).sum())
    b.text(0.03, 0.95, f"{exact} of {len(dist_break)} districts: {BREAK_YEAR}\n"
           f"{near} of {len(dist_break)}: within 2003-2005", transform=b.transAxes, va="top",
           fontsize=11, fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    b.set_ylim(0, max(vals) * 1.4)
    b.set_xlabel("Best break year of the district's annual rainfall")
    b.set_ylabel("Number of districts")
    b.set_title("Almost every district breaks in the same year", loc="left")
    b.grid(axis="y", alpha=0.3, zorder=0)
    fig.suptitle("When and where did the rainfall jump happen?", fontsize=15, fontweight="bold",
                 x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    save_fig(fig, out_dir, "15_break_timing.png")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python break_analysis.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "break_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "climate_daily_clean.csv"
    if not path.exists():
        sys.exit(f"Cannot find {path}. Run prepare_data.py first.")

    say("=" * 68)
    say("BREAK ANALYSIS - was the 2004 rainfall jump real?")
    say("=" * 68)
    d = pd.read_csv(path, usecols=["District", "Year", "Month"] + list(VARS))
    say(f"Loaded {len(d):,} daily records, {d['District'].nunique()} districts.")

    say("\nBackground (documentation checked online):")
    say("  - The Kaggle page says the data come from NASA's POWER project, extracted for 93")
    say("    weather-station locations in 62 districts. That is MODELLED data, not station")
    say("    measurements, so changes in the underlying product can create artificial jumps.")
    say("  - An Open Data Nepal listing with the same date range and the same total row count")
    say("    (2 x 441,564 = 883,128) carries a note that precipitation is reported differently")
    say("    before and after January 2018. That is not 2004, so it does not explain this jump.")
    say("  - No documentation found for a change in 2004. The cause is UNVERIFIED.")

    annual, anomaly, summary, dist_break = analyse(d)

    say(f"\nCHECK 1 - Which variables show a step? (national yearly averages, 1981-2019)")
    show = summary[["Label", "Best_break_year", "Variation_explained", "Permutation_p",
                    f"Shift_at_{BREAK_YEAR}_in_SD", "Change", "Break_month"]]
    say(f"(Shift and Change compare {BREAK_YEAR}-2019 with 1981-{BREAK_YEAR - 1})")
    say(show.rename(columns={"Best_break_year": "Break_year", "Variation_explained": "Explained",
                             f"Shift_at_{BREAK_YEAR}_in_SD": "Shift_SD"}).to_string(index=False))
    jumped = summary.loc[summary["Breaks_near_2004"], "Label"].tolist()
    stable = summary.loc[~summary["Breaks_near_2004"], "Label"].tolist()
    say(f"\nBreak near 2004 (best year 2003-2005 and explains >= {MIN_EXPLAINED:.0%}): "
        f"{', '.join(jumped)}")
    say(f"No clear break near 2004: {', '.join(stable)}")

    say("\nCHECK 2 - Timing (monthly, seasonal cycle removed):")
    bm = summary.loc[summary["Break_month"] != "", ["Label", "Break_month"]]
    say(bm.to_string(index=False))

    near = int(dist_break.isin(NEAR).sum())
    say(f"\nCHECK 3 - Spread across districts: best break year of annual rainfall is "
        f"{BREAK_YEAR} in {int((dist_break == BREAK_YEAR).sum())} of {len(dist_break)} districts, "
        f"and within 2003-2005 in {near}.")

    say("\nCHECK 4 - Is the physics consistent? (1981-2003 vs 2005-2019)")
    say(physics_check(d).round(2).to_string())
    say("  The rain-humidity correlation is similar in both periods, but the LEVELS of rainfall")
    say("  and humidity both stepped up. Real climate shifts and product changes can both do this,")
    say("  so this check cannot settle the question.")

    say("\nWHAT THE EVIDENCE SUPPORTS")
    say("  - The jump is abrupt (within months), nearly simultaneous in almost every district,")
    say("    and appears in several moisture-related variables together, while mean temperature")
    say("    and pressure show no matching step.")
    say("  - That pattern is typical of a change in how a modelled dataset was produced, and it")
    say("    is difficult to attribute to natural climate variability. It is NOT proof: independent station")
    say("    records (e.g. from Nepal's Department of Hydrology and Meteorology) would be needed.")
    say("  - Suggested wording: 'The data contain an abrupt, nationwide step change in rainfall")
    say("    and humidity in early 2004, most likely a data-product artifact.'")
    say("\nNote: p-values come from a permutation test that treats years as independent, so they")
    say("are approximate. The size of the step in SD is the more useful number.")

    say("\nSaving figures and tables:")
    plot_variables(annual, summary, out_dir)
    plot_timing(anomaly, summary, dist_break, out_dir)
    summary.to_csv(out_dir / "break_summary.csv", index=False)
    (out_dir / "break_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
