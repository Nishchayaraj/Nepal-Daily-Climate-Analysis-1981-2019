import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DISTRICTS = {
    "Banke": ("#D55E00", "Banke (Nepalgunj) - Terai"),
    "Kathmandu": ("#009E73", "Kathmandu - Hills"),
    "Humla": ("#0072B2", "Humla - High Himalaya"),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_SPANS = [  # (start, end, label, colour)
    (0.5, 2.5, "Winter", "#dbe9f6"),
    (2.5, 5.5, "Pre-monsoon", "#fdebd0"),
    (5.5, 9.5, "Monsoon", "#d5f0e3"),
    (9.5, 11.5, "Post-monsoon", "#fcf3cf"),
    (11.5, 12.5, "Winter", "#dbe9f6"),
]
SOURCE = "Source: Nepal Daily Climate Dataset (Kaggle), 1981-2019, 62 districts"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


def save(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}")


def shade_seasons(ax, label=True):
    for x0, x1, lab, col in SEASON_SPANS:
        ax.axvspan(x0, x1, color=col, zorder=0, lw=0)
        if label:
            ax.text((x0 + x1) / 2, 0.985, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#555555")


def load(data_dir):
    monthly = pd.read_csv(data_dir / "climate_monthly.csv", parse_dates=["Date"])
    feats = pd.read_csv(data_dir / "district_features.csv")
    daily = pd.read_csv(data_dir / "climate_daily_clean.csv",
                        usecols=["District", "Year", "Month", "Precip", "Temp_2m"])
    return monthly, feats, daily


# 1. Seasonal cycle (national average)

def plot_seasonal_cycle(monthly, out_dir):
    clim = monthly.groupby(["District", "Month"])[["Temp_mean", "Precip_total"]].mean()
    nat = clim.groupby("Month").mean()

    fig, ax = plt.subplots(figsize=(10, 5.6))
    shade_seasons(ax)
    ax.bar(nat.index, nat["Precip_total"], width=0.7, color="#5dade2", zorder=3,
           label="Rainfall (left axis)")
    ax.set_ylabel("Average monthly rainfall (mm)")
    ax.set_ylim(0, nat["Precip_total"].max() * 1.3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS)
    ax.set_xlim(0.5, 12.5)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(nat.index, nat["Temp_mean"], color="#c0392b", marker="o", lw=2.6,
             zorder=5, label="Temperature (right axis)")
    ax2.set_ylabel("Average temperature (C)")
    ax2.set_ylim(0, nat["Temp_mean"].max() * 1.3)

    peak_rain = nat["Precip_total"].idxmax()
    peak_temp = nat["Temp_mean"].idxmax()
    cold = nat["Temp_mean"].idxmin()
    ax.annotate(f"Wettest: {MONTHS[peak_rain - 1]}\n{nat['Precip_total'].max():.0f} mm",
                (peak_rain, nat["Precip_total"].max()), xytext=(26, 8),
                textcoords="offset points", ha="center", fontsize=9.5, zorder=6)
    ax2.annotate(f"Hottest: {MONTHS[peak_temp - 1]}\n{nat['Temp_mean'].max():.1f} C",
                 (peak_temp, nat["Temp_mean"].max()), xytext=(-30, 10),
                 textcoords="offset points", ha="center", fontsize=9.5,
                 color="#c0392b", zorder=6)
    ax2.annotate(f"Coldest: {MONTHS[cold - 1]}\n{nat['Temp_mean'].min():.1f} C",
                 (cold, nat["Temp_mean"].min()), xytext=(-12, -36),
                 textcoords="offset points", ha="left", fontsize=9.5,
                 color="#c0392b", zorder=6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, 0.88),
              frameon=True, fontsize=9)
    ax.set_title("Nepal's climate runs on the monsoon: rain peaks in July-August,\n"
                 "while temperature peaks earlier, in June", loc="left")
    save(fig, out_dir, "1_seasonal_cycle.png")

    share = nat.loc[6:9, "Precip_total"].sum() / nat["Precip_total"].sum()
    print(f"     Wettest month: {MONTHS[peak_rain-1]} ({nat['Precip_total'].max():.1f} mm); "
          f"hottest: {MONTHS[peak_temp-1]} ({nat['Temp_mean'].max():.1f} C); "
          f"coldest: {MONTHS[cold-1]} ({nat['Temp_mean'].min():.1f} C); "
          f"Jun-Sep rain share: {share:.0%}")



# 2. Three contrasting districts

def plot_three_districts(monthly, out_dir):
    cols = ["Temp_mean", "TempMax_mean", "TempMin_mean", "Precip_total"]
    clim = monthly.groupby(["District", "Month"])[cols].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    axT, axR = axes
    shade_seasons(axT, label=True)
    shade_seasons(axR, label=True)

    w = 0.27
    for i, (d, (col, lab)) in enumerate(DISTRICTS.items()):
        s = clim[clim["District"] == d].sort_values("Month")
        axT.fill_between(s["Month"], s["TempMin_mean"], s["TempMax_mean"],
                         color=col, alpha=0.14, zorder=2, lw=0)
        axT.plot(s["Month"], s["Temp_mean"], color=col, lw=2.6, marker="o",
                 label=lab, zorder=4)
        axR.bar(s["Month"] + (i - 1) * w, s["Precip_total"], width=w, color=col,
                label=lab, zorder=3)

    axT.set_title("Temperature: three very different climates", loc="left")
    axT.set_ylabel("Average temperature (C)\nline = daily mean, band = daily min to max")
    axR.set_title("Rainfall: same monsoon timing, very different amounts", loc="left")
    axR.set_ylabel("Average monthly rainfall (mm)")
    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTHS)
        ax.set_xlim(0.5, 12.5)
    axT.set_ylim(axT.get_ylim()[0], axT.get_ylim()[1] + 4)
    axR.set_ylim(0, axR.get_ylim()[1] * 1.15)
    axT.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
    axT.legend(loc="lower center", frameon=True, fontsize=9, ncol=1,
               bbox_to_anchor=(0.5, 0.06))
    axR.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=True, fontsize=9)
    fig.suptitle("Why one national model won't work: Terai vs Hills vs Himalaya",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "2_three_districts.png")

    ann = monthly.groupby(["District", "Year"]).agg(
        T=("Temp_mean", "mean"), R=("Precip_total", "sum")).groupby("District").mean()
    print("     Annual mean temp (C) / annual rain (mm):")
    for d in DISTRICTS:
        print(f"       {d}: {ann.loc[d, 'T']:.1f} C, {ann.loc[d, 'R']:.0f} mm")



# 3. Rainfall distribution

def plot_rain_distribution(daily, out_dir):
    p = daily["Precip"]
    bins = [-np.inf, 0, 1, 10, 50, np.inf]
    names = ["Dry\n(0 mm)", "Drizzle\n(0-1 mm)", "Light-moderate\n(1-10 mm)",
             "Heavy\n(10-50 mm)", "Extreme\n(>50 mm)"]
    cats = pd.cut(p, bins=bins, labels=names)
    share = cats.value_counts(normalize=True).reindex(names)

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.3), gridspec_kw={"width_ratios": [1, 1.15]})
    colors = ["#d5d8dc", "#aed6f1", "#5dade2", "#2874a6", "#1b2631"]
    bars = a.bar(range(5), share.values * 100, color=colors, zorder=3)
    for r, v in zip(bars, share.values):
        a.text(r.get_x() + r.get_width() / 2, v * 100 + 1, f"{v:.1%}", ha="center",
               fontsize=10, fontweight="bold")
    a.set_xticks(range(5))
    a.set_xticklabels(names, fontsize=9.5)
    a.set_ylabel("Share of all district-days (%)")
    a.set_ylim(0, share.max() * 100 * 1.15)
    a.set_title("More than 4 in 10 days have no rain at all", loc="left")
    a.grid(axis="y", alpha=0.3, zorder=0)

    rainy = p[p > 1]
    p95 = rainy.quantile(0.95)
    b.hist(rainy.clip(upper=100), bins=np.arange(1, 102, 2), color="#2e86c1", zorder=3)
    b.set_yscale("log")
    b.axvline(p95, color="#c0392b", ls="--", lw=1.8, zorder=4)
    b.text(p95 + 1.5, b.get_ylim()[1] * 0.45, f"95th percentile\nof rainy days:\n{p95:.0f} mm",
           color="#c0392b", fontsize=9.5, va="top")
    b.set_xlabel("Daily rainfall on rainy days (mm; values above 100 mm grouped at 100)")
    b.set_ylabel("Number of district-days (log scale)")
    b.set_title("Among rainy days: many light, very few extreme", loc="left")
    b.grid(axis="y", alpha=0.3, zorder=0)
    fig.suptitle("Rainfall is highly skewed, which makes it hard to forecast",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "3_rainfall_distribution.png")

    print(f"     Dry days: {(p == 0).mean():.1%}; rainy (>1 mm): {(p > 1).mean():.1%}; "
          f"mean {p.mean():.2f} mm, median {p.median():.2f} mm, max {p.max():.1f} mm; "
          f"days >50 mm: {(p > 50).mean():.2%} ({int((p > 50).sum()):,} district-days)")


# 4. Correlation heatmap

def plot_heatmap(feats, out_dir):
    pretty = {
        "Latitude": "Latitude", "Longitude": "Longitude",
        "Temp_mean": "Mean temp", "TempRange_mean": "Daily temp range",
        "Pressure_mean": "Pressure", "Humidity_mean": "Specific humidity",
        "RH_mean": "Relative humidity", "Wind10_mean": "Wind (10 m)",
        "RainyDay_share": "Share of rainy days", "Annual_rain_mm": "Annual rainfall",
        "Monsoon_rain_share": "Monsoon rain share",
        "Seasonal_temp_swing": "Seasonal temp swing", "Winter_MinTemp": "Winter min temp",
    }
    corr = feats.drop(columns="District").corr().rename(index=pretty, columns=pretty)
    n = len(corr)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    data = np.ma.array(corr.values, mask=mask)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.65 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Temperature, pressure and winter cold move together\n"
                 "(correlations between the 62 districts' climate features)", loc="left")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Correlation")
    fig.tight_layout()
    save(fig, out_dir, "4_correlation_heatmap.png")

    c = corr.where(~mask & ~np.eye(n, dtype=bool)).stack()
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(6)
    print("     Strongest correlations:")
    for (x, y), v in top.items():
        print(f"       {x} vs {y}: {v:+.2f}")
    print(f"     Annual rainfall vs Mean temp: {corr.loc['Annual rainfall', 'Mean temp']:+.2f}; "
          f"Annual rainfall vs Longitude: {corr.loc['Annual rainfall', 'Longitude']:+.2f}; "
          f"Monsoon share vs Mean temp: {corr.loc['Monsoon rain share', 'Mean temp']:+.2f}")


# 5. Maps

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


def plot_maps(feats, out_dir, outline):
    panels = [
        ("Temp_mean", "Mean temperature (C)", "RdYlBu_r", "Warm Terai, cold mountains"),
        ("Annual_rain_mm", "Mean annual rainfall (mm)", "YlGnBu", "Rainfall (relative patterns)"),
        ("Pressure_mean", "Mean surface pressure (kPa)", "viridis",
         "Pressure: low = high altitude"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for ax, (col, lab, cmap, title) in zip(axes, panels):
        draw_outline(ax, outline)
        sc = ax.scatter(feats["Longitude"], feats["Latitude"], c=feats[col], cmap=cmap,
                        s=170, edgecolor="#2c3e50", linewidth=0.7, zorder=3)
        for d, (colr, _) in DISTRICTS.items():
            r = feats[feats["District"] == d].iloc[0]
            ax.scatter(r["Longitude"], r["Latitude"], s=360, facecolor="none",
                       edgecolor="black", linewidth=1.8, zorder=4)
            ax.annotate(d, (r["Longitude"], r["Latitude"]), xytext=(0, 17),
                        textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(79.9, 88.4)
        ax.set_ylim(26.1, 30.6)
        ax.set_aspect(1 / np.cos(np.radians(28)))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, loc="left")
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.18, shrink=0.85)
        cb.set_label(lab)
    fig.suptitle("Geography shows up before any modelling: climate follows altitude, "
                 "from the Terai (south) to the Himalaya (north)",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "5_maps.png")

    r = feats.sort_values("Annual_rain_mm")
    print(f"     Driest: {r.iloc[0]['District']} ({r.iloc[0]['Annual_rain_mm']:.0f} mm); "
          f"wettest: {r.iloc[-1]['District']} ({r.iloc[-1]['Annual_rain_mm']:.0f} mm)")


# 6. Long-term view

def plot_long_term(daily, out_dir):
    ann_t = daily.groupby("Year")["Temp_2m"].mean()
    ann_r = daily.groupby(["District", "Year"])["Precip"].sum().groupby("Year").mean()

    slopes = {}
    for d, g in daily.groupby("District"):
        a = g.groupby("Year")["Temp_2m"].mean()
        res = stats.linregress(a.index, a.values)
        slopes[d] = (res.slope * 10, res.pvalue)
    sl = pd.DataFrame(slopes, index=["slope", "p"]).T
    sig = sl["p"] < 0.05

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # (a) temperature: straight-line trend
    ax = axes[0]
    res = stats.linregress(ann_t.index, ann_t.values)
    ax.plot(ann_t.index, ann_t.values, color="#c0392b", marker="o", ms=4, lw=1.6, zorder=3)
    ax.plot(ann_t.index, res.intercept + res.slope * ann_t.index, color="black",
            ls="--", lw=1.8, zorder=4)
    ax.text(0.03, 0.05, f"Trend: {res.slope * 10:+.3f} C per decade\np-value = {res.pvalue:.2f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean temperature (C)")
    ax.set_title("Temperature: no clear trend", loc="left")
    ax.grid(alpha=0.3)
    print(f"     Temperature: slope {res.slope * 10:+.3f} C per decade, p = {res.pvalue:.3f}")

    # (b) rainfall: a straight line is misleading, so show the step change
    ax = axes[1]
    brk = int(ann_r.diff().abs().idxmax())       # year of the largest year-to-year jump
    pre, post = ann_r.loc[: brk - 1], ann_r.loc[brk:]
    r_dist = daily.groupby(["District", "Year"])["Precip"].sum().unstack()
    ratios = r_dist.loc[:, brk:].mean(axis=1) / r_dist.loc[:, : brk - 1].mean(axis=1)
    ax.plot(ann_r.index, ann_r.values, color="#2e86c1", marker="o", ms=4, lw=1.6, zorder=3)
    ax.hlines(pre.mean(), pre.index.min(), pre.index.max(), color="black", ls="--", lw=2, zorder=4)
    ax.hlines(post.mean(), post.index.min(), post.index.max(), color="black", ls="--", lw=2, zorder=4)
    ax.axvline(brk - 0.5, color="#c0392b", ls=":", lw=2, zorder=2)
    ax.text(float(np.mean(pre.index)), pre.mean() - 245, f"1981-{brk - 1} average\n{pre.mean():.0f} mm",
            ha="center", va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    ax.text(float(np.mean(post.index)), post.mean() - 330, f"{brk}-2019 average\n{post.mean():.0f} mm "
            f"(+{post.mean() / pre.mean() - 1:.0%})", ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    ax.text(brk - 0.9, ann_r.max() * 1.02, f"{brk}", color="#c0392b", ha="right", fontsize=10,
            fontweight="bold")
    ax.text(0.03, 0.96, f"Rose in {int((ratios > 1).sum())} of {len(ratios)} districts\n"
            f"(median x{ratios.median():.2f}, range x{ratios.min():.2f}-x{ratios.max():.2f})",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual rainfall per district (mm)")
    ax.set_title(f"Rainfall: an abrupt jump in {brk}, not a gradual trend", loc="left")
    ax.set_ylim(0, ann_r.max() * 1.1)
    ax.grid(alpha=0.3)
    print(f"     Rainfall: largest year-to-year jump is {brk}; avg {pre.mean():.0f} mm before vs "
          f"{post.mean():.0f} mm after (+{post.mean() / pre.mean() - 1:.0%}); rose in "
          f"{int((ratios > 1).sum())}/{len(ratios)} districts (median x{ratios.median():.2f})")

    ax = axes[2]
    ax.hist(sl.loc[~sig, "slope"], bins=np.arange(-0.15, 0.21, 0.025), color="#aab7b8",
            label=f"Not significant ({(~sig).sum()} districts)", zorder=3)
    ax.hist(sl.loc[sig, "slope"], bins=np.arange(-0.15, 0.21, 0.025), color="#c0392b",
            label=f"Significant, p<0.05 ({sig.sum()} districts)", zorder=3)
    ax.axvline(0, color="black", lw=1.4, zorder=6)
    ax.set_xlabel("Temperature trend per district (C per decade)")
    ax.set_ylabel("Number of districts")
    ax.set_title("District trends cluster around zero", loc="left")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(f"1981-2019: no real warming trend, but rainfall jumps abruptly in {brk}",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "6_long_term_trends.png")

    print(f"     Districts with significant temp trend: {sig.sum()}/62; "
          f"slope range {sl['slope'].min():+.3f} to {sl['slope'].max():+.3f} C/decade; "
          f"warming (slope>0): {(sl['slope'] > 0).sum()}, cooling: {(sl['slope'] < 0).sum()}")
    print(f"     Hottest year {ann_t.idxmax()} ({ann_t.max():.2f} C), "
          f"coolest {ann_t.idxmin()} ({ann_t.min():.2f} C); "
          f"wettest {ann_r.idxmax()} ({ann_r.max():.0f} mm), driest {ann_r.idxmin()} ({ann_r.min():.0f} mm)")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python eda_plots.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "eda_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    outline = data_dir / "nepal_outline.geojson"

    print("Loading data ...")
    monthly, feats, daily = load(data_dir)

    print("1/6 Seasonal cycle");        plot_seasonal_cycle(monthly, out_dir)
    print("2/6 Three districts");       plot_three_districts(monthly, out_dir)
    print("3/6 Rainfall distribution"); plot_rain_distribution(daily, out_dir)
    print("4/6 Correlation heatmap");   plot_heatmap(feats, out_dir)
    print("5/6 Maps");                  plot_maps(feats, out_dir, outline)
    print("6/6 Long-term trends");      plot_long_term(daily, out_dir)
    print(f"\nDone. Figures are in {out_dir}")


if __name__ == "__main__":
    main()
