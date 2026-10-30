import itertools
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_DISTRICTS = ["Nawalparasi", "Lamjung", "Dolpa"]  
SPLIT_DATE = pd.Timestamp("2015-01-01")   
BREAK_YEAR = 2004                         
PLOT_START = pd.Timestamp("2010-01-01")    

VARIABLES = {
    "Temp_mean": ("Temperature", "Monthly mean temperature (C)"),
    "Precip_total": ("Rainfall", "Monthly rainfall (mm)"),
}


# SARIMA candidates (p, d, q) x (P, D, Q, 12);
# training data only, so the test period never influences the choice.
ORDERS = [(1, 0, 0), (1, 0, 1), (2, 0, 0), (0, 0, 1)]
SEASONAL_ORDERS = [(0, 1, 1), (1, 1, 1)]

M_SN, M_CL, M_SA = "Seasonal-naive", "Climatology", "SARIMA"
COLORS = {M_SN: "#E69F00", M_CL: "#7f8c8d", M_SA: "#c0392b"}
SOURCE = "Source: Nepal Daily Climate Dataset (Kaggle), 1981-2019"

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


# Choose districts

def pick_districts(data_dir, monthly):
    path = data_dir / "clustering" / "representative_districts.csv"
    zones = {}
    if path.exists():
        reps = pd.read_csv(path)
        districts = reps["District"].tolist()
        zones = dict(zip(reps["District"], reps["Zone"]))
        source = "representative districts from Step 4"
    else:
        districts = DEFAULT_DISTRICTS
        source = "default list (clustering output not found)"
    missing = [d for d in districts if d not in set(monthly["District"])]
    if missing:
        sys.exit(f"Districts not found in climate_monthly.csv: {missing}")
    return districts, zones, source


# Baselines

def seasonal_naive(train, index):
    """Repeat the last 12 training months, month by month."""
    last = train.iloc[-12:]
    lookup = {d.month: v for d, v in zip(last.index, last.values)}
    return pd.Series([lookup[d.month] for d in index], index=index)


def climatology(train, index):
    """Average of each calendar month over the training period."""
    avg = train.groupby(train.index.month).mean()
    return pd.Series(avg.reindex([d.month for d in index]).values, index=index)


# SARIMA

def fit_sarima(train):
    """Try each candidate, keep the one with the lowest AIC (training data only)."""
    best = None
    for order, sorder in itertools.product(ORDERS, SEASONAL_ORDERS):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(train, order=order, seasonal_order=sorder + (12,),
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, maxiter=200)
        except Exception:
            continue
        if best is None or res.aic < best[0].aic:
            best = (res, order, sorder)
    if best is None:
        raise RuntimeError("No SARIMA model could be fitted.")
    return best


def pct(x):
    """Format a percentage change without showing '-0%'."""
    v = int(round(x))
    return "0%" if v == 0 else f"{v:+d}%"


# Metrics

def score(actual, pred):
    err = np.asarray(pred, dtype=float) - np.asarray(actual, dtype=float)
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "Bias": float(np.mean(err))}   # forecast minus actual: negative = too low


# Run one series (one district, one variable)

def run_series(y, var):
    train = y[y.index < SPLIT_DATE]
    test = y[y.index >= SPLIT_DATE]
    lower = 0.0 if var == "Precip_total" else None

    f_sn = seasonal_naive(train, test.index)
    f_cl = climatology(train, test.index)

    res, order, sorder = fit_sarima(train)
    fc = res.get_forecast(steps=len(test))
    f_sa = pd.Series(fc.predicted_mean.values, index=test.index).clip(lower=lower)
    ci = fc.conf_int(alpha=0.05)
    ci_lo = pd.Series(ci.iloc[:, 0].values, index=test.index).clip(lower=lower)
    ci_hi = pd.Series(ci.iloc[:, 1].values, index=test.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_all = res.apply(y, refit=False)
    f_sa1 = pd.Series(res_all.get_prediction(start=len(train), dynamic=False)
                      .predicted_mean.values, index=test.index).clip(lower=lower)
    f_sn1 = y.shift(12)[test.index]          # same month last year, using actual data

    train_post = train[train.index.year >= BREAK_YEAR]
    f_cl_post = climatology(train_post, test.index)

    return {
        "y": y, "train": train, "test": test, "order": order, "sorder": sorder, "aic": res.aic,
        "long": {M_SN: f_sn, M_CL: f_cl, M_SA: f_sa},
        "one": {M_SN: f_sn1, M_SA: f_sa1},
        "ci_lo": ci_lo, "ci_hi": ci_hi, "clim_post": f_cl_post,
        "coverage": float(((test >= ci_lo) & (test <= ci_hi)).mean()),
    }


# Figures

def label_for(d, zones):
    return f"{d} ({zones[d]})" if d in zones else d


def plot_forecasts(store, districts, zones, var, fname, title, out_dir):
    name, ylabel = VARIABLES[var]
    fig, axes = plt.subplots(len(districts), 1, figsize=(13.5, 3.7 * len(districts)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, d) in enumerate(zip(axes, districts)):
        r = store[(d, var)]
        test = r["test"]
        show = r["y"][r["y"].index >= PLOT_START]
        ax.axvspan(test.index[0], test.index[-1] + pd.offsets.MonthEnd(0), color="#f7f1dc", zorder=0)
        ax.fill_between(test.index, r["ci_lo"], r["ci_hi"], color=COLORS[M_SA], alpha=0.12,
                        zorder=2, lw=0, label="SARIMA 95% interval")
        ax.plot(test.index, r["long"][M_SN], color=COLORS[M_SN], lw=1.6, ls=":", zorder=3,
                label=M_SN)
        ax.plot(test.index, r["long"][M_CL], color=COLORS[M_CL], lw=1.8, ls="--", zorder=3,
                label=M_CL)
        ax.plot(test.index, r["long"][M_SA], color=COLORS[M_SA], lw=2.2, zorder=4,
                label=f"{M_SA} {r['order']}x{r['sorder']}[12]")
        ax.plot(show.index, show.values, color="black", lw=1.7, zorder=5, label="Actual")
        m = {k: score(test, v)["RMSE"] for k, v in r["long"].items()}
        f = "{:.2f}" if var == "Temp_mean" else "{:.1f}"      
        ax.text(0.012, 0.95, f"RMSE:  {M_SA} {f.format(m[M_SA])}   |   {M_CL} {f.format(m[M_CL])}"
                f"   |   {M_SN} {f.format(m[M_SN])}", transform=ax.transAxes, fontsize=9.5,
                va="top", bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"), zorder=6)
        ax.set_title(label_for(d, zones), loc="left")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.text(test.index[0] + pd.Timedelta(days=20), 1.02, "forecast period (2015-2019)",
                    transform=ax.get_xaxis_transform(), fontsize=9, color="#8a6d00", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [l for l in by_label if l == "Actual"] + \
            [l for l in by_label if l.startswith(M_SA) and "interval" not in l] + \
            [l for l in by_label if "interval" in l] + [M_CL, M_SN]
    fig.legend([by_label[l] for l in order], order, loc="upper left",
               bbox_to_anchor=(0.01, 0.955), ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    save_fig(fig, out_dir, fname)


def plot_errors(store, districts, zones, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    models = [M_SN, M_CL, M_SA]
    w = 0.26
    for ax, var in zip(axes, VARIABLES):
        name, _ = VARIABLES[var]
        unit = "C" if var == "Temp_mean" else "mm"
        for j, mod in enumerate(models):
            vals = [score(store[(d, var)]["test"], store[(d, var)]["long"][mod])["RMSE"]
                    for d in districts]
            bars = ax.bar(np.arange(len(districts)) + (j - 1) * w, vals, width=w,
                          color=COLORS[mod], label=mod, zorder=3)
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v,
                        f"{v:.2f}" if var == "Temp_mean" else f"{v:.1f}", ha="center",
                        va="bottom", fontsize=9)
        ax.set_xticks(range(len(districts)))
        ax.set_xticklabels([d + (f"\n{zones[d]}" if d in zones else "") for d in districts])
        ax.set_ylabel(f"RMSE ({unit}), lower is better")
        ax.set_title(f"{name}: long-range forecast error, 2015-2019", loc="left")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.legend(frameon=True, fontsize=9)
    fig.suptitle("How well do the models forecast 60 months ahead?", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "12_forecast_errors.png")


def plot_break(store, districts, zones, out_dir):
    fig, axes = plt.subplots(1, len(districts), figsize=(5.8 * len(districts), 5.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, districts):
        y = store[(d, "Precip_total")]["y"]
        annual = y.groupby(y.index.year).sum()
        annual = annual[annual.index <= 2019]
        train_avg = annual.loc[:SPLIT_DATE.year - 1].mean()
        test_avg = annual.loc[SPLIT_DATE.year:].mean()
        ax.axvspan(SPLIT_DATE.year - 0.5, 2019.5, color="#f7f1dc", zorder=0)
        ax.plot(annual.index, annual.values, color="#2e86c1", marker="o", ms=3.5, lw=1.5,
                zorder=3, label="Annual rainfall")
        ax.axhline(train_avg, color="black", ls="--", lw=1.8, zorder=4,
                   label="1981-2014 average (the level full-history models learn)")
        ax.hlines(test_avg, SPLIT_DATE.year, 2019, color="#c0392b", lw=3.5, zorder=4,
                  label="2015-2019 actual average (test period)")
        ax.axvline(BREAK_YEAR - 0.5, color="#c0392b", ls=":", lw=1.8, zorder=2,
                   label=f"Rainfall jump ({BREAK_YEAR})")
        ax.text(0.03, 0.97,
                f"Training average: {train_avg:,.0f} mm\nActual 2015-2019: {test_avg:,.0f} mm "
                f"({test_avg / train_avg - 1:+.0%})",
                transform=ax.transAxes, fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"), zorder=6)
        ax.set_title(label_for(d, zones), loc="left")
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual rainfall (mm)")
        ax.set_ylim(0, annual.max() * 1.25)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.56, -0.01))
    fig.suptitle(f"Limitation: training on the full history means learning a rainfall level that "
                 f"changed in {BREAK_YEAR}", fontsize=14, fontweight="bold", x=0.01,
                 ha="left", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, out_dir, "13_rainfall_break_limitation.png")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python forecasting.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "forecasting"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / "climate_monthly.csv"
    if not path.exists():
        sys.exit(f"Cannot find {path}. Run prepare_data.py first.")
    monthly = pd.read_csv(path, parse_dates=["Date"])
    districts, zones, source = pick_districts(data_dir, monthly)

    say("=" * 66)
    say("FORECASTING REPORT - monthly temperature and rainfall")
    say("=" * 66)
    say(f"Districts: {', '.join(districts)}  ({source})")
    say(f"Training: 1981-01 to {(SPLIT_DATE - pd.offsets.MonthBegin(1)):%Y-%m} (full history, "
        f"{(SPLIT_DATE.year - 1981) * 12} months)")
    say("Testing : 2015-01 to 2019-12 (60 months)")
    say("The 2004 rainfall jump is NOT removed; it is measured as a limitation below.")

    store = {}
    say("\nFitting models (about a minute) ...")
    for d in districts:
        s = monthly[monthly["District"] == d].set_index("Date").sort_index().asfreq("MS")
        for var in VARIABLES:
            store[(d, var)] = run_series(s[var], var)
            r = store[(d, var)]
            say(f"  {d:<12} {VARIABLES[var][0]:<12} SARIMA{r['order']}x{r['sorder']}[12]  "
                f"AIC {r['aic']:.0f}")

    # metrics tables
    rows = []
    for (d, var), r in store.items():
        for mod, pred in r["long"].items():
            rows.append({"District": d, "Zone": zones.get(d, ""), "Variable": VARIABLES[var][0],
                         "Horizon": "Long-range (60 months at once)", "Model": mod,
                         **score(r["test"], pred)})
        for mod, pred in r["one"].items():
            rows.append({"District": d, "Zone": zones.get(d, ""), "Variable": VARIABLES[var][0],
                         "Horizon": "1 month ahead", "Model": mod, **score(r["test"], pred)})
        rows.append({"District": d, "Zone": zones.get(d, ""), "Variable": VARIABLES[var][0],
                     "Horizon": "Long-range (60 months at once)",
                     "Model": "DIAGNOSTIC: climatology 2004+ only",
                     **score(r["test"], r["clim_post"])})
    metrics = pd.DataFrame(rows).round(3)

    for var, (vname, _) in VARIABLES.items():
        unit = "C" if var == "Temp_mean" else "mm"
        say(f"\n--- {vname.upper()}: long-range error (RMSE in {unit}; lower is better) ---")
        t = metrics[(metrics["Variable"] == vname) &
                    (metrics["Horizon"].str.startswith("Long"))]
        tab = t.pivot(index="District", columns="Model", values="RMSE").reindex(districts)
        tab = tab[[M_SN, M_CL, M_SA, "DIAGNOSTIC: climatology 2004+ only"]]
        tab["SARIMA vs climatology"] = ((1 - tab[M_SA] / tab[M_CL]) * 100).map(pct)
        say(tab.round(2).to_string())
        bias = t.pivot(index="District", columns="Model", values="Bias").reindex(districts)
        say(f"\nBias in {unit} (forecast minus actual; negative = forecasts too low):")
        say(bias[[M_SN, M_CL, M_SA]].round(2).to_string())
        say("\nShare of the 60 test months inside SARIMA's 95% interval: " +
            ", ".join(f"{d} {store[(d, var)]['coverage']:.0%}" for d in districts))

        say(f"\n--- {vname.upper()}: 1 month ahead (RMSE in {unit}) ---")
        t1 = metrics[(metrics["Variable"] == vname) & (metrics["Horizon"] == "1 month ahead")]
        tab1 = t1.pivot(index="District", columns="Model", values="RMSE").reindex(districts)
        tab1 = tab1[[M_SN, M_SA]]
        tab1["SARIMA vs seasonal-naive"] = ((1 - tab1[M_SA] / tab1[M_SN]) * 100).map(pct)
        say(tab1.round(2).to_string())

    # rainfall limitation
    say("\n--- RAINFALL LIMITATION: the 2004 jump ---")
    say(f"{'District':<13}{'1981-' + str(BREAK_YEAR - 1):>12}{str(BREAK_YEAR) + '-2014':>12}"
        f"{'2015-2019':>12}{'train avg':>12}   (annual rainfall, mm)")
    for d in districts:
        y = store[(d, "Precip_total")]["y"]
        a = y.groupby(y.index.year).sum()
        say(f"{d:<13}{a.loc[1981:BREAK_YEAR - 1].mean():>12.0f}{a.loc[BREAK_YEAR:2014].mean():>12.0f}"
            f"{a.loc[2015:2019].mean():>12.0f}{a.loc[1981:2014].mean():>12.0f}")
    say("A model trained on all of 1981-2014 learns the 'train avg' level, but 2015-2019 is "
        "much wetter,")
    say("so climatology under-forecasts rainfall (negative bias above). SARIMA copes better "
        "because it works")
    say("with year-to-year seasonal changes, but it cannot know the jump happened.")

    # figures and files
    say("\nSaving figures and tables:")
    plot_forecasts(store, districts, zones, "Temp_mean", "10_temperature_forecasts.png",
                   "Temperature forecasts, 2015-2019: the seasonal cycle does most of the work",
                   out_dir)
    plot_forecasts(store, districts, zones, "Precip_total", "11_rainfall_forecasts.png",
                   "Rainfall forecasts, 2015-2019: much harder, and affected by the 2004 jump",
                   out_dir)
    plot_errors(store, districts, zones, out_dir)
    plot_break(store, districts, zones, out_dir)

    metrics.to_csv(out_dir / "forecast_metrics.csv", index=False)
    frames = []
    for (d, var), r in store.items():
        frames.append(pd.DataFrame({
            "Date": r["test"].index, "District": d, "Variable": VARIABLES[var][0],
            "Actual": r["test"].values,
            "Seasonal_naive": r["long"][M_SN].values, "Climatology": r["long"][M_CL].values,
            "SARIMA_long_range": r["long"][M_SA].values,
            "SARIMA_lower95": r["ci_lo"].values, "SARIMA_upper95": r["ci_hi"].values,
            "SARIMA_1_month_ahead": r["one"][M_SA].values,
            "Seasonal_naive_1_month_ahead": r["one"][M_SN].values,
            "Diagnostic_climatology_2004plus": r["clim_post"].values}))
    out_df = pd.concat(frames)
    num_cols = out_df.select_dtypes("number").columns
    out_df[num_cols] = out_df[num_cols].round(3)
    out_df.to_csv(out_dir / "forecasts.csv", index=False)
    (out_dir / "forecasting_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
