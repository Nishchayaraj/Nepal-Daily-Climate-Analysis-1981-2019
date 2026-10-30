import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Settings

TEST_WINDOWS = [(2005, 2009), (2010, 2014), (2015, 2019)]   # each is 60 months
FINAL = (2015, 2019)
BREAK_DATE = pd.Timestamp("2004-01-01")
VARIABLES = {"Temp_mean": "Temperature", "Precip_total": "Rainfall"}
SARIMA_SPEC = {"Temp_mean": ((1, 0, 0), (0, 1, 1, 12)),
               "Precip_total": ((2, 0, 0), (1, 1, 1, 12))}
PREDICTORS = ["Humidity_mean", "RH_mean", "Pressure_mean", "Wind10_mean", "TempRange_mean"]
PREDICTOR_NAMES = {"Humidity_mean": "Specific humidity", "RH_mean": "Relative humidity",
                   "Pressure_mean": "Pressure", "Wind10_mean": "Wind speed (10 m)",
                   "TempRange_mean": "Daily temperature range"}
GB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)

H_LONG, H_ONE = "Long-range (60 months at once)", "1 month ahead"

MODELS = [
    (H_LONG, "Seasonal-naive", "LR_sn"),
    (H_LONG, "Climatology", "LR_cl"),
    (H_LONG, "SARIMA", "LR_sa"),
    (H_LONG, "SARIMA + 2004 step", "LR_step"),
    (H_LONG, "SARIMA on log scale", "LR_log"),
    (H_ONE, "Seasonal-naive", "OS_sn"),
    (H_ONE, "Climatology", "LR_cl"),
    (H_ONE, "SARIMA", "OS_sa"),
    (H_ONE, "Gradient boosting (own history)", "OS_gb_own"),
    (H_ONE, "Gradient boosting (+ predictors)", "OS_gb_full"),
    (H_ONE, "Average of SARIMA and gradient boosting", "OS_ens"),
]
ONE_PLOT = ["Seasonal-naive", "SARIMA", "Gradient boosting (own history)",
            "Gradient boosting (+ predictors)", "Average of SARIMA and gradient boosting"]
COLORS = {"Seasonal-naive": "#E69F00", "Climatology": "#7f8c8d", "SARIMA": "#c0392b",
          "SARIMA + 2004 step": "#e59866", "SARIMA on log scale": "#7b241c",
          "Gradient boosting (own history)": "#b39ddb",
          "Gradient boosting (+ predictors)": "#7d3c98",
          "Average of SARIMA and gradient boosting": "#1e8449"}
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
    print(text, flush=True)
    REPORT.append(str(text))


def progress(text):
    """Show progress on screen without adding it to the saved report."""
    print(text, flush=True)


def save_fig(fig, out_dir, name):
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color="grey", ha="left", va="bottom")
    fig.savefig(out_dir / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"  saved {name}")


def wname(w):
    return f"{w[0]}-{w[1]}"


# Time-series models (one district at a time)

def climatology(train, index):
    avg = train.groupby(train.index.month).mean()
    return avg.reindex([d.month for d in index]).values


def seasonal_naive_long(train, index):
    last = train.iloc[-12:]
    lookup = {d.month: v for d, v in zip(last.index, last.values)}
    return np.array([lookup[d.month] for d in index])


def fit_sarima(y, order, sorder, exog=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, order=order, seasonal_order=sorder, exog=exog,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=200)


def district_forecasts(y, var, origin, end, do_step, do_log):
    """All time-series forecasts for one district and one test window."""
    train = y[y.index < origin]
    test = y[(y.index >= origin) & (y.index < end)]
    n = len(test)
    rain = var == "Precip_total"
    lo = 0.0 if rain else None
    order, sorder = SARIMA_SPEC[var]
    nan = np.full(n, np.nan)
    out = {"LR_sn": seasonal_naive_long(train, test.index),
           "LR_cl": climatology(train, test.index),
           "OS_sn": y.shift(12)[test.index].values,
           "LR_sa": nan, "OS_sa": nan, "LR_step": nan, "LR_log": nan}
    try:
        res = fit_sarima(train, order, sorder)
        out["LR_sa"] = pd.Series(res.get_forecast(steps=n).predicted_mean.values).clip(lower=lo).values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_all = res.apply(y[y.index < end], refit=False)
        out["OS_sa"] = pd.Series(res_all.get_prediction(start=len(train), dynamic=False)
                                 .predicted_mean.values).clip(lower=lo).values
    except Exception:
        pass
    if do_step and rain:
        try:
            ex_tr = (train.index >= BREAK_DATE).astype(float).reshape(-1, 1)
            ex_te = (test.index >= BREAK_DATE).astype(float).reshape(-1, 1)
            res = fit_sarima(train, order, sorder, exog=ex_tr)
            f = res.get_forecast(steps=n, exog=ex_te).predicted_mean.values
            out["LR_step"] = np.clip(f, 0, None)
        except Exception:
            pass
    if do_log and rain:
        try:
            res = fit_sarima(np.log1p(train), order, sorder)
            f = np.expm1(res.get_forecast(steps=n).predicted_mean.values)
            out["LR_log"] = np.clip(f, 0, None)
        except Exception:
            pass
    return test, out


# Gradient boosting (one model for all districts)

def build_features(m, var, origin):
    """Features for forecasting month t from information up to month t-1.
    Everything is expressed as a deviation from the district's own monthly climatology,
    which is computed from TRAINING data only."""
    d = m[["District", "Date", "Latitude", "Longitude", "Temp_mean", "Precip_total"]
          + PREDICTORS].copy()
    d["own"] = d[var]
    d["other"] = np.log1p(d["Precip_total"]) if var == "Temp_mean" else d["Temp_mean"]
    d["mon"] = d["Date"].dt.month
    tr = d["Date"] < origin

    raw = ["own", "other"] + PREDICTORS
    anom = pd.DataFrame({"District": d["District"], "Date": d["Date"]})
    for c in raw:
        clim = d[tr].groupby(["District", "mon"])[c].mean().rename("cl")
        cl = d.join(clim, on=["District", "mon"])["cl"]
        anom["a_" + c] = d[c] - cl
        anom["cl_" + c] = cl
    g = anom.groupby("District")

    F = pd.DataFrame({"District": d["District"], "Date": d["Date"]})
    F["mon"] = d["mon"]
    for lag in (1, 2, 3, 12):
        F[f"own_l{lag}"] = g["a_own"].shift(lag)
    F["own_r12"] = g["a_own"].transform(lambda s: s.shift(1).rolling(12).mean())
    F["lat"] = d["Latitude"]
    F["lon"] = d["Longitude"]
    F["pmean"] = d["District"].map(d[tr].groupby("District")["Pressure_mean"].mean())
    own_cols = [c for c in F.columns if c not in ("District", "Date")]
    for k in ["other"] + PREDICTORS:
        F[f"{k}_l1"] = g["a_" + k].shift(1)
        F[f"{k}_r12"] = g["a_" + k].transform(lambda s: s.shift(1).rolling(12).mean())
    F["post"] = (d["Date"] >= BREAK_DATE).astype(float)
    full_cols = [c for c in F.columns if c not in ("District", "Date")]
    return d, anom, F, own_cols, full_cols, tr


def gradient_boosting(m, var, origin, end):
    d, anom, F, own_cols, full_cols, tr = build_features(m, var, origin)
    test = (d["Date"] >= origin) & (d["Date"] < end)
    train_rows = tr & F[full_cols].notna().all(axis=1)
    target = anom["a_own"]
    rain = var == "Precip_total"
    preds = pd.DataFrame({"District": d.loc[test, "District"].values,
                          "Date": d.loc[test, "Date"].values})
    models = {}
    for name, cols in [("OS_gb_own", own_cols), ("OS_gb_full", full_cols)]:
        mdl = HistGradientBoostingRegressor(**GB_PARAMS)
        mdl.fit(F.loc[train_rows, cols], target[train_rows])
        yhat = anom.loc[test, "cl_own"] + mdl.predict(F.loc[test, cols])
        preds[name] = np.clip(yhat.values, 0, None) if rain else yhat.values
        models[name] = (mdl, cols)
    extra = (models["OS_gb_full"][0], F.loc[test, full_cols], target[test], full_cols)
    return preds, extra


# Metrics

def score(actual, pred, var, scale):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    if np.isnan(p).any():
        return None
    err = p - a
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rain = var == "Precip_total"
    return {"RMSE": rmse, "MAE": mae, "Bias": float(np.mean(err)),
            "Error": 100 * rmse / float(np.mean(a)) if rain else rmse,
            "Bias_pct": 100 * float(np.mean(err)) / float(np.mean(a)) if rain else np.nan,
            "MASE": mae / scale}


def pretty_feature(f, var):
    base = {"mon": "Month of year", "lat": "Latitude", "lon": "Longitude",
            "pmean": "Altitude proxy (mean pressure)", "post": "After-2004 flag",
            "own_l1": "Own value, last month", "own_l2": "Own value, 2 months ago",
            "own_l3": "Own value, 3 months ago", "own_l12": "Own value, same month last year",
            "own_r12": "Own 12-month average level"}
    if f in base:
        return base[f]
    k, suffix = f.rsplit("_", 1)
    name = ("Rainfall" if var == "Temp_mean" else "Temperature") if k == "other" \
        else PREDICTOR_NAMES[k]
    return f"{name}, " + ("last month" if suffix == "l1" else "12-month average")


# Figures

def plot_rolling(summary, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    wins = [wname(w) for w in TEST_WINDOWS]
    for ax, (var, vname) in zip(axes, VARIABLES.items()):
        s = summary[(summary["Variable"] == vname) & (summary["Horizon"] == H_ONE)]
        w = 0.8 / len(ONE_PLOT)
        for j, mod in enumerate(ONE_PLOT):
            vals = [s[(s["Window"] == win) & (s["Model"] == mod)]["Median_error"].iloc[0] for win in wins]
            bars = ax.bar(np.arange(len(wins)) + (j - (len(ONE_PLOT) - 1) / 2) * w, vals, width=w,
                          color=COLORS[mod], label=mod, zorder=3)
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}" if var == "Temp_mean" else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(range(len(wins)))
        ax.set_xticklabels([f"Test {x}" for x in wins])
        ax.set_ylabel("Median error across 62 districts, lower is better\n" +
                      ("(RMSE, C)" if var == "Temp_mean" else "(RMSE, % of average monthly rainfall)"))
        ax.set_title(f"{vname}: one month ahead", loc="left")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.6, -0.02))
    fig.suptitle("Does the best model change with the test window? Three rolling windows",
                 fontsize=15, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    save_fig(fig, out_dir, "20_rolling_windows.png")


def plot_longrange(summary, res, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
    mods = ["Seasonal-naive", "Climatology", "SARIMA", "SARIMA + 2004 step", "SARIMA on log scale"]
    wins = [wname(w) for w in TEST_WINDOWS]
    s = summary[(summary["Variable"] == "Rainfall") & (summary["Horizon"] == H_LONG)]
    w = 0.8 / len(mods)
    for j, mod in enumerate(mods):
        for i, win in enumerate(wins):
            row = s[(s["Window"] == win) & (s["Model"] == mod)]
            if row.empty or np.isnan(row["Median_error"].iloc[0]):
                continue
            v = row["Median_error"].iloc[0]
            x = i + (j - (len(mods) - 1) / 2) * w
            a.bar(x, v, width=w, color=COLORS[mod], zorder=3, label=mod if i == len(wins) - 1 or
                  (mod != "SARIMA on log scale" and i == 0) else None)
            a.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    a.set_xticks(range(len(wins)))
    a.set_xticklabels([f"Test {x}" for x in wins])
    a.set_ylabel("Median error, % of average monthly rainfall")
    a.set_title("Rainfall, 60 months at once: effect of the 2004 step and log scale", loc="left")
    a.grid(axis="y", alpha=0.3, zorder=0)
    a.set_ylim(0, a.get_ylim()[1] * 1.12)
    h, l = a.get_legend_handles_labels()
    by = dict(zip(l, h))
    a.legend([by[m_] for m_ in mods if m_ in by], [m_ for m_ in mods if m_ in by], fontsize=9,
             frameon=True, loc="upper right")

    fin = res[(res["Variable"] == "Rainfall") & (res["Horizon"] == H_LONG) & (res["Window"] == wname(FINAL))]
    bias = fin.groupby("Model")["Bias_pct"].median().reindex(["Climatology", "SARIMA",
                                                              "SARIMA + 2004 step", "SARIMA on log scale"])
    bars = b.bar(range(len(bias)), bias.values, color=[COLORS[i] for i in bias.index], zorder=3)
    for r_, v in zip(bars, bias.values):
        b.text(r_.get_x() + r_.get_width() / 2, v + (2 if v >= 0 else -2), f"{v:+.0f}%", ha="center",
               va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    b.axhline(0, color="black", lw=1)
    b.set_xticks(range(len(bias)))
    b.set_xticklabels([i.replace(" + ", "\n+ ").replace(" on ", "\non ") for i in bias.index], fontsize=9)
    b.set_ylabel("Median bias, % of average rainfall\n(negative = forecasts too low)")
    b.set_title(f"Bias in {wname(FINAL)}", loc="left")
    b.grid(axis="y", alpha=0.3, zorder=0)
    b.set_ylim(min(bias.min() * 1.3, -10), max(bias.max() * 1.3, 10))
    fig.suptitle("Long-range rainfall: what the 2004 step and the log scale did", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "21_longrange_variants.png")


def plot_ensemble(res, zones, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    zlist = sorted(set(zones.values()), key=lambda z: list(zones.values()).index(z)) if zones else []
    zcol = {z: ZONE_COLORS[i % len(ZONE_COLORS)] for i, z in enumerate(zlist)}
    for ax, (var, vname) in zip(axes, VARIABLES.items()):
        sub = res[(res["Variable"] == vname) & (res["Horizon"] == H_ONE) & (res["Window"] == wname(FINAL))]
        a = sub[sub["Model"] == "SARIMA"].set_index("District")["Error"]
        e = sub[sub["Model"] == "Average of SARIMA and gradient boosting"].set_index("District")["Error"]
        g = sub[sub["Model"] == "Gradient boosting (+ predictors)"].set_index("District")["Error"]
        idx = a.index.intersection(e.index)
        if zones:
            for z in zlist:
                ii = [i for i in idx if zones.get(i) == z]
                ax.scatter(a[ii], e[ii], s=55, color=zcol[z], edgecolor="#2c3e50", linewidth=0.6, label=z, zorder=3)
        else:
            ax.scatter(a[idx], e[idx], s=55, color="#c0392b", edgecolor="#2c3e50", linewidth=0.6, zorder=3)
        lo = min(a[idx].min(), e[idx].min()) * 0.95
        hi = max(a[idx].max(), e[idx].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.3, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        better = int((e[idx] < a[idx]).sum())
        both = int(((e[idx] < a[idx]) & (e[idx] < g[idx])).sum())
        chg = 100 * (e[idx] / a[idx] - 1).median()
        ax.text(0.04, 0.96, f"Ensemble beats SARIMA in {better} of {len(idx)} districts\n"
                f"(median change {chg:+.1f}%)\nbeats both models in {both} of {len(idx)}",
                transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb"))
        ax.set_xlabel("SARIMA error")
        ax.set_ylabel("Ensemble error (average of SARIMA and gradient boosting)")
        ax.set_title(f"{vname}, one month ahead, {wname(FINAL)}", loc="left")
        ax.grid(alpha=0.3)
        if var == "Temp_mean" and zones:
            ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("District by district: is the ensemble better than SARIMA alone?", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.text(0.01, 0.955, "Points below the dashed line: the ensemble had the smaller error.",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "22_ensemble_vs_sarima.png")


def plot_importance(imp, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, (var, vname) in zip(axes, VARIABLES.items()):
        s = imp[imp["Variable"] == vname].sort_values("Importance", ascending=False).head(10).iloc[::-1]
        colors = ["#7d3c98" if any(k in f for k in ("humidity", "Pressure", "Wind", "range", "Rainfall",
                                                    "Temperature, ", "2004")) else "#5d6d7e"
                  for f in s["Feature"]]
        ax.barh(s["Feature"], s["Importance"], color=colors, zorder=3)
        for y_, v in enumerate(s["Importance"]):
            ax.text(v, y_, f" {v:.3f}", va="center", fontsize=9)
        unit = "C" if var == "Temp_mean" else "mm"
        ax.set_xlabel(f"Increase in error ({unit}) when the feature is scrambled\n"
                      "(bigger = the model relies on it more)")
        ax.set_title(f"{vname}: top 10 features", loc="left")
        ax.grid(axis="x", alpha=0.3, zorder=0)
        ax.set_xlim(0, s["Importance"].max() * 1.18)
    fig.suptitle("What does gradient boosting rely on? (2015-2019 test months)", fontsize=15,
                 fontweight="bold", x=0.01, ha="left", y=1.02)
    fig.text(0.01, 0.955, "Purple = outside predictors (humidity, pressure, wind, ...); grey = the "
             "district's own history, season and location.", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "23_feature_importance.png")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python improved_models.py <data_dir> [output_dir]")
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "improved_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = data_dir / "climate_monthly.csv"
    if not mpath.exists():
        sys.exit(f"Cannot find {mpath}. Run prepare_data.py first.")
    m = pd.read_csv(mpath, parse_dates=["Date"]).sort_values(["District", "Date"]).reset_index(drop=True)
    cpath = data_dir / "clustering" / "district_clusters.csv"
    zones = {}
    if cpath.exists():
        info = pd.read_csv(cpath)
        zones = dict(zip(info["District"], info["Zone"]))
        zone_order = info.sort_values("Cluster")["Zone"].drop_duplicates().tolist()
    else:
        zone_order = []
    districts = sorted(m["District"].unique())

    say("=" * 72)
    say("IMPROVED MODELS REPORT - rolling-origin evaluation, 62 districts")
    say("=" * 72)
    say(f"Test windows: {', '.join(wname(w) for w in TEST_WINDOWS)} (each trains only on earlier data)")
    progress("Estimated run time: 6 to 10 minutes. Progress is shown below.\n")

    t_start = time.time()
    rows = []
    importance_rows = []
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    for w in TEST_WINDOWS:
        origin, end = pd.Timestamp(f"{w[0]}-01-01"), pd.Timestamp(f"{w[1] + 1}-01-01")
        is_final = w == FINAL
        for var, vname in VARIABLES.items():
            cache_file = cache_dir / f"{wname(w)}_{var}.pkl"
            if cache_file.exists():
                saved = pickle.loads(cache_file.read_bytes())
                rows.extend(saved["rows"])
                importance_rows.extend(saved["importance"])
                progress(f"window {wname(w)}, {vname}: loaded from cache")
                continue
            piece_rows, piece_imp = [], []
            progress(f"[{(time.time() - t_start) / 60:4.1f} min] window {wname(w)}, {vname}: "
                     f"time-series models ...")
            frames, scales = [], {}
            for dname in districts:
                y = m[m["District"] == dname].set_index("Date").sort_index().asfreq("MS")[var]
                train = y[y.index < origin]
                scales[dname] = float(np.mean(np.abs(train.values[12:] - train.values[:-12])))
                test, out = district_forecasts(y, var, origin, end, do_step=True, do_log=is_final)
                f = pd.DataFrame({"District": dname, "Date": test.index, "Actual": test.values})
                for k, v in out.items():
                    f[k] = v
                frames.append(f)
            P = pd.concat(frames, ignore_index=True)

            progress(f"[{(time.time() - t_start) / 60:4.1f} min] window {wname(w)}, {vname}: "
                     f"gradient boosting ...")
            gb, extra = gradient_boosting(m, var, origin, end)
            P = P.merge(gb, on=["District", "Date"], how="left")
            P["OS_ens"] = (P["OS_sa"] + P["OS_gb_full"]) / 2

            for (horizon, mname, col) in MODELS:
                for dname, g in P.groupby("District"):
                    sc = score(g["Actual"], g[col], var, scales[dname])
                    if sc is not None:
                        piece_rows.append({"Window": wname(w), "Variable": vname, "Horizon": horizon,
                                           "Model": mname, "District": dname,
                                           "Zone": zones.get(dname, ""), **sc})

            if is_final:
                mdl, X, yt, cols = extra
                pi = permutation_importance(mdl, X, yt, n_repeats=5, random_state=0,
                                            scoring="neg_root_mean_squared_error")
                for c, v in zip(cols, pi.importances_mean):
                    piece_imp.append({"Variable": vname, "Feature": pretty_feature(c, var),
                                      "Importance": float(v)})
            rows.extend(piece_rows)
            importance_rows.extend(piece_imp)
            cache_file.write_bytes(pickle.dumps({"rows": piece_rows, "importance": piece_imp}))

    progress(f"[{(time.time() - t_start) / 60:4.1f} min] all models fitted.")
    res = pd.DataFrame(rows)
    imp = pd.DataFrame(importance_rows)
    summary = (res.groupby(["Variable", "Horizon", "Window", "Model"])
               .agg(Districts=("Error", "size"), Median_error=("Error", "median"),
                    Median_MASE=("MASE", "median"), Median_bias_pct=("Bias_pct", "median"))
               .reset_index())

    # Part A: long-range
    say("\n" + "=" * 72)
    say("A. LONG-RANGE (60 months at once): did the log scale or the 2004 step help?")
    say("=" * 72)
    for var, vname in VARIABLES.items():
        s = summary[(summary["Variable"] == vname) & (summary["Horizon"] == H_LONG)]
        tab = s.pivot(index="Model", columns="Window", values="Median_error")
        order = [x for x in ["Seasonal-naive", "Climatology", "SARIMA", "SARIMA + 2004 step",
                             "SARIMA on log scale"] if x in tab.index]
        unit = "RMSE in C" if var == "Temp_mean" else "RMSE as % of average monthly rainfall"
        say(f"\n{vname}: median error across 62 districts ({unit})")
        say(tab.loc[order].round(2).to_string())
    fin = res[(res["Variable"] == "Rainfall") & (res["Horizon"] == H_LONG) & (res["Window"] == wname(FINAL))]
    p = fin.pivot_table(index="District", columns="Model", values="Error")
    say(f"\nRainfall {wname(FINAL)}: SARIMA + 2004 step beats plain SARIMA in "
        f"{int((p['SARIMA + 2004 step'] < p['SARIMA']).sum())} of {len(p)} districts; "
        f"the log scale beats it in {int((p['SARIMA on log scale'] < p['SARIMA']).sum())} of {len(p)}.")
    bias = fin.groupby("Model")["Bias_pct"].median()
    say("Median bias (forecast minus actual, % of average rainfall): " +
        ", ".join(f"{k} {bias[k]:+.1f}%" for k in ["Climatology", "SARIMA", "SARIMA + 2004 step",
                                                   "SARIMA on log scale"]))
    say("The log scale converts back to rainfall by exponentiating, which pulls the monsoon peaks")
    say("down; a bias correction could reduce this, but it is a limitation of the plain version.")

    # Part B: one month ahead
    say("\n" + "=" * 72)
    say("B. ONE MONTH AHEAD on three rolling windows")
    say("=" * 72)
    wins = [wname(w) for w in TEST_WINDOWS]
    best_counts = {}
    for var, vname in VARIABLES.items():
        s = summary[(summary["Variable"] == vname) & (summary["Horizon"] == H_ONE)]
        unit = "RMSE in C" if var == "Temp_mean" else "RMSE as % of average monthly rainfall"
        tab = s.pivot(index="Model", columns="Window", values="Median_error")
        order = [mm for (hh, mm, cc) in MODELS if hh == H_ONE and mm in tab.index]
        say(f"\n{vname}: median error across 62 districts ({unit})")
        say(tab.loc[order, wins].round(2).to_string())
        tab2 = s.pivot(index="Model", columns="Window", values="Median_MASE")
        say(f"\n{vname}: median MASE (below 1 = better than repeating last year's pattern)")
        say(tab2.loc[order, wins].round(2).to_string())
        if var == "Precip_total":
            say("Caution: rainfall MASE is inflated by the 2004 break. The training years (mostly")
            say("before 2004) were drier, so the 'last year' benchmark it divides by is small.")
            say("Compare models with each other, and prefer the Error table for rainfall.")
        for win in wins:
            best = tab[win].idxmin()
            best_counts[(vname, win)] = best
    say("\nBest model (lowest median error) in each window:")
    for (vname, win), b_ in best_counts.items():
        say(f"  {vname:<12} {win}: {b_}")
    n_ens = sum(1 for v in best_counts.values() if v.startswith("Average"))
    say(f"The average of SARIMA and gradient boosting was best in {n_ens} of {len(best_counts)} "
        f"window-and-variable cases.")

    say("\nHow often does the ensemble beat BOTH of its parts, district by district?")
    for var, vname in VARIABLES.items():
        for win in wins:
            sub = res[(res["Variable"] == vname) & (res["Horizon"] == H_ONE) & (res["Window"] == win)]
            pv = sub.pivot_table(index="District", columns="Model", values="Error")
            e = pv["Average of SARIMA and gradient boosting"]
            both = int(((e < pv["SARIMA"]) & (e < pv["Gradient boosting (+ predictors)"])).sum())
            say(f"  {vname:<12} {win}: {both} of {len(pv)} districts")

    say("\nDo the outside predictors help gradient boosting? (median error, with vs without)")
    for var, vname in VARIABLES.items():
        s = summary[(summary["Variable"] == vname) & (summary["Horizon"] == H_ONE)]
        t = s.pivot(index="Model", columns="Window", values="Median_error")
        own, full = t.loc["Gradient boosting (own history)", wins], t.loc["Gradient boosting (+ predictors)", wins]
        say(f"  {vname:<12} " + "   ".join(f"{w_}: {o:.2f} -> {f_:.2f}" for w_, o, f_ in zip(wins, own, full)))

    # Part C: zones and importance
    if zones:
        say("\n" + "=" * 72)
        say(f"C. BY CLIMATE ZONE, one month ahead, {wname(FINAL)} (median error)")
        say("=" * 72)
        for var, vname in VARIABLES.items():
            sub = res[(res["Variable"] == vname) & (res["Horizon"] == H_ONE) & (res["Window"] == wname(FINAL))
                      & res["Model"].isin(["SARIMA", "Gradient boosting (+ predictors)",
                                           "Average of SARIMA and gradient boosting"])]
            t = sub.groupby(["Zone", "Model"])["Error"].median().unstack().reindex(zone_order)
            say(f"\n{vname}")
            say(t.round(2).to_string())

    say("\n" + "=" * 72)
    say(f"D. WHICH PREDICTORS DOES GRADIENT BOOSTING USE? (permutation importance, {wname(FINAL)})")
    say("=" * 72)
    for var, vname in VARIABLES.items():
        s = imp[imp["Variable"] == vname].sort_values("Importance", ascending=False).head(8)
        say(f"\n{vname}: top 8 features")
        for r in s.itertuples():
            say(f"  {r.Feature:<40} {r.Importance:.3f}")

    say("\n" + "=" * 72)
    say("SUMMARY: WHAT HELPED AND WHAT DID NOT (median over 62 districts)")
    say("=" * 72)
    lr = summary[(summary["Variable"] == "Rainfall") & (summary["Horizon"] == H_LONG)] \
        .pivot(index="Model", columns="Window", values="Median_error")
    fw = wname(FINAL)
    say(f"1. Log scale (rainfall, long-range, {fw}): error {lr.loc['SARIMA on log scale', fw]:.1f}% "
        f"vs {lr.loc['SARIMA', fw]:.1f}% for plain SARIMA "
        f"({100 * (lr.loc['SARIMA on log scale', fw] / lr.loc['SARIMA', fw] - 1):+.0f}%).")
    steps = ", ".join(f"{w_} {100 * (lr.loc['SARIMA + 2004 step', w_] / lr.loc['SARIMA', w_] - 1):+.1f}%"
                      for w_ in wins)
    say(f"2. 2004 step (rainfall, long-range), change in error vs plain SARIMA: {steps}.")
    say("   A big gain in the earliest window and little later would mean the step matters mostly")
    say("   soon after a break; once the model has seen several post-break years it adds little.")
    one = summary[summary["Horizon"] == H_ONE]
    for var, vname in VARIABLES.items():
        t = one[one["Variable"] == vname].pivot(index="Model", columns="Window", values="Median_error")
        pred = ", ".join(f"{w_} {100 * (t.loc['Gradient boosting (+ predictors)', w_] / t.loc['Gradient boosting (own history)', w_] - 1):+.0f}%"
                         for w_ in wins)
        say(f"3. Outside predictors, {vname.lower()} (gradient boosting, change in error): {pred}.")
    for var, vname in VARIABLES.items():
        t = one[one["Variable"] == vname].pivot(index="Model", columns="Window", values="Median_error")
        ens = ", ".join(f"{w_} {100 * (t.loc['Average of SARIMA and gradient boosting', w_] / t.loc['SARIMA', w_] - 1):+.1f}%"
                        for w_ in wins)
        say(f"4. Average of SARIMA and gradient boosting vs SARIMA alone, {vname.lower()}: {ens}.")
    say(f"   The average was the best model in {n_ens} of {len(best_counts)} window-and-variable cases.")
    say("Remember: these are medians over 62 districts; individual districts vary a lot, and")
    say("even small improvements should be described as small.")

    say("\nSaving figures and tables:")
    plot_rolling(summary, out_dir)
    plot_longrange(summary, res, out_dir)
    plot_ensemble(res, zones, out_dir)
    plot_importance(imp, out_dir)
    res.round(4).to_csv(out_dir / "improved_metrics.csv", index=False)
    summary.round(4).to_csv(out_dir / "improved_summary.csv", index=False)
    imp.round(5).to_csv(out_dir / "feature_importance.csv", index=False)
    (out_dir / "improved_models_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    say(f"\nDone in {(time.time() - t_start) / 60:.1f} minutes. Everything is in {out_dir}")


if __name__ == "__main__":
    main()
