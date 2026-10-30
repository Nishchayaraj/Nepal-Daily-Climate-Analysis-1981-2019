# Nepal Daily Climate Analysis, 1981-2019

**Climate zones and forecasting in Nepal.** A capstone project from MINspire 2026 by Team CAT-MAN-DU.


MINspire 2026 was a hybrid machine learning and mathematics bootcamp led by Prof. Trushant Majmudar (NYU Courant). This project asks what 39 years of daily climate data can and cannot tell us about Nepal.

## Three questions

1. Can the data alone group Nepal's districts into climate zones?
2. How well can we forecast monthly temperature and rainfall, and is it harder in some zones?
3. How far can we trust the data?

## Data

- **Source:** Nepal Daily Climate 1981-2019 on Kaggle (uploaded by saimondahal). It is modelled data (NASA POWER), not station readings.
- **Size:** 62 districts, 39 years, 883,128 daily rows, no missing values.
- **Why this matters:** modelled data can contain artificial jumps when the way it was produced changes (see Findings).

The raw file `dailyclimate.csv` is **not** included here because it is very large. Download it from Kaggle, extract it, and put it in this folder. Please check the dataset's license on Kaggle before reusing it.

## Method

1. **Zones.** Four features per district (mean temperature, annual rainfall, monsoon share, seasonal temperature swing) are grouped with K-Means.
2. **Forecasts.** Monthly temperature and rainfall are forecast for 2015-2019, training only on earlier years. We compare simple baselines, SARIMA, gradient boosting, and an average of SARIMA and gradient boosting.
3. **Stress tests.** Results are checked across districts, test windows and methods.

## Main findings

- **Three zones.** Using temperature and altitude, three zones are clearly supported (silhouette 0.54, stability 0.99): 29 lowland, 24 hill and 9 high-mountain districts. All 17 Terai districts fall in the lowland zone, and 71% of districts match Nepal's official Terai / Hill / Mountain belts. The disagreements are at the hill-mountain edge. When rainfall features are added, two zones score higher.
- **Temperature is easy, rainfall is hard.** SARIMA beats the baselines in most districts. Temperature error is about 1 degree C. Rainfall error is about 50-65% of average monthly rainfall.
- **Better models help a little.** Averaging SARIMA with gradient boosting was best in 5 of 6 tests, by about 2-5%.
- **Errors grow toward the west.** Forecast error correlates with longitude (-0.62 for temperature, -0.69 for rainfall). We did not test the cause.
- **A likely data artifact.** There is a jump in the data around 2004. We could not prove its cause. The finding of no significant warming trend (+0.014 degrees C per decade, p = 0.80) may be affected by it.

## Limitations

- The data is modelled, not measured at stations.
- The 2004 jump is unexplained.
- The high-mountain zone has few districts (9).
- Most of the forecast testing uses one window (2015-2019).
- The west-to-east pattern in the errors is not explained.

Possible next steps: check the 2004 jump against station records, add elevation as a feature, fit separate west and east models, and test longer forecast horizons.

## How to run

Install the libraries:

```
pip install -r requirements.txt
```

Then run the scripts in this order. Each script takes `.` (the current folder) as its input folder:

```
python prepare_data.py dailyclimate.csv
python eda_plots.py .
python clustering.py .
python forecasting.py .
python break_analysis.py .
python zone_forecasting.py .
python improved_models.py .
python clustering_robustness.py .
python zone_robustness_check.py .
```

`prepare_data.py` creates three CSV files: `climate_daily_clean.csv`, `climate_monthly.csv` and `district_features.csv`. Copy them into this folder. The last two are already included here. The first is large, so run `prepare_data.py` to regenerate it.

## What is in this repository

| File or folder | What it holds |
|---|---|
| `prepare_data.py` | Cleans the raw data and builds the monthly and per-district tables |
| `eda_plots.py` | Exploratory plots (output in `eda_plots/`) |
| `clustering.py` | K-Means climate zones (output in `clustering/`) |
| `forecasting.py` | Forecasts for 2015-2019 (output in `forecasting/`) |
| `break_analysis.py` | Looks for the jump in the data (output in `break_analysis/`) |
| `zone_forecasting.py` | Forecast accuracy by zone (output in `zone_forecasting/`) |
| `improved_models.py` | Gradient boosting and the SARIMA + gradient boosting average (output in `improved_models/`) |
| `clustering_robustness.py` | Checks how stable the zones are (output in `clustering_robustness/`) |
| `zone_robustness_check.py` | Further zone checks (output in `zone_robustness/`) |
| `nepal_outline.geojson` | Map outline used for the figures |
| `capstone_slides.pdf` | The 17-slide capstone presentation |

## Acknowledgements

Developed during MINspire 2026 (Mathematics Initiatives in Nepal). Data: Nepal Daily Climate 1981-2019 on Kaggle (saimondahal), derived from NASA POWER.
