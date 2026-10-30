import sys
from pathlib import Path

import numpy as np
import pandas as pd


RAINY_DAY_MM = 1.0  

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Pre-monsoon", 4: "Pre-monsoon", 5: "Pre-monsoon",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon",
}



def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(SEASON_MAP)
    df["RainyDay"] = (df["Precip"] > RAINY_DAY_MM).astype(int)

    df = df.sort_values(["District", "Date"]).reset_index(drop=True)
    return df


def quality_report(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Rows x columns      : {df.shape[0]:,} x {df.shape[1]}")
    print(f"Date range          : {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Districts           : {df['District'].nunique()}")
    print(f"Missing values      : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows      : {int(df.duplicated(['Date', 'District']).sum())}")

    full_days = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    days_per_district = df.groupby("District")["Date"].nunique()
    incomplete = days_per_district[days_per_district != len(full_days)]
    print(f"Expected days/dist. : {len(full_days):,}")
    print(f"Incomplete districts: {len(incomplete)}")

    
    print("\nLogic checks (rows that look physically odd):")
    checks = {
        "MaxTemp < MinTemp": (df["MaxTemp_2m"] < df["MinTemp_2m"]).sum(),
        "Temp_2m above MaxTemp": (df["Temp_2m"] > df["MaxTemp_2m"] + 0.01).sum(),
        "Temp_2m below MinTemp": (df["Temp_2m"] < df["MinTemp_2m"] - 0.01).sum(),
        "Negative precipitation": (df["Precip"] < 0).sum(),
        "RH_2m outside 0-100": ((df["RH_2m"] < 0) | (df["RH_2m"] > 100)).sum(),
        "Negative wind speed": (df["WindSpeed_10m"] < 0).sum(),
    }
    for name, n in checks.items():
        print(f"  {name:<25}: {int(n):,}")

    print("\nExtremes worth a glance:")
    print(f"  Coldest MinTemp_2m : {df['MinTemp_2m'].min():.2f} C "
          f"({df.loc[df['MinTemp_2m'].idxmin(), 'District']})")
    print(f"  Hottest MaxTemp_2m : {df['MaxTemp_2m'].max():.2f} C "
          f"({df.loc[df['MaxTemp_2m'].idxmax(), 'District']})")
    print(f"  Wettest day        : {df['Precip'].max():.1f} mm "
          f"({df.loc[df['Precip'].idxmax(), 'District']}, "
          f"{df.loc[df['Precip'].idxmax(), 'Date'].date()})")
    print(f"  Dry days (0 mm)    : {(df['Precip'] == 0).mean():.1%}")
    print(f"  Rainy days (>{RAINY_DAY_MM:g} mm): {df['RainyDay'].mean():.1%}")
    print("=" * 60)



def make_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["District", "Latitude", "Longitude", "Year", "Month"], as_index=False)
    monthly = g.agg(
        Precip_total=("Precip", "sum"),             
        Precip_max_day=("Precip", "max"),           
        Rainy_days=("RainyDay", "sum"),           
        Temp_mean=("Temp_2m", "mean"),
        TempMax_mean=("MaxTemp_2m", "mean"),
        TempMin_mean=("MinTemp_2m", "mean"),
        TempRange_mean=("TempRange_2m", "mean"),
        TempMax_peak=("MaxTemp_2m", "max"),        
        TempMin_low=("MinTemp_2m", "min"),         
        WetBulbTemp_mean=("WetBulbTemp_2m", "mean"),
        EarthSkinTemp_mean=("EarthSkinTemp", "mean"),
        Pressure_mean=("Pressure", "mean"),
        Humidity_mean=("Humidity_2m", "mean"),
        RH_mean=("RH_2m", "mean"),
        Wind10_mean=("WindSpeed_10m", "mean"),
        Wind50_mean=("WindSpeed_50m", "mean"),
    )
    monthly["Season"] = monthly["Month"].map(SEASON_MAP)
    monthly["Date"] = pd.to_datetime(dict(year=monthly["Year"], month=monthly["Month"], day=1))
    cols = ["District", "Date", "Year", "Month", "Season"] + [
        c for c in monthly.columns
        if c not in ("District", "Date", "Year", "Month", "Season")
    ]
    return monthly[cols].sort_values(["District", "Date"]).reset_index(drop=True)



def make_district_features(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    
    yearly_rain = df.groupby(["District", "Year"])["Precip"].sum().groupby("District").mean()

   
    monsoon_rain = (
        df[df["Season"] == "Monsoon"].groupby(["District", "Year"])["Precip"].sum()
        .groupby("District").mean()
    )

    feats = df.groupby("District").agg(
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
        Temp_mean=("Temp_2m", "mean"),
        TempRange_mean=("TempRange_2m", "mean"),
        Pressure_mean=("Pressure", "mean"),   
        Humidity_mean=("Humidity_2m", "mean"),
        RH_mean=("RH_2m", "mean"),
        Wind10_mean=("WindSpeed_10m", "mean"),
        RainyDay_share=("RainyDay", "mean"),
    )
    feats["Annual_rain_mm"] = yearly_rain
    feats["Monsoon_rain_share"] = monsoon_rain / yearly_rain

   
    month_means = df.groupby(["District", "Month"])["Temp_2m"].mean().unstack()
    feats["Seasonal_temp_swing"] = month_means.max(axis=1) - month_means.min(axis=1)

   
    feats["Winter_MinTemp"] = (
        df[df["Season"] == "Winter"].groupby("District")["MinTemp_2m"].mean()
    )
    return feats.round(4).reset_index()



def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python prepare_data.py dailyclimate.csv [output_dir]")

    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = load_and_clean(csv_path)
    quality_report(df)

    monthly = make_monthly(df)
    feats = make_district_features(df, monthly)

    df.to_csv(out_dir / "climate_daily_clean.csv", index=False)
    monthly.to_csv(out_dir / "climate_monthly.csv", index=False)
    feats.to_csv(out_dir / "district_features.csv", index=False)

    print("\nSaved:")
    print(f"  climate_daily_clean.csv : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  climate_monthly.csv     : {monthly.shape[0]:,} rows x {monthly.shape[1]} cols")
    print(f"  district_features.csv   : {feats.shape[0]} rows x {feats.shape[1]} cols")


if __name__ == "__main__":
    main()
