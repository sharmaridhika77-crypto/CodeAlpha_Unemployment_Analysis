import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set(style="whitegrid")

def clean_numeric(s):
    if s.dtype == object:
        s = s.str.replace(r'[,%\u2009]', '', regex=True).str.strip()
    return pd.to_numeric(s, errors='coerce')

def safe_rename_columns(df):
    # map detected columns by keywords to the expected names
    mapping = {}
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    targets = {
        'region': 'Region',
        'date': 'Date',
        'frequency': 'Frequency',
        'unemploy': 'Estimated Unemployment Rate',
        'employed': 'Estimated Employed',
        'labour': 'Estimated Labour Participation Rate',
        'area': 'Area'
    }
    for i, cname in enumerate(lower):
        for key, target in targets.items():
            if key in cname and target not in mapping.values():
                mapping[cols[i]] = target
                break
    if mapping:
        return df.rename(columns=mapping)
    return df

def load_and_prepare(path: str):
    pathp = Path(path)
    if not pathp.exists():
        raise FileNotFoundError(f"{path} not found")

    # try to read and parse date if present
    try:
        df = pd.read_csv(path, parse_dates=['Date'], dayfirst=True, infer_datetime_format=True)
    except Exception:
        df = pd.read_csv(path)

    df = safe_rename_columns(df)

    # If Date not parsed, try to parse common date-like column
    if 'Date' in df.columns and not np.issubdtype(df['Date'].dtype, np.datetime64):
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce', infer_datetime_format=True)
        except Exception:
            pass

    # Clean numeric columns if present
    num_cols = []
    for col in df.columns:
        lc = col.lower()
        if any(k in lc for k in ['unemploy', 'employ', 'participation', 'rate']):
            df[col] = clean_numeric(df[col])
            num_cols.append(col)

    return df

def plot_all(df):
    # Ensure Region exists
    if 'Region' not in df.columns:
        print("Column 'Region' not found. Available columns:", df.columns.tolist())
        return

    # Graph 1: average Unemployment Rate by Region (sorted)
    if 'Estimated Unemployment Rate' in df.columns:
        agg = df.groupby('Region')['Estimated Unemployment Rate'].mean().sort_values(ascending=False)
        plt.figure(figsize=(12,6))
        sns.barplot(x=agg.values, y=agg.index, palette="viridis")
        plt.xlabel("Average Estimated Unemployment Rate")
        plt.title("Average Unemployment Rate by Region")
        plt.tight_layout()
        plt.show()
    else:
        print("Column 'Estimated Unemployment Rate' not found; skipping bar chart.")

    # Graph 2: trend over time (averaged across regions)
    if 'Date' in df.columns and 'Estimated Unemployment Rate' in df.columns:
        time_df = df.dropna(subset=['Date', 'Estimated Unemployment Rate']).groupby('Date')['Estimated Unemployment Rate'].mean().reset_index()
        plt.figure(figsize=(12,6))
        sns.lineplot(data=time_df, x='Date', y='Estimated Unemployment Rate', marker='o')
        plt.xticks(rotation=45)
        plt.title("Unemployment Trend Over Time (average)")
        plt.tight_layout()
        plt.show()
    else:
        print("Missing 'Date' or 'Estimated Unemployment Rate' for time series plot.")

    # Graph 3: distribution
    if 'Estimated Unemployment Rate' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df['Estimated Unemployment Rate'].dropna(), kde=True, bins=20, color='steelblue')
        plt.title("Distribution of Estimated Unemployment Rate")
        plt.tight_layout()
        plt.show()

    # Confident highest region (by mean)
    if 'Estimated Unemployment Rate' in df.columns:
        region_mean = df.groupby('Region')['Estimated Unemployment Rate'].mean()
        if not region_mean.empty:
            highest_region = region_mean.idxmax()
            highest_rate = region_mean.max()
            print("Highest Unemployment Region (average):", highest_region)
            print("Rate (average):", highest_rate)
        else:
            print("No data to compute highest region.")

def main():
    csv_path = "Unemployment.csv"
    try:
        df = load_and_prepare(csv_path)
    except FileNotFoundError as e:
        print(e)
        return

    print("Columns after processing:", df.columns.tolist())
    print(df.head())

    plot_all(df)

if __name__ == "__main__":
    main()