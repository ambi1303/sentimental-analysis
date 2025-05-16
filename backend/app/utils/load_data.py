# backend/app/utils/load_data.py

import pandas as pd

def load_daily(path: str = "data/daily.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # Enforce schema: user_id, date, mood_score, feeling_text
    expected = ["user_id","date","mood_score","feeling_text"]
    assert all(col in df.columns for col in expected), f"Missing daily cols"
    return df[expected]

def load_weekly(path: str = "data/weekly.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map overwhelm_freq → index 0–4
    freq_map = {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Always":4}
    df["overwhelm_idx"] = df["overwhelm_freq"].map(freq_map)
    expected = ["user_id","week","avg_stress","overwhelm_idx"]
    assert all(col in df.columns for col in expected), f"Missing weekly cols"
    return df[expected]

def load_monthly(path: str = "data/monthly.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["user_id","month","EE","DP","PA","BDI","GAD","MHC"]
    assert all(col in df.columns for col in expected), f"Missing monthly cols"
    return df[expected]

def load_all() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    daily = load_daily()
    weekly = load_weekly()
    monthly = load_monthly()
    return daily, weekly, monthly
