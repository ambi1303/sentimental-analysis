# backend/app/utils/preprocess_raw.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parents[3]  # → project root
DATA = ROOT / "data"

def make_daily():
    # 1) Sentiment140 proxy:
    df = pd.read_csv(
        DATA / "sentiment140" / "training.1600000.processed.noemoticon.csv",
        encoding="latin-1",
        names=["polarity","id","date","query","user","text"]
    )
    df["mood_score"] = df["polarity"].map({0:1,2:5,4:10})
    out = df.rename(columns={"user":"user_id","text":"feeling_text"})[
        ["user_id","date","mood_score","feeling_text"]
    ]
    (DATA / "daily.csv").parent.mkdir(exist_ok=True)
    out.to_csv(DATA / "daily.csv", index=False)
    print(f"Wrote {len(out)} rows to data/daily.csv")

def make_weekly():
    # 2) PSS-10 from tech_survey/survey.csv
    df = pd.read_csv(DATA / "tech_survey" / "survey.csv")
    pss_cols = [c for c in df.columns if c.lower().startswith("pss")]
    df["avg_stress"] = df[pss_cols].sum(axis=1)
    df["overwhelm_idx"] = pd.cut(
        df["avg_stress"], bins=[-1,8,16,24,32,40], labels=False
    ).astype(int)
    # assign week by simple grouping (e.g. row index // 10 +1)
    out = df.reset_index().rename(columns={"index":"user_id"})[
        ["user_id","avg_stress","overwhelm_idx"]
    ]
    out["week"] = (out["user_id"] // 10) + 1
    out = out[["user_id","week","avg_stress","overwhelm_idx"]]
    out.to_csv(DATA / "weekly.csv", index=False)
    print(f"Wrote {len(out)} rows to data/weekly.csv")

def make_monthly():
    # 3) Burnout proxy + PHQ/GAD from tech_survey
    burn = pd.read_csv(DATA / "burnout" / "train.csv")
    tech = pd.read_csv(DATA / "tech_survey" / "survey.csv")
    # map columns:
    m = burn.rename(columns={
        "burn_rate":"EE",
        "mental_fatigue":"DP",
        "work_accomplishment":"PA"
    })[["user_id","EE","DP","PA"]]
    # compute BDI (using phq_1..phq_9) and GAD (gad_1..gad_7)
    phq = [c for c in tech.columns if c.lower().startswith("phq")]
    gad = [c for c in tech.columns if c.lower().startswith("gad")]
    tech["BDI"] = tech[phq].sum(axis=1)
    tech["GAD"] = tech[gad].sum(axis=1)
    tech["MHC"] = 0    # placeholder
    tech = tech.reset_index().rename(columns={"index":"user_id"})
    out = m.merge(tech[["user_id","BDI","GAD","MHC"]], on="user_id")
    out["month"] = "2025-05"
    out = out[["user_id","month","EE","DP","PA","BDI","GAD","MHC"]]
    out.to_csv(DATA / "monthly.csv", index=False)
    print(f"Wrote {len(out)} rows to data/monthly.csv")

if __name__ == "__main__":
    make_daily()
    make_weekly()
    make_monthly()
