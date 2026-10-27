"""Download the raw files if missing, verify them against the pinned sha256, and cache the loan fields used."""
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KEEP = ("id loan_amnt funded_amnt term int_rate installment grade sub_grade emp_length home_ownership annual_inc "
        "verification_status issue_d loan_status purpose addr_state dti delinq_2yrs earliest_cr_line fico_range_low "
        "fico_range_high inq_last_6mths open_acc pub_rec revol_bal revol_util total_acc initial_list_status total_pymnt "
        "total_rec_prncp total_rec_int total_rec_late_fee recoveries collection_recovery_fee last_pymnt_d "
        "application_type mort_acc pub_rec_bankruptcies policy_code").split()
CATS = ("term grade sub_grade emp_length home_ownership verification_status loan_status purpose addr_state "
        "initial_list_status application_type").split()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: str) -> pd.DataFrame:
    """Read the 1.7 GB file in chunks, keeping only the fields used, with compact dtypes."""
    parts = []
    for ch in pd.read_csv(path, usecols=KEEP, chunksize=200_000, low_memory=False, dtype={"id": str}):
        ch = ch[ch["loan_amnt"].notna()].copy()  # the file ends with two summary lines
        for c in CATS:
            ch[c] = ch[c].astype("category")
        for c in ch.columns:
            if ch[c].dtype == np.float64:
                ch[c] = ch[c].astype(np.float32)
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    for c in CATS:
        df[c] = df[c].astype("category")
    return df


def main(cfg_path="configs/default.yaml", strict=True):
    cfg = yaml.safe_load(open(cfg_path))
    raw = Path("data/raw"); raw.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, src in cfg["sources"].items():
        p = raw / name
        if not p.exists():
            print(f"downloading {name}"); urllib.request.urlretrieve(src["url"], p)
        got = sha256(p)
        match = got == src["sha256"]
        ok &= match
        print(f"{name}: {'OK' if match else 'CHECKSUM CHANGED ' + got}")
    if strict and not ok:
        sys.exit("Source data changed since the pinned download. Review, then update the sha256 in the config.")
    cache = Path(cfg["loans_cache"])
    if not cache.exists():
        df = load_csv(cfg["loans_csv"])
        df.to_pickle(cache)
        print(f"cached {len(df):,} loans -> {cache}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
