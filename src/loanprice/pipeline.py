"""Did the rate pay for the risk? Pricing vs realised returns, vintage loss curves, early-read forecasts and an
out-of-time screen on 36-month consumer instalment loans (LendingClub, 2007-2018)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from loanprice import core

GRADES = list("ABCDEFG")


# ----------------------------------------------------------------------------- data
def prepare(cfg) -> pd.DataFrame:
    df = pd.read_pickle(cfg["loans_cache"])
    df = df[(df["policy_code"] == 1) & ~df["loan_status"].astype(str).str.startswith("Does not meet")]
    df = df[df["term"].astype(str).str.strip() == f"{cfg['term_months']} months"].copy()
    df["issue"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["last_pymnt"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y")
    df["bad"] = df["loan_status"].astype(str).isin(cfg["bad_status"]).to_numpy()
    df["sg"] = df["sub_grade"].astype(str)
    df["sg_code"] = df["sg"].map({f"{g}{i}": k * 5 + i for k, g in enumerate(GRADES) for i in range(1, 6)}).astype(float)
    df["grade"] = df["grade"].astype(str)
    df["year"] = df["issue"].dt.year
    df["vq"] = df["issue"].dt.to_period("Q")
    df["funded"] = df["funded_amnt"].astype(float)
    # leakage guard: a stopped payment can never precede origination
    assert (df["last_pymnt"].dropna() >= df.loc[df["last_pymnt"].notna(), "issue"]).all()
    return df


def treasury_3y(cfg) -> pd.Series:
    t = pd.read_parquet(cfg["treasury"])
    t["d"] = pd.to_datetime(t["report_date"])
    t["y3"] = pd.to_numeric(t["bc_3year"], errors="coerce")
    m = t.dropna(subset=["y3"]).set_index("d")["y3"].resample("MS").mean()
    return m  # monthly average 3-year Treasury yield, indexed by month start


# ----------------------------------------------------------------------------- 1. priced vs realised
def pool_table(df, keys, cfg, ust):
    n, b, r, rm = core.loan_cash_terms(df, cfg["servicing_fee"], cfg["recovery_lag_months"])
    g = df.groupby(keys, observed=True).ngroup().to_numpy()
    G = g.max() + 1
    D = core.pool_cashflows(g, G, df["funded"].to_numpy(), n, b, r, rm)
    lab = df.groupby(keys, observed=True).agg(
        loans=("funded", "size"), funded=("funded", "sum"))
    w = df["funded"].to_numpy()
    tmp = pd.DataFrame({"g": g, "w": w, "bad_w": w * df["bad"].to_numpy(), "rw": df["int_rate"].to_numpy(float) / 100 * w, "ust": df["issue"].map(ust).to_numpy() * w,
                        "loss": np.where(df["bad"], (df["funded"] - df["total_rec_prncp"] - (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0)).clip(lower=0), 0)})
    s = tmp.groupby("g")[["w", "bad_w", "rw", "ust", "loss"]].sum()
    lab = lab.reset_index()
    lab["rate"] = (s["rw"] / s["w"]).to_numpy()
    lab["default_rate"] = (s["bad_w"] / s["w"]).to_numpy()
    lab["net_loss_rate"] = (s["loss"] / s["w"]).to_numpy()
    lab["ust3"] = (s["ust"] / s["w"]).to_numpy()
    lab["irr"] = [core.irr_annual(D[i]) for i in range(G)]
    lab["excess"] = lab["irr"] - lab["ust3"]
    lab["shortfall"] = lab["rate"] - lab["irr"]
    return lab


# ----------------------------------------------------------------------------- 2. vintage curves + early read
def vintage_curves(df, cfg):
    snap = pd.Timestamp(cfg["snapshot"])
    lag = cfg["recognition_lag_months"]
    dm = core.default_month(df, df["bad"].to_numpy())
    rows, curves = [], {}
    for vq, idx in df.groupby("vq", observed=True).indices.items():
        start = vq.start_time
        last_issue = vq.end_time.to_period("M").start_time
        observed = (snap.year - last_issue.year) * 12 + (snap.month - last_issue.month) - lag  # MOB fully visible for every loan in the quarter
        c = core.cum_default_curve(df["funded"].to_numpy()[idx], dm[idx], 48)
        curves[str(vq)] = c
        rows.append({"vintage": str(vq), "start": start, "loans": len(idx), "funded": df["funded"].to_numpy()[idx].sum(),
                     "observed_mob": int(min(observed, 48)), "lifetime": c[-1] if last_issue <= pd.Timestamp(cfg["complete_through"]) else np.nan,
                     "mix_rate": np.average(df["int_rate"].to_numpy()[idx], weights=df["funded"].to_numpy()[idx]) / 100})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True), curves


def early_read(vt, curves, df, cfg):
    """Forecast each complete vintage's lifetime default rate using only information available at the time.

    Challenger (early read): cumulative default at month-on-book k times a development factor estimated from
    vintages that were fully seasoned at that date. Baselines: (a) the latest fully seasoned vintage's
    lifetime rate; (b) the vintage's sub-grade mix priced at the sub-grade lifetime default rates of fully
    seasoned vintages (what a pricing desk knows at origination).
    """
    lag = cfg["recognition_lag_months"]
    dm = core.default_month(df, df["bad"].to_numpy())
    df = df.assign(life_bad=np.where(df["bad"], df["funded"], 0.0))
    out = []
    ev = vt[(vt["start"] >= pd.Timestamp(cfg["early_read"]["evaluate_from"])) & vt["lifetime"].notna()]
    for _, v in ev.iterrows():
        rec = {"vintage": v["vintage"], "actual": v["lifetime"]}
        orig_end = pd.Period(v["vintage"], "Q").end_time.to_period("M").start_time
        for k in [0] + cfg["early_read"]["mobs"]:
            asof = orig_end + pd.DateOffset(months=k)  # date the forecast is made
            # reference vintages fully seasoned by `asof`: last issue month + 36 + lag <= asof
            ref = vt[vt["lifetime"].notna() & vt["vintage"].map(lambda q: pd.Period(q, "Q").end_time.to_period("M").start_time + pd.DateOffset(months=36 + lag) <= asof)]
            if len(ref) == 0:
                continue
            if k == 0:
                rec["last_seasoned"] = ref.iloc[-1]["lifetime"]
                sub = df[df["vq"].astype(str).isin(ref["vintage"])]
                sg_rate = sub.groupby("sg").apply(lambda s: s["life_bad"].sum() / s["funded"].sum())
                cur = df[df["vq"].astype(str) == v["vintage"]]
                rec["grade_mix"] = float(np.sum(cur.groupby("sg")["funded"].sum() * sg_rate.reindex(cur.groupby("sg")["funded"].sum().index).fillna(sg_rate.max())) / cur["funded"].sum())
            else:
                ck = np.array([curves[q][k - 1] for q in ref["vintage"]])
                w = ref["funded"].to_numpy()
                ldf = np.sum(w * ref["lifetime"].to_numpy()) / np.sum(w * ck)
                rec[f"read_{k}"] = curves[v["vintage"]][k - 1] * ldf
                rec[f"ldf_{k}"] = ldf
        out.append(rec)
    return pd.DataFrame(out)


def project_open_vintages(vt, curves, cfg):
    """Project lifetime default for vintages still running at the data cut, using all fully seasoned vintages."""
    ref = vt[vt["lifetime"].notna() & (vt["start"] >= pd.Timestamp("2012-01-01"))]
    rows = []
    for _, v in vt[vt["lifetime"].isna()].iterrows():
        k = int(v["observed_mob"])
        if k < 6:
            continue
        ck = np.array([curves[q][k - 1] for q in ref["vintage"]])
        ldf = np.sum(ref["funded"] * ref["lifetime"]) / np.sum(ref["funded"].to_numpy() * ck)
        rows.append({"vintage": v["vintage"], "observed_mob": k, "observed": curves[v["vintage"]][k - 1],
                     "projected_lifetime": curves[v["vintage"]][k - 1] * ldf, "mix_rate": v["mix_rate"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. out-of-time screen
def features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["sg_code"] = df["sg_code"]
    X["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    X["dti"] = df["dti"].clip(0, 60)
    X["log_inc"] = np.log1p(df["annual_inc"].clip(0, 2e6))
    X["loan_to_inc"] = df["loan_amnt"] / df["annual_inc"].clip(lower=1000)
    X["pay_to_inc"] = df["installment"] * 12 / df["annual_inc"].clip(lower=1000)
    X["loan_amnt"] = df["loan_amnt"]
    X["revol_util"] = df["revol_util"].clip(0, 150)
    X["log_revol_bal"] = np.log1p(df["revol_bal"].clip(lower=0))
    X["inq_6m"] = df["inq_last_6mths"]
    X["delinq_2y"] = df["delinq_2yrs"]
    X["open_acc"] = df["open_acc"]
    X["total_acc"] = df["total_acc"]
    X["pub_rec"] = df["pub_rec"]
    X["bankruptcies"] = df["pub_rec_bankruptcies"]
    X["mort_acc"] = df["mort_acc"]
    ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    X["hist_years"] = (df["issue"] - ecl).dt.days / 365.25
    X["emp_years"] = df["emp_length"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    for c in ["home_ownership", "verification_status", "purpose"]:
        X[c] = df[c].astype(str).astype("category")
    return X


SOURCE_FIELDS = {  # each model feature -> the application fields it is built from (checked in tests)
    "sg_code": {"sub_grade"}, "fico": {"fico_range_low", "fico_range_high"}, "dti": {"dti"}, "log_inc": {"annual_inc"},
    "loan_to_inc": {"loan_amnt", "annual_inc"}, "pay_to_inc": {"installment", "annual_inc"}, "loan_amnt": {"loan_amnt"},
    "revol_util": {"revol_util"}, "log_revol_bal": {"revol_bal"}, "inq_6m": {"inq_last_6mths"}, "delinq_2y": {"delinq_2yrs"},
    "open_acc": {"open_acc"}, "total_acc": {"total_acc"}, "pub_rec": {"pub_rec"}, "bankruptcies": {"pub_rec_bankruptcies"},
    "mort_acc": {"mort_acc"}, "hist_years": {"earliest_cr_line", "issue_d"}, "emp_years": {"emp_length"},
    "home_ownership": {"home_ownership"}, "verification_status": {"verification_status"}, "purpose": {"purpose"},
}


def screen(df, cfg, ust):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    sc = cfg["screen"]
    span = lambda a, b: df[(df["issue"] >= a) & (df["issue"] <= b)]
    tr, va, te = span(*sc["train"]), span(*sc["valid"]), span(*sc["test"])
    Xtr, Xva, Xte = features(tr), features(va), features(te)
    cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    for c in cats:  # categories fixed from the training window only
        levels = Xtr[c].cat.categories
        for X in (Xva, Xte):  # levels unseen in training become missing
            v = X[c].astype(str)
            X[c] = pd.Categorical(v.where(v.isin(levels)), categories=levels)
    ytr, yva, yte = tr["bad"].to_numpy(), va["bad"].to_numpy(), te["bad"].to_numpy()

    # baseline 1: the platform's own price (sub-grade only)
    oh = lambda d: pd.get_dummies(d["sg"]).reindex(columns=sorted(df["sg"].unique()), fill_value=0).to_numpy(float)
    lr_g = LogisticRegression(max_iter=2000).fit(oh(tr), ytr)
    p_grade = lr_g.predict_proba(oh(te))[:, 1]

    # baseline 2: linear scorecard on the same application fields
    num = [c for c in Xtr.columns if c not in cats]
    med = Xtr[num].median()
    mu, sd = Xtr[num].fillna(med).mean(), Xtr[num].fillna(med).std().replace(0, 1)
    lin = lambda X: np.hstack([((X[num].fillna(med) - mu) / sd).to_numpy(), pd.get_dummies(X[cats]).reindex(columns=pd.get_dummies(Xtr[cats]).columns, fill_value=0).to_numpy(float)])
    lr_s = LogisticRegression(max_iter=3000, C=1.0).fit(lin(Xtr), ytr)
    p_lin = lr_s.predict_proba(lin(Xte))[:, 1]

    # challenger: gradient boosting, early-stopped on the 2014 validation year only
    p = dict(sc["lgbm"]); nest = p.pop("n_estimators"); esr = p.pop("early_stopping_rounds")
    params = dict(objective="binary", metric="auc", seed=sc["seed"], deterministic=True, force_row_wise=True,
                  num_threads=2, verbose=-1, **p)
    t0 = time.time()
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, categorical_feature=cats, reference=dtr)
    m = lgb.train(params, dtr, nest, valid_sets=[dva], callbacks=[lgb.early_stopping(esr, verbose=False)])
    p_gbm = m.predict(Xte, num_iteration=m.best_iteration)
    p_gbm_tr = m.predict(Xtr, num_iteration=m.best_iteration)
    train_secs = time.time() - t0

    def within(score):
        a = te.assign(s=score, y=yte).groupby("sg").apply(lambda d: (core.auc(d["y"], d["s"]), len(d)))
        v = np.array([x[0] for x in a]); w = np.array([x[1] for x in a]); ok = ~np.isnan(v)
        return float(np.average(v[ok], weights=w[ok]))

    res = {"n_train": len(tr), "n_valid": len(va), "n_test": len(te),
           "bad_rate_train": float(ytr.mean()), "bad_rate_test": float(yte.mean()),
           "best_iteration": int(m.best_iteration), "train_seconds": round(train_secs, 1),
           "auc": {"grade_only": core.auc(yte, p_grade), "linear_scorecard": core.auc(yte, p_lin), "gbm": core.auc(yte, p_gbm)},
           "auc_within_subgrade": {"grade_only": 0.5, "linear_scorecard": within(p_lin), "gbm": within(p_gbm)},
           "brier": {"grade_only": float(np.mean((p_grade - yte) ** 2)), "linear_scorecard": float(np.mean((p_lin - yte) ** 2)), "gbm": float(np.mean((p_gbm - yte) ** 2))},
           "psi_score_train_vs_test": core.psi(p_gbm_tr, p_gbm),
           "mean_pd_vs_actual_test": {"gbm": float(p_gbm.mean()), "linear_scorecard": float(p_lin.mean()), "grade_only": float(p_grade.mean()), "actual": float(yte.mean())}}
    imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    res["top_features_gain_share"] = (imp / imp.sum()).head(8).round(3).to_dict()

    # calibration by decile of the challenger score
    q = pd.qcut(p_gbm, 10, labels=False, duplicates="drop")
    res["calibration"] = pd.DataFrame({"q": q, "p": p_gbm, "y": yte}).groupby("q").agg(pred=("p", "mean"), actual=("y", "mean")).round(4).reset_index().to_dict("records")

    # business test: within each sub-grade, skip the riskiest share by score; compare pool returns on the 2015 vintage
    te = te.assign(gbm=p_gbm, lin=p_lin, rnd=np.random.default_rng(sc["seed"]).random(len(te)))
    biz = []
    for share in sc["exclude_shares"]:
        for name in ["gbm", "lin", "rnd"]:
            cut = te.groupby("sg")[name].transform(lambda s: s.rank(pct=True, method="first") > 1 - share)
            kept = te[~cut]
            a = pool_table(kept.assign(_all=1), ["grade"], cfg, ust).set_index("grade")
            tot = pool_table(kept.assign(_all=1), ["_all"], cfg, ust).iloc[0]
            biz.append({"share": share, "screen": name, "irr_all": tot["irr"], "loss_all": tot["net_loss_rate"], "default_all": tot["default_rate"],
                        "funded_kept": float(kept["funded"].sum()), **{f"irr_{g}": a.loc[g, "irr"] for g in GRADES if g in a.index}})
    base_g = pool_table(te.assign(_all=1), ["grade"], cfg, ust).set_index("grade")
    base = pool_table(te.assign(_all=1), ["_all"], cfg, ust).iloc[0]
    biz.append({"share": 0.0, "screen": "none", "irr_all": base["irr"], "loss_all": base["net_loss_rate"], "default_all": base["default_rate"],
                "funded_kept": float(te["funded"].sum()), **{f"irr_{g}": base_g.loc[g, "irr"] for g in GRADES}})
    # bootstrap (loans resampled within sub-grade) for the 20% model screen vs buying everything
    rng = np.random.default_rng(sc["seed"])
    cut20 = te.groupby("sg")["gbm"].transform(lambda s: s.rank(pct=True, method="first") > 0.8).to_numpy()
    diffs = []
    for _ in range(sc.get("bootstrap", 100)):
        idx = np.concatenate([rng.choice(ix, len(ix)) for ix in te.groupby("sg").indices.values()])
        b = te.iloc[idx].assign(_all=1)
        k = b[~cut20[idx]]
        diffs.append(pool_table(k, ["_all"], cfg, ust)["irr"].iloc[0] - pool_table(b, ["_all"], cfg, ust)["irr"].iloc[0])
    res["uplift_20pct_bootstrap"] = {"mean": float(np.mean(diffs)), "p05": float(np.quantile(diffs, 0.05)), "p95": float(np.quantile(diffs, 0.95)),
                                     "share_positive": float(np.mean(np.array(diffs) > 0)), "reps": len(diffs)}
    return res, pd.DataFrame(biz), te[["sg", "grade", "bad", "gbm", "lin"]].assign(grade_pd=p_grade)


def robustness(comp, cfg, ust):
    """Grade returns under alternative cash-flow assumptions (recovery timing, servicing fee)."""
    rows = {}
    for label, over in {"base": {}, "recoveries_after_12m": {"recovery_lag_months": 12}, "recoveries_after_3m": {"recovery_lag_months": 3},
                        "no_servicing_fee": {"servicing_fee": 0.0}}.items():
        c = {**cfg, **over}
        rows[label] = pool_table(comp, ["grade"], c, ust).set_index("grade")["irr"].round(4).to_dict()
    # subperiods: grade returns, vintages 2010-2013 vs 2014-2015
    rows["vintages_2010_2013"] = pool_table(comp[(comp["year"] >= 2010) & (comp["year"] <= 2013)], ["grade"], cfg, ust).set_index("grade")["irr"].round(4).to_dict()
    rows["vintages_2014_2015"] = pool_table(comp[comp["year"] >= 2014], ["grade"], cfg, ust).set_index("grade")["irr"].round(4).to_dict()
    return rows


def business(er, vt, allp, sres, biz, cfg):
    """Dollar translation per $1bn of purchases (illustrative; derivations in the README)."""
    late = er[er["vintage"] >= "2014Q4"].copy()
    w = late["vintage"].map(vt.set_index("vintage")["funded"]).to_numpy()
    sev = float(allp["net_loss_rate"].iloc[0] / allp["default_rate"].iloc[0])  # net loss per $ of defaulted principal
    gap = float(np.average(late["actual"] - late["grade_mix"], weights=w))
    base = biz[biz["screen"] == "none"].iloc[0]
    g20 = biz[(biz["screen"] == "gbm") & (biz["share"] == 0.2)].iloc[0]
    book = cfg["book_usd"]
    return {
        "late_vintages": "2014Q4-2015Q4", "late_vintages_funded_usd": float(w.sum()),
        "expected_at_issue": float(np.average(late["grade_mix"], weights=w)), "actual": float(np.average(late["actual"], weights=w)),
        "read_12": float(np.average(late["read_12"], weights=w)), "read_18": float(np.average(late["read_18"], weights=w)),
        "default_gap": gap, "severity": sev, "extra_net_loss_per_bn_usd": gap * sev * book,
        "screen20_irr_uplift": float(g20["irr_all"] - base["irr_all"]), "screen20_uplift_per_bn_usd_per_year": float((g20["irr_all"] - base["irr_all"]) * book),
        "screen20_net_loss_change": float(g20["loss_all"] - base["loss_all"]),
    }


def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_dir"]); (out / "figures").mkdir(parents=True, exist_ok=True)
    df = prepare(cfg)
    ust = treasury_3y(cfg)
    comp = df[df["issue"] <= pd.Timestamp(cfg["complete_through"])]

    sg = pool_table(comp, ["sg"], cfg, ust)
    gr = pool_table(comp, ["grade"], cfg, ust)
    gy = pool_table(comp, ["grade", "year"], cfg, ust)
    allp = pool_table(comp.assign(_all=1), ["_all"], cfg, ust)
    vt, curves = vintage_curves(df, cfg)
    er = early_read(vt, curves, df, cfg)
    proj = project_open_vintages(vt, curves, cfg)
    sres, biz, scores = screen(df, cfg, ust)
    rob = robustness(comp, cfg, ust)

    for name, t in [("subgrade_returns", sg), ("grade_returns", gr), ("grade_year_returns", gy), ("vintages", vt),
                    ("early_read", er), ("open_vintage_projection", proj), ("screen_business", biz)]:
        t.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "vintage_curves.csv", index_label="mob_minus_1")
    scores.to_pickle("data/test_scores.pkl")

    mae = {c: float(np.mean(np.abs(er[c] - er["actual"]))) for c in er.columns if c.startswith(("read_", "last_", "grade_mix"))}
    metrics = {
        "sample": {"loans_36m_all": int(len(df)), "loans_36m_complete": int(len(comp)), "funded_complete_usd": float(comp["funded"].sum()),
                   "complete_vintages": f"{comp['issue'].min():%Y-%m} to {comp['issue'].max():%Y-%m}"},
        "pool_complete": allp.drop(columns="_all").iloc[0].to_dict(),
        "by_grade": gr.set_index("grade").to_dict("index"),
        "early_read_mae": mae, "early_read_n": int(len(er)),
        "screen": sres,
        "robustness_grade_irr": rob,
        "business": business(er, vt, allp, sres, biz, cfg),
        "open_vintages": proj.round(4).to_dict("records"),
    }
    from loanprice import figures
    figures.make_all(out, sg, gy, curves, vt, er, proj, biz, sres["calibration"])
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    print(json.dumps(metrics, indent=1, default=float)[:6000])
    return df, comp, sg, gr, gy, vt, curves, er, proj, sres, biz


if __name__ == "__main__":
    main(*sys.argv[1:2])
