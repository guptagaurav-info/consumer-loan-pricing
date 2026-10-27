"""Charts for the write-up. Every chart is drawn from the CSVs the pipeline writes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

NAVY, GOLD, GREY, RED, TEAL = "#1f3a5f", "#c9a227", "#8a8f98", "#b23a3a", "#2a7f7f"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12.5, "figure.dpi": 150, "savefig.bbox": "tight"})
PCT = PercentFormatter(1.0, decimals=0)


def priced_vs_realised(sg: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(sg))
    ax.bar(x, sg["rate"], color="#d9dee6", label="Promised interest rate (funded-weighted)")
    ax.bar(x, sg["irr"].clip(lower=0), color=NAVY, label="Realised annual return to the loan buyer")
    neg = (sg["irr"] < 0).to_numpy()
    ax.bar(x[neg], sg["irr"][neg], color=RED, label="Realised return below zero")
    thin = (sg["loans"] < 1000).to_numpy()
    if thin.any():
        ax.axvspan(x[thin].min() - 0.5, x[thin].max() + 0.5, color=GOLD, alpha=0.10, lw=0)
        ax.text(x[thin].mean(), ax.get_ylim()[1] * 0.97, "Fewer than 1,000\nloans each: noisy", ha="center", va="top", fontsize=8.5, color="#6b5a12")
    ax.set_xticks(x, sg["sg"], rotation=90, fontsize=8.5)
    ax.yaxis.set_major_formatter(PCT)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Per year")
    ax.set_title("The coupon rose with risk; the return did not\n36-month loans issued 2007–2015, all fully run off")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.savefig(path); plt.close(fig)


def grade_by_year(gy: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    piv = gy[gy["year"] >= 2010].pivot(index="year", columns="grade", values="irr")
    colors = dict(A="#9db4cf", B="#5d82ad", C=NAVY, D=GOLD, E="#d27d2d", F=RED, G="#6b1e1e")
    for g in ["A", "B", "C", "D", "E", "F", "G"]:
        thin = g == "G"
        ax.plot(piv.index, piv[g], marker="o", lw=1.0 if thin else (2.2 if g in "AE" else 1.4), ls=":" if thin else "-",
                alpha=0.6 if thin else 1, color=colors[g], label="Grade G (thin, < 300 loans a year)" if thin else f"Grade {g}")
    ax.axhline(0, color="black", lw=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Year the loans were issued (each vintage followed to the end of its 36-month term)")
    ax.set_ylabel("Realised annual return")
    ax.set_title("The risk premium in the lowest grades disappeared by the 2015 vintage")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="lower left")
    fig.savefig(path); plt.close(fig)


def vintage_curves(curves: pd.DataFrame, vt: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    pick = {"2012Q1": GREY, "2013Q1": "#9db4cf", "2014Q1": "#5d82ad", "2015Q1": NAVY, "2016Q1": GOLD, "2017Q1": "#d27d2d", "2018Q1": RED}
    for q, c in pick.items():
        k = int(vt.set_index("vintage").loc[q, "observed_mob"])
        y = curves[q].to_numpy()[:k]
        ax.plot(np.arange(1, k + 1), y, color=c, lw=2, label=q)
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Months since the loans were issued")
    ax.set_ylabel("Share of the vintage ($) that has stopped paying")
    ax.set_title("From 2014 on, each new vintage stopped paying faster than the last\nCumulative defaults by month on book, first-quarter vintages")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.savefig(path); plt.close(fig)


def early_read_error(er: pd.DataFrame, path: Path):
    labels = ["Latest fully\nrun-off vintage", "Grade mix at\nhistorical rates", "Month-6\nread", "Month-12\nread", "Month-18\nread", "Month-24\nread"]
    cols = ["last_seasoned", "grade_mix", "read_6", "read_12", "read_18", "read_24"]
    mae = [np.mean(np.abs(er[c] - er["actual"])) for c in cols]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    b = ax.bar(labels, mae, color=[GREY, GREY, "#9db4cf", "#5d82ad", NAVY, NAVY])
    for r, v in zip(b, mae):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.0008, f"{v * 100:.1f} pts", ha="center", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylabel("Average miss on lifetime default rate")
    ax.set_title("By month 18 the vintage's final loss is known within a point\n16 quarterly vintages, 2012–2015, each forecast with data available at the time")
    fig.savefig(path); plt.close(fig)


def forecast_vs_actual(er: pd.DataFrame, proj: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(er))
    ax.plot(x, er["actual"], color="black", lw=2.4, marker="o", label="Actual lifetime default rate")
    ax.plot(x, er["grade_mix"], color=GREY, lw=1.6, ls="--", label="Expected at issue (grade mix × history)")
    ax.plot(x, er["read_18"], color=NAVY, lw=1.6, marker="s", ms=4, label="Month-18 read")
    x2 = np.arange(len(er), len(er) + len(proj))
    ax.plot(x2, proj["projected_lifetime"], color=RED, lw=2, marker="o", ls=":", label="Projected (vintages still running at the data cut)")
    ax.set_xticks(np.r_[x, x2], list(er["vintage"]) + list(proj["vintage"]), rotation=90, fontsize=8.5)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Share of the vintage ($) that defaults")
    ax.set_title("History said 11%; the 2014–2015 vintages delivered 14–15%")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(path); plt.close(fig)


def screen_uplift(biz: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    base = biz[biz["screen"] == "none"]["irr_all"].iloc[0]
    names = {"rnd": "Random skip (control)", "lin": "Linear scorecard", "gbm": "Gradient-boosted model"}
    w = 0.26
    for i, s in enumerate(["rnd", "lin", "gbm"]):
        d = biz[biz["screen"] == s].sort_values("share")
        ax.bar(np.arange(len(d)) + (i - 1) * w, (d["irr_all"] - base) * 1e4, w, color=[GREY, GOLD, NAVY][i], label=names[s])
    ax.set_xticks(range(2), ["Skip riskiest 10% in each sub-grade", "Skip riskiest 20% in each sub-grade"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Change in realised return (basis points a year)")
    ax.set_title(f"Screening inside each grade added return out of sample\n2015 vintage (model trained on 2010–2013); buy-everything return {base * 100:.2f}% a year")
    ax.legend(frameon=False, fontsize=9.5)
    fig.savefig(path); plt.close(fig)


def calibration(cal: list, path: Path):
    c = pd.DataFrame(cal)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    m = max(c["pred"].max(), c["actual"].max()) * 1.08
    ax.plot([0, m], [0, m], color=GREY, ls="--", lw=1, label="Perfect calibration")
    ax.plot(c["pred"], c["actual"], color=NAVY, marker="o", lw=2, label="Model, 2015 vintage")
    ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("Predicted default rate (score decile)")
    ax.set_ylabel("Actual default rate")
    ax.set_title("Ranks well, but under-predicts the later vintage")
    ax.legend(frameon=False, fontsize=9.5)
    fig.savefig(path); plt.close(fig)


def make_all(out: Path, sg, gy, curves, vt, er, proj, biz, cal):
    f = out / "figures"; f.mkdir(parents=True, exist_ok=True)
    priced_vs_realised(sg, f / "priced_vs_realised.png")
    grade_by_year(gy, f / "grade_by_year.png")
    vintage_curves(pd.DataFrame(curves), vt, f / "vintage_curves.png")
    early_read_error(er, f / "early_read_error.png")
    forecast_vs_actual(er, proj, f / "forecast_vs_actual.png")
    screen_uplift(biz, f / "screen_uplift.png")
    calibration(cal, f / "calibration.png")
