"""Pure functions: loan cash flows, pool IRR, vintage default curves, early-read forecasts, scoring metrics."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# Fields known when the loan is approved. Anything else (payments, recoveries, last FICO pull,
# hardship or settlement flags) is observed after origination and must never enter a model.
APPLICATION_FIELDS = {
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies", "application_type",
    "initial_list_status",
}

H = 84  # months of cash-flow grid (36-month term + late recoveries, with room for 60-month loans)


def months_between(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Whole months from a to b (both month-start timestamps)."""
    return ((b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)).to_numpy()


def loan_cash_terms(df: pd.DataFrame, fee: float, rec_lag: int):
    """Per-loan inputs for the cash-flow grid.

    Borrower payments (excluding recoveries) are spread evenly from month 1 to the month of the last
    payment, less the servicing fee. Net recoveries (after collection fees) arrive rec_lag months after
    the last payment. The month-level detail of prepayments is not in the file, so even spreading is an
    approximation; it is applied identically to every group compared.
    """
    n = months_between(df["issue"], df["last_pymnt"])
    n = np.where(df["last_pymnt"].isna(), 0, np.clip(n, 1, H - rec_lag - 2)).astype(int)
    borrower = (df["total_pymnt"] - df["recoveries"]).to_numpy(float) * (1 - fee)
    borrower = np.where(n == 0, 0.0, borrower)
    rec = (df["recoveries"] - df["collection_recovery_fee"]).clip(lower=0).to_numpy(float)
    rec_m = np.clip(np.maximum(n, 1) + rec_lag, 1, H - 1)
    return n, borrower, rec, rec_m


def pool_cashflows(groups: np.ndarray, G: int, funded, n, borrower, rec, rec_m) -> np.ndarray:
    """Aggregate monthly investor cash flows for G groups (row g = group g). Column 0 is the purchase."""
    D = np.zeros((G, H + 1))
    np.add.at(D, (groups, 0), -np.asarray(funded, float))
    pay = n > 0
    per = np.zeros_like(borrower)
    per[pay] = borrower[pay] / n[pay]
    # difference array: +per at month 1, -per at month n+1, then cumulative sum over months >= 1
    S = np.zeros((G, H + 2))
    np.add.at(S, (groups[pay], 1), per[pay])
    np.add.at(S, (groups[pay], n[pay] + 1), -per[pay])
    D[:, 1:] += np.cumsum(S, axis=1)[:, 1:H + 1]
    np.add.at(D, (groups, rec_m), rec)
    D[np.abs(D) < 0.01] = 0.0  # remove sub-cent floating residue left by the cumulative sum
    return D


def irr_annual(cf: np.ndarray) -> float:
    """Annualised IRR of a monthly cash-flow vector (cf[0] < 0)."""
    t = np.arange(len(cf))
    f = lambda r: np.sum(cf / (1 + r) ** t)
    if f(-0.2) * f(0.2) > 0:
        return np.nan
    r = brentq(f, -0.2, 0.2, xtol=1e-12)
    return (1 + r) ** 12 - 1


def default_month(df: pd.DataFrame, bad: np.ndarray) -> np.ndarray:
    """Month on book in which a bad loan stopped paying (first missed payment); NaN for good loans."""
    n = months_between(df["issue"], df["last_pymnt"])
    n = np.where(df["last_pymnt"].isna(), 0, np.maximum(n, 0))
    return np.where(bad, n + 1, np.nan)


def cum_default_curve(funded: np.ndarray, dmonth: np.ndarray, max_mob: int) -> np.ndarray:
    """Dollar-weighted cumulative share of the vintage that has stopped paying by each month on book (1..max_mob)."""
    tot = funded.sum()
    out = np.zeros(max_mob)
    m = ~np.isnan(dmonth)
    if tot <= 0:
        return out
    h = np.bincount(np.clip(dmonth[m].astype(int), 1, max_mob + 200), weights=funded[m], minlength=max_mob + 201)
    return np.cumsum(h)[1:max_mob + 1] / tot


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank AUC (probability a random bad scores above a random good), ties counted half."""
    y = np.asarray(y).astype(bool)
    r = pd.Series(s).rank(method="average").to_numpy()
    nb, ng = y.sum(), (~y).sum()
    if nb == 0 or ng == 0:
        return np.nan
    return (r[y].sum() - nb * (nb + 1) / 2) / (nb * ng)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index of actual vs expected, on expected-decile bins."""
    q = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    e = np.histogram(expected, q)[0] / len(expected)
    a = np.histogram(actual, q)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
