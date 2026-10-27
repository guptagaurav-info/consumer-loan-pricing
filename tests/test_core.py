import numpy as np
import pandas as pd

from loanprice import core, pipeline


def _loan(n_paid, total, rec=0.0, fee=0.0, funded=1000.0):
    issue = pd.Timestamp("2014-01-01")
    lp = issue + pd.DateOffset(months=n_paid) if n_paid else pd.NaT
    return pd.DataFrame({"issue": [issue], "last_pymnt": [lp], "total_pymnt": [total + rec], "recoveries": [rec],
                         "collection_recovery_fee": [fee], "funded": [funded]})


def test_level_loan_irr_equals_coupon():
    r = 0.01  # 1% a month
    pmt = 1000 * r / (1 - (1 + r) ** -36)
    d = _loan(36, pmt * 36)
    n, b, rec, rm = core.loan_cash_terms(d, fee=0.0, rec_lag=6)
    D = core.pool_cashflows(np.array([0]), 1, d["funded"], n, b, rec, rm)
    assert abs(core.irr_annual(D[0]) - ((1 + r) ** 12 - 1)) < 1e-6


def test_cash_is_conserved_and_fee_applied():
    d = pd.concat([_loan(10, 400.0, rec=150.0, fee=20.0), _loan(0, 0.0, rec=50.0)], ignore_index=True)
    n, b, rec, rm = core.loan_cash_terms(d, fee=0.01, rec_lag=6)
    D = core.pool_cashflows(np.array([0, 0]), 1, d["funded"], n, b, rec, rm)
    assert abs(D[0, 1:].sum() - (400 * 0.99 + 130 + 50)) < 1e-6
    assert D[0, 0] == -2000


def test_default_curve_monotone_and_complete():
    d = pd.concat([_loan(5, 100.0), _loan(36, 1200.0), _loan(0, 0.0)], ignore_index=True)
    bad = np.array([True, False, True])
    dm = core.default_month(d, bad)
    assert list(dm[[0, 2]]) == [6, 1] and np.isnan(dm[1])
    c = core.cum_default_curve(d["funded"].to_numpy(), dm, 48)
    assert np.all(np.diff(c) >= 0) and abs(c[-1] - 2 / 3) < 1e-12


def test_auc_and_psi():
    assert core.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert core.auc([0, 1], [0.5, 0.5]) == 0.5
    x = np.random.default_rng(0).normal(size=5000)
    assert core.psi(x, x) < 1e-9


def test_model_uses_application_fields_only():
    used = set().union(*pipeline.SOURCE_FIELDS.values())
    assert used <= core.APPLICATION_FIELDS
    forbidden = {"total_pymnt", "recoveries", "last_pymnt_d", "loan_status", "total_rec_prncp", "out_prncp"}
    assert not used & forbidden


def test_feature_columns_match_declared_sources():
    row = {"sub_grade": "B3", "sg_code": 8.0, "fico_range_low": 700, "fico_range_high": 704, "dti": 15.0, "annual_inc": 60000.0,
           "loan_amnt": 10000.0, "installment": 330.0, "revol_util": 40.0, "revol_bal": 8000.0, "inq_last_6mths": 1,
           "delinq_2yrs": 0, "open_acc": 9, "total_acc": 20, "pub_rec": 0, "pub_rec_bankruptcies": 0, "mort_acc": 1,
           "earliest_cr_line": "Jan-2005", "issue": pd.Timestamp("2014-01-01"), "emp_length": "5 years",
           "home_ownership": "RENT", "verification_status": "Verified", "purpose": "debt_consolidation"}
    X = pipeline.features(pd.DataFrame([row]))
    assert set(X.columns) == set(pipeline.SOURCE_FIELDS)
    assert abs(X["hist_years"].iloc[0] - 9.0) < 0.01 and X["emp_years"].iloc[0] == 5
