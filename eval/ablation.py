"""
What we tried: compares model variants on the same temporal holdout as evaluate.py.

Run:  python -m eval.ablation

Caveat: all variants are scored on the same 41 held-out posts, so picking the
winner here is mild selection on the test set. The chosen setup (small, heavily
regularised trees + volatility-normalised ranges) was picked on principle (tiny
dataset -> strong regularisation; pages differ in volatility -> normalise), and
this table is here to show the alternatives weren't better, not to tune.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimator.data import load_posts  # noqa: E402
from estimator.model import feature_frame, make_regressor, rolling_baseline, rolling_sigma  # noqa: E402
from eval.evaluate import temporal_split  # noqa: E402

VARIANTS = {
    "naive (zero uplift)": None,
    "HGB depth3, 150 iters": HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=8, l2_regularization=1.0, random_state=0
    ),
    "HGB depth2, 60 iters (shipped)": make_regressor(),
    "Ridge alpha=10": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "Ridge alpha=50": make_pipeline(StandardScaler(), Ridge(alpha=50)),
}


def main() -> None:
    df = load_posts()
    bl, bc, sg = rolling_baseline(df, "likes"), rolling_baseline(df, "comments"), rolling_sigma(df, "likes")
    train, test = temporal_split(df)
    tr = train.index[bl[train.index].notna()]
    X_tr, X_te = feature_frame(df.loc[tr], bl, bc), feature_frame(test, bl, bc)
    y_tr = np.log1p(df.loc[tr, "likes"]) - np.log1p(bl[tr])
    actual = test["likes"].to_numpy()
    cv = KFold(5, shuffle=True, random_state=0)

    print("| variant | range scaling | MAE | median % error | in 80% range | Ubiquiti range width | TP-Link DE in range |")
    print("|---|---|---|---|---|---|---|")
    for name, est in VARIANTS.items():
        if est is None:
            oof, p_te = np.zeros(len(y_tr)), np.zeros(len(test))
        else:
            oof = cross_val_predict(est, X_tr, y_tr, cv=cv)
            p_te = est.fit(X_tr, y_tr).predict(X_te)
        pl = p_te + np.log1p(bl[test.index].to_numpy())
        pred = np.expm1(pl)
        for norm in (False, True):
            r = (y_tr - oof) / (sg[tr] if norm else 1)
            lo_q, hi_q = np.quantile(r, [0.1, 0.9])
            s = sg[test.index].to_numpy() if norm else 1.0
            lo, hi = np.floor(np.expm1(pl + lo_q * s)), np.ceil(np.expm1(pl + hi_q * s))
            inr = (actual >= lo) & (actual <= hi)
            ape = np.abs(pred - actual) / np.maximum(actual, 1)
            ubi = (test["company"] == "ubiquiti").to_numpy()
            tpde = (test["company"] == "tp-link-deutschland").to_numpy()
            print(
                f"| {name} | {'per-page volatility' if norm else 'global'} | {np.mean(np.abs(pred - actual)):.1f} | "
                f"{np.median(ape):.0%} | {inr.mean():.0%} | {np.median((hi[ubi] + 1) / (lo[ubi] + 1)):.1f}x | {inr[tpde].mean():.0%} |"
            )


if __name__ == "__main__":
    main()
