"""
Accuracy evaluation.

Run:  python -m eval.evaluate

Two tests, both on posts the model never saw during training:

1. Temporal holdout (the realistic one). For every page, the most recent 25% of
   posts are held out. The model is trained on the older 75%. Each test post's
   baseline is the median of the 8 posts the page published before it. This mimics "predict the next
   post for a page we already track". Ubiquiti and TP-Link Deutschland (the two
   reference pages) are reported individually.

2. Leave-one-page-out cold start (the stress test). Train on 3 pages, predict the
   4th knowing only its follower count. This shows what happens for a page the
   model has no history for.

Every result is compared with a naive baseline: "predict the median of the page's
last 8 posts", with a range built the same conformal way from the training data.
If the model can't beat that, it isn't adding anything and we say so.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from estimator.data import load_posts  # noqa: E402
from estimator.model import INTERVAL_HIGH_Q, INTERVAL_LOW_Q, EngagementModel, feature_frame, rolling_baseline, rolling_sigma  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
HOLDOUT_FRAC = 0.25
REFERENCE_PAGES = ["ubiquiti", "tp-link-deutschland"]


def temporal_split(df: pd.DataFrame):
    train_idx, test_idx = [], []
    for _, g in df.sort_values("posted_at").groupby("company"):
        n_test = max(1, int(round(len(g) * HOLDOUT_FRAC)))
        train_idx += list(g.index[:-n_test])
        test_idx += list(g.index[-n_test:])
    return df.loc[train_idx], df.loc[test_idx]


def predict_logs(model: EngagementModel, test: pd.DataFrame, base_l: pd.Series, base_c: pd.Series):
    X = feature_frame(test, base_l, base_c)
    pl = model.likes_model.predict(X) + np.log1p(base_l.to_numpy())
    pc = model.comments_model.predict(X) + np.log1p(base_c.to_numpy())
    return pl, pc


def naive_quantiles(train: pd.DataFrame):
    """Same normalised conformal recipe as the model, but with a zero-uplift 'model'."""
    base, sig = rolling_baseline(train, "likes"), rolling_sigma(train, "likes")
    ok = base.notna()
    r = (np.log1p(train.loc[ok, "likes"]) - np.log1p(base[ok])) / sig[ok]
    return float(np.quantile(r, INTERVAL_LOW_Q)), float(np.quantile(r, INTERVAL_HIGH_Q))


def metrics(actual: np.ndarray, pred_log: np.ndarray, q: tuple[float, float], sigma) -> dict:
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), pred_log.shape)
    pred = np.expm1(pred_log)
    lo, hi = np.expm1(pred_log + q[0] * sigma), np.expm1(pred_log + q[1] * sigma)
    ape = np.abs(pred - actual) / np.maximum(actual, 1)
    return {
        "n": len(actual),
        "MAE": float(np.mean(np.abs(pred - actual))),
        "MdAPE": float(np.median(ape)),
        "within_30pct": float(np.mean(ape <= 0.30)),
        "range_coverage": float(np.mean((actual >= np.floor(lo)) & (actual <= np.ceil(hi)))),
        "range_width_x": float(np.median((hi + 1) / (lo + 1))),
    }


def fmt_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['n']} | {m['MAE']:.1f} | {m['MdAPE']:.0%} | {m['within_30pct']:.0%} | "
        f"{m['range_coverage']:.0%} | {m['range_width_x']:.2f}x |"
    )


HEADER = (
    "| | n | MAE | median % error | within ±30% | in 80% range | range high/low |\n"
    "|---|---|---|---|---|---|---|"
)


def temporal_holdout(df: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    train, test = temporal_split(df)
    model = EngagementModel.fit(train)
    # Walk-forward: each test post's baseline is the median of the page's 8 posts
    # published before it (older test posts included, since their real numbers are
    # public by the time the next post goes out). The model itself never trains on test posts.
    base_l = rolling_baseline(df, "likes").loc[test.index]
    base_c = rolling_baseline(df, "comments").loc[test.index]
    sigma = rolling_sigma(df, "likes").loc[test.index].to_numpy()
    pl, pc = predict_logs(model, test, base_l, base_c)
    nq_l = naive_quantiles(train)

    out = test[["company", "post_id", "posted_at", "media_type", "hook", "likes", "comments"]].copy()
    out["pred_likes"] = np.round(np.expm1(pl)).astype(int)
    out["pred_likes_low"] = np.floor(np.expm1(pl + model.likes_resid_q[0] * sigma)).astype(int)
    out["pred_likes_high"] = np.ceil(np.expm1(pl + model.likes_resid_q[1] * sigma)).astype(int)
    out["pred_comments"] = np.round(np.expm1(pc)).astype(int)
    out["naive_likes"] = base_l.round().astype(int)
    out["in_range"] = (out["likes"] >= out["pred_likes_low"]) & (out["likes"] <= out["pred_likes_high"])

    lines += [
        "## 1. Temporal holdout (latest 25% of each page's posts)",
        "",
        f"Trained on {len(train)} older posts, tested on {len(test)} newer ones. "
        "Each test post's baseline is the median of the page's previous 8 posts (walk-forward); "
        "the naive row predicts exactly that baseline, so the model has to beat \"same as recent posts\".",
        "",
        "### Likes",
        "",
        HEADER,
    ]
    groups = [("All pages", out.index)] + [(c, out.index[out["company"] == c]) for c in sorted(out["company"].unique())]
    for name, idx in groups:
        a = out.loc[idx, "likes"].to_numpy()
        pos = out.index.get_indexer(idx)
        lines.append(fmt_row(f"**{name}** model", metrics(a, pl[pos], model.likes_resid_q, sigma[pos])))
        lines.append(fmt_row(f"{name} naive recent median", metrics(a, np.log1p(base_l.loc[idx].to_numpy()), nq_l, sigma[pos])))

    lines += ["", "### Comments", "", "| | n | model MAE | naive MAE | model exact | naive exact |", "|---|---|---|---|---|---|"]
    for name, idx in groups:
        a = out.loc[idx, "comments"].to_numpy()
        p = np.round(np.expm1(pc[out.index.get_indexer(idx)]))
        nb = np.round(base_c.loc[idx].to_numpy())
        lines.append(
            f"| {name} | {len(a)} | {np.mean(np.abs(p - a)):.2f} | {np.mean(np.abs(nb - a)):.2f} | "
            f"{np.mean(p == a):.0%} | {np.mean(nb == a):.0%} |"
        )
    lines.append("")
    return out


def leave_one_page_out(df: pd.DataFrame, lines: list[str]) -> None:
    lines += [
        "## 2. Cold start: leave one page out",
        "",
        "Model trained on the other pages; the held-out page is described only by its follower count.",
        "",
        HEADER,
    ]
    for company in sorted(df["company"].unique()):
        train, test = df[df["company"] != company], df[df["company"] == company]
        model = EngagementModel.fit(train)
        followers = int(test["company_followers"].iloc[0])
        a_l, b_l = model.cold_start["likes"]
        a_c, b_c = model.cold_start["comments"]
        base_l = pd.Series(math.expm1(a_l + b_l * math.log1p(followers)), index=test.index)
        base_c = pd.Series(max(0.0, math.expm1(a_c + b_c * math.log1p(followers))), index=test.index)
        pl, _ = predict_logs(model, test, base_l, base_c)
        worst_sigma = max(v["sigma"] for v in model.company_stats.values())
        lines.append(fmt_row(
            f"{company} ({followers:,} followers)",
            metrics(test["likes"].to_numpy(), pl, model.likes_resid_q, worst_sigma),
        ))
    lines.append("")


def main() -> None:
    df = load_posts()
    lines = [
        "# Accuracy report",
        "",
        f"Dataset: {len(df)} posts from {df['company'].nunique()} company pages "
        f"({', '.join(f'{c}: {n}' for c, n in df['company'].value_counts().sort_index().items())}), "
        f"posted {df['posted_at'].min():%Y-%m-%d} to {df['posted_at'].max():%Y-%m-%d}.",
        "",
        "Columns: **MAE** mean absolute error in likes. **median % error** typical miss relative to the true value. "
        "**within ±30%** share of posts where the point estimate was within 30% of reality. "
        "**in 80% range** share of posts whose real likes fell inside the predicted range (target: ~80%). "
        "**range high/low** how wide the range is (2.0x means e.g. 100-200).",
        "",
    ]
    preds = temporal_holdout(df, lines)
    leave_one_page_out(df, lines)

    (OUT_DIR / "results.md").write_text("\n".join(lines))
    preds.to_csv(OUT_DIR / "holdout_predictions.csv", index=False)
    print("\n".join(lines))
    print("\nReference-page holdout predictions:")
    cols = ["company", "posted_at", "media_type", "likes", "pred_likes_low", "pred_likes", "pred_likes_high", "comments", "pred_comments"]
    print(preds[preds["company"].isin(REFERENCE_PAGES)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
