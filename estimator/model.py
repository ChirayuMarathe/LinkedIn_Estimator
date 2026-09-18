"""
Engagement model: predicts likes (as a range) and comments for a LinkedIn post.

How it works
------------
1. Page baseline. Every page has its own "normal" engagement level (Ubiquiti's
   median post gets ~700 likes, TP-Link Deutschland's gets ~16). We take the
   median likes/comments of the page's recent posts as the baseline.
2. Post-level uplift. A gradient-boosted tree model predicts how far a specific
   post lands above or below that baseline, in log space, from its format, text
   and timing features.
3. Range. The likes range comes from split-conformal calibration: the 10th and
   90th percentile of the model's out-of-fold errors on the training set are
   added to the point prediction. On data like the training data, ~80% of real
   outcomes should land inside the range (we check this on held-out posts in
   eval/evaluate.py rather than assuming it).

If the page has no history in the model, a cold-start baseline is estimated from
follower count alone. That is a much weaker estimate and the API says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from .features import FEATURE_COLUMNS, TextStats, build_feature_row, text_stats

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "engagement_model.joblib"

INTERVAL_LOW_Q = 0.10
INTERVAL_HIGH_Q = 0.90


def make_regressor() -> HistGradientBoostingRegressor:
    # Small data (~130 rows): stumps-ish trees, big leaves, heavy L2, few iterations.
    # A deeper/longer model (max_depth=3, 150 iters) memorised page quirks and did
    # worse on held-out posts; see README "What we tried".
    return HistGradientBoostingRegressor(
        max_depth=2,
        learning_rate=0.05,
        max_iter=60,
        min_samples_leaf=12,
        l2_regularization=3.0,
        random_state=0,
    )


BASELINE_WINDOW = 8   # page baseline = median of the page's last N posts
MIN_HISTORY = 3       # posts with fewer earlier posts than this get no baseline


def rolling_baseline(df: pd.DataFrame, col: str, window: int = BASELINE_WINDOW) -> pd.Series:
    """
    For each post: median `col` of the same page's previous `window` posts (strictly
    earlier in time). This is exactly what is known at the moment a post goes live,
    so there is no target leakage, and it tracks drift (e.g. TP-Link Deutschland's
    engagement jumped around IFA 2026, so an all-time median would be stale).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.sort_values("posted_at").groupby("company"):
        vals = g[col].to_numpy(dtype=float)
        for i, idx in enumerate(g.index):
            prev = vals[max(0, i - window):i]
            if len(prev) >= MIN_HISTORY:
                out.loc[idx] = float(np.median(prev))
    return out


SIGMA_FLOOR = 0.15


def rolling_sigma(df: pd.DataFrame, col: str = "likes", window: int = BASELINE_WINDOW) -> pd.Series:
    """Std-dev of log1p(`col`) over the page's previous `window` posts: how volatile the page is."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, g in df.sort_values("posted_at").groupby("company"):
        vals = np.log1p(g[col].to_numpy(dtype=float))
        for i, idx in enumerate(g.index):
            prev = vals[max(0, i - window):i]
            if len(prev) >= MIN_HISTORY:
                out.loc[idx] = max(float(np.std(prev, ddof=1)), SIGMA_FLOOR)
    return out


def feature_frame(df: pd.DataFrame, baseline_likes: pd.Series, baseline_comments: pd.Series) -> pd.DataFrame:
    rows = [
        build_feature_row(
            hook=r.hook,
            stats=TextStats(r.text_len, r.hashtags, r.emoji, r.links, r.questions),
            media_type=r.media_type,
            num_images=r.num_images,
            company_followers=r.company_followers,
            baseline_likes=baseline_likes.loc[i],
            baseline_comments=baseline_comments.loc[i],
            posted_at=r.posted_at,
        )
        for i, r in df.iterrows()
    ]
    return pd.DataFrame(rows, index=df.index)[FEATURE_COLUMNS]


@dataclass
class EngagementModel:
    likes_model: HistGradientBoostingRegressor
    comments_model: HistGradientBoostingRegressor
    likes_resid_q: tuple[float, float]      # quantiles of residual / page sigma
    comments_resid_q: tuple[float, float]   # quantiles of raw log residual
    default_sigma: float
    company_stats: dict
    cold_start: dict
    trained_on: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(cls, df: pd.DataFrame, n_splits: int = 5) -> "EngagementModel":
        full = df.sort_values("posted_at").copy()
        base_l_all = rolling_baseline(full, "likes")
        base_c_all = rolling_baseline(full, "comments")
        sigma_all = rolling_sigma(full, "likes")
        keep = base_l_all.notna()
        df, base_l, base_c, sigma = full[keep], base_l_all[keep], base_c_all[keep], sigma_all[keep]
        X = feature_frame(df, base_l, base_c)
        y_l = np.log1p(df["likes"]) - np.log1p(base_l)
        y_c = np.log1p(df["comments"]) - np.log1p(base_c)

        cv = KFold(n_splits=min(n_splits, len(df)), shuffle=True, random_state=0)
        oof_l = cross_val_predict(make_regressor(), X, y_l, cv=cv)
        oof_c = cross_val_predict(make_regressor(), X, y_c, cv=cv)
        # Normalised conformal score: a miss of 0.3 log-units is huge for a steady page
        # (Ubiquiti) and routine for a volatile one (TP-Link DE), so scale by page volatility.
        resid_l = (y_l - oof_l) / sigma
        resid_c = y_c - oof_c

        likes_model = make_regressor().fit(X, y_l)
        comments_model = make_regressor().fit(X, y_c)

        stats = {}
        for company, g in full.groupby("company"):
            recent = g.sort_values("posted_at").tail(BASELINE_WINDOW)
            stats[company] = {
                "sigma": max(float(np.std(np.log1p(recent["likes"]), ddof=1)), SIGMA_FLOOR),
                "followers": int(g["company_followers"].iloc[0]),
                "median_likes": float(recent["likes"].median()),        # current baseline
                "median_comments": float(recent["comments"].median()),
                "alltime_median_likes": float(g["likes"].median()),
                "alltime_median_comments": float(g["comments"].median()),
                "n_posts": int(len(g)),
                "last_post": g["posted_at"].max().isoformat(),
            }

        # Cold start: log(median likes) ~ a + b * log(followers), fit on page medians.
        # Only as many points as pages, so this is a rough prior, not a model.
        f = np.log1p([s["followers"] for s in stats.values()])
        ml = np.log1p([s["alltime_median_likes"] for s in stats.values()])
        mc = np.log1p([s["alltime_median_comments"] for s in stats.values()])
        b_l, a_l = np.polyfit(f, ml, 1) if len(stats) > 1 else (0.0, float(ml.mean()))
        b_c, a_c = np.polyfit(f, mc, 1) if len(stats) > 1 else (0.0, float(mc.mean()))

        return cls(
            likes_model=likes_model,
            comments_model=comments_model,
            likes_resid_q=(float(np.quantile(resid_l, INTERVAL_LOW_Q)), float(np.quantile(resid_l, INTERVAL_HIGH_Q))),
            comments_resid_q=(float(np.quantile(resid_c, INTERVAL_LOW_Q)), float(np.quantile(resid_c, INTERVAL_HIGH_Q))),
            default_sigma=float(np.median([v["sigma"] for v in stats.values()])),
            company_stats=stats,
            cold_start={"likes": (float(a_l), float(b_l)), "comments": (float(a_c), float(b_c))},
            trained_on={
                "n_posts": int(len(df)),
                "companies": sorted(stats),
                "oof_mae_log_likes": float(np.mean(np.abs(y_l - oof_l))),
                "oof_mae_log_comments": float(np.mean(np.abs(resid_c))),
            },
        )

    # ------------------------------------------------------------------ inference
    def _baseline(self, company, company_followers, baseline_likes, baseline_comments):
        if baseline_likes is not None:
            return (
                float(baseline_likes),
                float(baseline_comments if baseline_comments is not None else baseline_likes * 0.02),
                company_followers or 0,
                self.default_sigma,
                "user_baseline",
            )
        if company and company in self.company_stats:
            s = self.company_stats[company]
            return s["median_likes"], s["median_comments"], company_followers or s["followers"], s["sigma"], "page_history"
        if not company_followers:
            raise ValueError(
                "Unknown page: pass `company_followers` (and ideally `baseline_likes`, the page's "
                f"typical likes per post). Known pages: {', '.join(sorted(self.company_stats))}"
            )
        lf = math.log1p(company_followers)
        a, b = self.cold_start["likes"]
        ac, bc = self.cold_start["comments"]
        # No history: we don't know the page's volatility either, so use the widest one seen.
        worst = max(v["sigma"] for v in self.company_stats.values())
        return math.expm1(a + b * lf), max(0.0, math.expm1(ac + bc * lf)), company_followers, worst, "cold_start_followers_only"

    def predict(
        self,
        text: str,
        media_type: str = "text_only",
        num_images: int = 0,
        company: str | None = None,
        company_followers: int | None = None,
        baseline_likes: float | None = None,
        baseline_comments: float | None = None,
        posted_at: datetime | None = None,
    ) -> dict:
        bl, bc, followers, sigma, mode = self._baseline(company, company_followers, baseline_likes, baseline_comments)
        row = build_feature_row(
            hook=text,
            stats=text_stats(text),
            media_type=media_type,
            num_images=num_images,
            company_followers=followers,
            baseline_likes=bl,
            baseline_comments=bc,
            posted_at=posted_at,
        )
        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        pl = float(self.likes_model.predict(X)[0]) + math.log1p(bl)
        pc = float(self.comments_model.predict(X)[0]) + math.log1p(bc)

        def to_range(p, q):
            lo, hi = q
            return {
                "low": max(0, round(math.expm1(p + lo))),
                "estimate": max(0, round(math.expm1(p))),
                "high": max(0, round(math.expm1(p + hi))),
            }

        return {
            "likes": to_range(pl, (self.likes_resid_q[0] * sigma, self.likes_resid_q[1] * sigma)),
            "comments": to_range(pc, self.comments_resid_q),
            "baseline": {"likes": round(bl, 1), "comments": round(bc, 1), "source": mode},
            "interval": f"{int((INTERVAL_HIGH_Q - INTERVAL_LOW_Q) * 100)}% calibrated range",
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "EngagementModel":
        return joblib.load(path)
