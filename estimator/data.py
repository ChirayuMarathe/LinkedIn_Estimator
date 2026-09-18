"""Load the scraped JSON dumps into one clean DataFrame."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .features import HOOK_CHARS, TextStats, text_stats

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "json"
PROCESSED_CSV = ROOT / "data" / "processed" / "posts.csv"

# Posts younger than this are still collecting reactions, so their counts
# understate final engagement. They are dropped from training/eval.
MIN_AGE_DAYS = 2


def _parse_ts(ts: str) -> datetime:
    ts = ts.replace("Z", "")
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def _normalize(raw: dict, company: str, followers: int, scraped_at: str) -> dict | None:
    # Two dump formats exist: the verbose first pass (post_id/text/likes/...) and the
    # compact later pass (id/t/l/... with text stats precomputed on the full text).
    if "post_id" in raw:
        if raw.get("reshare"):
            return None
        text = raw.get("text", "")
        stats = text_stats(text)
        rec = dict(
            post_id=raw["post_id"], hook=text[:HOOK_CHARS], likes=raw["likes"], comments=raw["comments"],
            reposts=raw.get("reposts", 0), media_type=raw["media_type"], num_images=raw.get("num_images", 0),
            posted_at=raw["posted_at"],
        )
    else:
        if raw.get("rp"):
            return None  # repost of another account: engagement is not this page's
        text = raw.get("t", "")
        if "tl" in raw:
            stats = TextStats(raw["tl"], raw["h"], raw["e"], raw["u"], raw["q"])
        else:
            stats = text_stats(text)
        rec = dict(
            post_id=raw["id"], hook=text[:HOOK_CHARS], likes=raw["l"], comments=raw["c"],
            reposts=raw.get("r", 0), media_type=raw["m"], num_images=raw.get("n", 0), posted_at=raw["ts"],
        )
    rec.update(
        company=company, company_followers=followers, scraped_at=scraped_at,
        text_len=stats.length, hashtags=stats.hashtags, emoji=stats.emoji,
        links=stats.links, questions=stats.questions,
    )
    return rec


def load_posts(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        dump = json.loads(path.read_text(encoding="utf-8"))
        for raw in dump["posts"]:
            rec = _normalize(raw, dump["company"], dump["company_followers"], dump["scraped_at"])
            if rec:
                rows.append(rec)

    df = pd.DataFrame(rows)
    df["posted_at"] = df["posted_at"].map(_parse_ts)
    scraped = pd.to_datetime(df["scraped_at"]).dt.tz_localize("UTC")
    df["age_days"] = (scraped - df["posted_at"]).dt.total_seconds() / 86400

    df = df.drop_duplicates("post_id")
    # Same post published twice (seen on Ubiquiti): keep the first.
    df = df.drop_duplicates(["company", "hook", "likes", "comments"])
    df = df[df["age_days"] >= MIN_AGE_DAYS]
    return df.sort_values(["company", "posted_at"]).reset_index(drop=True)


if __name__ == "__main__":
    posts = load_posts()
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    posts.to_csv(PROCESSED_CSV, index=False)
    print(posts.groupby("company").agg(
        n=("post_id", "size"), followers=("company_followers", "first"),
        med_likes=("likes", "median"), med_comments=("comments", "median"),
    ))
    print(posts["media_type"].value_counts())
    print(f"wrote {len(posts)} rows -> {PROCESSED_CSV}")
