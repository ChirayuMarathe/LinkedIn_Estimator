"""
Feature engineering shared by training and inference.

Everything the model sees is computed here, so a post typed into the demo goes
through exactly the same transformation as a scraped training post.

Design notes
------------
* Text stats (length, hashtags, emoji, links, question marks) are computed on the
  FULL text. For some scraped rows only a text snippet was stored, so the scraper
  computed these stats in the browser on the full text and they are passed in
  directly (see `text_stats`).
* Keyword flags are computed on the "hook" only: the first HOOK_CHARS characters.
  On LinkedIn that is roughly what shows above the "...more" fold, which is what a
  scrolling user actually reads before deciding to react. It also keeps training
  and inference consistent, because every stored snippet is at least that long.
* The single most important signal is the page's own baseline (median likes of its
  recent posts). Follower count alone is a weak proxy: in our data Ubiquiti gets
  ~0.5% of followers as likes, NETGEAR ~0.04%. So the model predicts how much a
  post will over/under-perform its page's baseline, not raw likes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

HOOK_CHARS = 120

MEDIA_TYPES = ["text_only", "image", "multi_image", "carousel", "video", "document", "poll", "article"]

# Aliases accepted from users / the API.
MEDIA_ALIASES = {
    "text": "text_only",
    "photo": "image",
    "images": "multi_image",
    "gallery": "multi_image",
    "pdf": "document",
    "link": "article",
    "reel": "video",
}

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF\U0000FE0F]"
)
_HASHTAG_RE = re.compile(r"#\w+")
_LINK_RE = re.compile(r"https?://|lnkd\.in")

_KEYWORDS = {
    "kw_launch": r"\b(introduc\w*|meet|new|now available|launch\w*|announc\w*|unveil\w*|neu\w*|enth(ü|u)ll\w*|vorgestellt)\b",
    "kw_event": r"\b(booth|stand|halle|hall|ifa|ces|infocomm|ibc|expo|conference|register|webinar|messe|live)\b",
    "kw_award": r"\b(award\w*|winner|editor'?s choice|best|auszeichn\w*|ausgezeichnet|preis|top picks?)\b",
    "kw_cta_engage": r"\b(comment\w*|vote|drop your|tell us|kommentar\w*|ratet|tippt|schreib\w*)\b",
    "kw_people": r"\b(team|intern\w*|colleague\w*|kolleg\w*|people|mitarbeiter\w*|celebrat\w*)\b",
    "kw_promo": r"\b(offer|deal|sale|discount|black ?friday|free shipping|angebot|rabatt)\b",
}
_KEYWORD_RES = {k: re.compile(v, re.IGNORECASE) for k, v in _KEYWORDS.items()}

_GERMAN_STOPWORDS = {
    "und", "der", "die", "das", "wir", "ihr", "ist", "mit", "für", "auf", "nicht",
    "euch", "unser", "unsere", "eine", "ein", "den", "dem", "sich", "auch", "oder", "bei",
}


@dataclass
class TextStats:
    length: int
    hashtags: int
    emoji: int
    links: int
    questions: int


def text_stats(text: str) -> TextStats:
    text = text or ""
    return TextStats(
        length=len(text),
        hashtags=len(_HASHTAG_RE.findall(text)),
        emoji=len(_EMOJI_RE.findall(text)),
        links=len(_LINK_RE.findall(text)),
        questions=text.count("?"),
    )


def normalize_media_type(media_type: str | None) -> str:
    m = (media_type or "text_only").strip().lower().replace("-", "_").replace(" ", "_")
    m = MEDIA_ALIASES.get(m, m)
    if m not in MEDIA_TYPES:
        raise ValueError(f"Unknown media_type '{media_type}'. Use one of: {', '.join(MEDIA_TYPES)}")
    return m


def is_german(snippet: str) -> int:
    words = re.findall(r"[a-zäöüß]+", (snippet or "").lower())
    if not words:
        return 0
    hits = sum(w in _GERMAN_STOPWORDS for w in words)
    has_umlaut = any(ch in snippet.lower() for ch in "äöüß")
    return int(hits >= 2 or (has_umlaut and hits >= 1))


def build_feature_row(
    *,
    hook: str,
    stats: TextStats,
    media_type: str,
    num_images: int,
    company_followers: int,
    baseline_likes: float,
    baseline_comments: float,
    posted_at: datetime | None,
) -> dict:
    """Return the model feature dict for one post."""
    media_type = normalize_media_type(media_type)
    hook = (hook or "")[:HOOK_CHARS]
    posted_at = posted_at or datetime.now(timezone.utc)

    row = {
        "log_followers": math.log1p(max(company_followers, 0)),
        "log_baseline_likes": math.log1p(max(baseline_likes, 0)),
        "log_baseline_comments": math.log1p(max(baseline_comments, 0)),
        "log_text_len": math.log1p(stats.length),
        "hashtags": min(stats.hashtags, 15),
        "emoji": min(stats.emoji, 15),
        "links": min(stats.links, 5),
        "questions": min(stats.questions, 5),
        "num_images": min(max(num_images, 0), 20),
        "is_german": is_german(hook),
        "hour_utc": posted_at.hour,
        "weekday": posted_at.weekday(),
        "is_weekend": int(posted_at.weekday() >= 5),
    }
    for m in MEDIA_TYPES:
        row[f"media_{m}"] = int(media_type == m)
    for name, rx in _KEYWORD_RES.items():
        row[name] = int(bool(rx.search(hook)))
    return row


FEATURE_COLUMNS = list(
    build_feature_row(
        hook="",
        stats=TextStats(0, 0, 0, 0, 0),
        media_type="text_only",
        num_images=0,
        company_followers=0,
        baseline_likes=0,
        baseline_comments=0,
        posted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ).keys()
)
