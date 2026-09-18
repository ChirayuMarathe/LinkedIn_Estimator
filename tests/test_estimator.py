import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from demo.app import app  # noqa: E402
from estimator.data import load_posts  # noqa: E402
from estimator.features import MEDIA_TYPES, is_german, normalize_media_type, text_stats  # noqa: E402
from estimator.model import EngagementModel, rolling_baseline  # noqa: E402

client = TestClient(app)


def test_text_stats():
    s = text_stats("Hello #wifi #mesh 🚀 see https://x.io and lnkd.in/abc?")
    assert (s.hashtags, s.emoji, s.links, s.questions) == (2, 1, 2, 1)


def test_media_aliases_and_rejects_unknown():
    assert normalize_media_type("PDF") == "document"
    assert normalize_media_type("text") == "text_only"
    with pytest.raises(ValueError):
        normalize_media_type("hologram")


def test_german_detection():
    assert is_german("Wir freuen uns auf euch und die IFA")
    assert not is_german("We are excited to see you at IFA")


def test_dataset_has_no_reposts_or_dupes():
    df = load_posts()
    assert df["post_id"].is_unique
    assert len(df) > 100
    assert (df["age_days"] >= 2).all()


def test_rolling_baseline_uses_only_past_posts():
    df = load_posts()
    base = rolling_baseline(df, "likes")
    g = df[df["company"] == "ubiquiti"].sort_values("posted_at")
    fourth = g.index[3]
    assert base[fourth] == g["likes"].iloc[:3].median()
    assert base[g.index[:3]].isna().all()  # not enough history yet


@pytest.mark.parametrize("fmt", MEDIA_TYPES)
def test_predict_every_format(fmt):
    m = EngagementModel.load()
    out = m.predict("Introducing our new router", media_type=fmt, company="ubiquiti")
    assert 0 <= out["likes"]["low"] <= out["likes"]["estimate"] <= out["likes"]["high"]
    assert 0 <= out["comments"]["low"] <= out["comments"]["estimate"] <= out["comments"]["high"]


def test_unknown_page_needs_followers():
    m = EngagementModel.load()
    with pytest.raises(ValueError):
        m.predict("hi", company="some-unknown-page")
    out = m.predict("hi", company_followers=10000, posted_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
    assert out["baseline"]["source"] == "cold_start_followers_only"


def test_api_roundtrip():
    r = client.post("/predict", json={"text": "Tag 1 auf der IFA!", "media_type": "multi_image", "num_images": 4, "company": "tp-link-deutschland"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"likes", "comments", "baseline"}
    assert client.post("/predict", json={"text": "x", "media_type": "nope", "company": "ubiquiti"}).status_code == 422
    assert client.get("/").status_code == 200
    assert "ubiquiti" in client.get("/pages").json()
