"""
Train the production model on every scraped post and save it.

Run:  python scripts/train.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from estimator.data import PROCESSED_CSV, load_posts  # noqa: E402
from estimator.model import MODEL_PATH, EngagementModel  # noqa: E402


def main() -> None:
    df = load_posts()
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)

    model = EngagementModel.fit(df)
    path = model.save()

    card = {
        "trained_on": model.trained_on,
        "likes_range_quantiles_x_sigma": model.likes_resid_q,
        "comments_range_quantiles": model.comments_resid_q,
        "default_sigma": model.default_sigma,
        "cold_start": model.cold_start,
        "pages": model.company_stats,
    }
    card_path = MODEL_PATH.with_name("model_card.json")
    card_path.write_text(json.dumps(card, indent=2, default=str))
    print(f"saved {path} ({path.stat().st_size / 1024:.0f} KB) and {card_path}")
    print(json.dumps(model.trained_on, indent=2))


if __name__ == "__main__":
    main()

