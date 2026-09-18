"""
Command-line demo.

Examples:
  python demo/cli.py --company ubiquiti --format video "Introducing UniFi Network 11 ..."
  python demo/cli.py --followers 20000 --baseline-likes 40 --format image --images 1 "We just shipped ..."
  python demo/cli.py --company tp-link-deutschland --format multi_image --images 4 --file post.txt
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from estimator.features import MEDIA_TYPES  # noqa: E402
from estimator.model import EngagementModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate likes and comments for a LinkedIn post.")
    p.add_argument("text", nargs="?", help="post text (or use --file)")
    p.add_argument("--file", type=Path, help="read post text from a file")
    p.add_argument("--format", default="text_only", choices=MEDIA_TYPES, dest="media_type")
    p.add_argument("--images", type=int, default=0, dest="num_images")
    p.add_argument("--company", help="a page the model knows")
    p.add_argument("--followers", type=int, dest="company_followers")
    p.add_argument("--baseline-likes", type=float)
    p.add_argument("--baseline-comments", type=float)
    p.add_argument("--json", action="store_true", help="print raw JSON")
    args = p.parse_args()

    text = args.file.read_text() if args.file else args.text
    if not text:
        p.error("provide post text or --file")

    model = EngagementModel.load()
    try:
        out = model.predict(
            text=text,
            media_type=args.media_type,
            num_images=args.num_images,
            company=args.company,
            company_followers=args.company_followers,
            baseline_likes=args.baseline_likes,
            baseline_comments=args.baseline_comments,
        )
    except ValueError as e:
        sys.exit(f"error: {e}")

    if args.json:
        print(json.dumps(out, indent=2))
        return
    l, c, b = out["likes"], out["comments"], out["baseline"]
    print(f"Likes:    {l['low']:,} - {l['high']:,}  (estimate {l['estimate']:,}, {out['interval']})")
    print(f"Comments: {c['estimate']}  (range {c['low']} - {c['high']})")
    print(f"Baseline: {b['likes']} likes / {b['comments']} comments  [{b['source']}]")


if __name__ == "__main__":
    main()
