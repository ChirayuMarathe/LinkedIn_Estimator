# Data

`data/raw/json/*.json` holds the scraped posts, one file per scraping pass:

```json
{"company": "ubiquiti", "company_followers": 129191, "scraped_at": "2026-09-17", "posts": [ ... ]}
```

Two post formats exist, both handled by `estimator/data.py`:

| verbose (first pass) | compact (later passes) | meaning |
|---|---|---|
| `post_id` | `id` | LinkedIn activity id. Its top 41 bits are the creation time in ms. |
| `text` | `t` | Post text. Compact dumps keep only the first 120-300 chars. |
| (computed) | `tl`, `h`, `e`, `u`, `q` | Full-text length, hashtags, emoji, links, question marks, computed in the browser on the full text. |
| `likes` | `l` | Reaction count (all reaction types). |
| `comments` | `c` | Comment count. |
| `reposts` | `r` | Repost count (collected, not modelled). |
| `media_type` | `m` | `text_only`, `image`, `multi_image`, `carousel`, `video`, `document`, `poll`, `article` |
| `num_images` | `n` | Image count for image posts. |
| `posted_at` | `ts` | UTC timestamp decoded from the activity id. |
| `reshare` | `rp` | Repost of another account. Dropped: the numbers belong to the original author. |

Cleaning in `load_posts()`: drop reposts, drop duplicate ids and same-post-published-twice rows,
drop posts younger than 2 days (still collecting reactions).

`python -m estimator.data` writes the cleaned table to `data/processed/posts.csv` (git-ignored, regenerated on demand).
