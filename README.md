# How This Thing Actually Works

So you want to know how this LinkedIn engagement estimator works. Fair warning — there's more going on under the hood than it looks. I'll walk through it piece by piece, starting simple and building up.

## The Problem

You're about to publish a post on a company's LinkedIn page. You want to know: **roughly how many likes and comments will it get?**

Not an exact number — that's impossible. But a reasonable range, like "somewhere between 500 and 900 likes." That's what this does.

---

## The Big Idea (Before Any Code)

Here's the core insight that drives everything:

**The single best predictor of how a post will do is how the page's *recent* posts have been doing.**

If Ubiquiti's last 8 posts averaged ~700 likes each, the next one will probably land somewhere around 700 too. If TP-Link Deutschland's last 8 posts averaged ~16 likes, their next one will probably be around 16.

That sounds obvious, but it's important because it means:
- The ML model's job is NOT to predict likes from scratch
- Its job is to predict **how much better or worse** a specific post will do compared to that page's recent average
- And honestly? With only 165 training posts, the model barely moves the needle beyond just guessing "about the same as usual"

The real value is in the **range** — telling you how confident (or not) the prediction is.

---

## Step 1: Getting the Data

LinkedIn doesn't have a public API for this. So the data was collected manually.

There's a JavaScript file (`scraper/extract_posts.js`) that you paste into your browser's DevTools console while logged into LinkedIn and viewing a company's posts page. You scroll through the posts manually, and every few scrolls you run `__extractPosts()` in the console. It reads the posts from the page's HTML.

For each post it grabs:
- The text (first 300 characters)
- Like count, comment count, repost count
- What type of media it has (video, image, carousel, etc.)
- The exact timestamp (decoded from LinkedIn's internal activity ID — the top 41 bits are milliseconds since epoch)
- Text stats like hashtag count, emoji count, links, question marks

One clever thing: the text stats are computed **in the browser on the full text**, even though only the first 300 chars of text are saved. This matters because LinkedIn truncates long posts with "...more", and the scraper can see the full text before truncation.

The raw data lives in `data/raw/json/` — one JSON file per company page. 165 usable posts from 4 pages: Ubiquiti, NETGEAR, TP-Link Systems, and TP-Link Deutschland.

---

## Step 2: Cleaning the Data

`estimator/data.py` loads those JSON files and does some housekeeping:

- **Drops reposts** — if a page reshared someone else's post, those likes belong to the original author, not this page
- **Drops duplicates** — same post ID, or same (company + text + likes + comments) combo
- **Drops posts less than 2 days old** — they're still accumulating reactions, so their counts are artificially low
- **Handles two dump formats** — earlier scrapes used verbose field names (`post_id`, `text`, `likes`), later scrapes used compact ones (`id`, `t`, `l`) to save space

Result: a clean pandas DataFrame with ~165 rows, sorted by company and posting time.

---

## Step 3: Building Features

`estimator/features.py` turns each post into a vector of numbers the model can chew on. There are about 27 features. The important ones:

### The baseline features (most important!)
- **`log_baseline_likes`** — log of the median likes of this page's last 8 posts. This is the anchor.
- **`log_baseline_comments`** — same for comments
- **`log_followers`** — log of the page's follower count

### Text features
- **`log_text_len`** — how long the post is
- **`hashtags`**, **`emoji`**, **`links`**, **`questions`** — counts of each (capped at sensible maximums)

### Format features (one-hot encoded)
- Is it a video? Image? Multi-image? Carousel? Poll? Document? Article? Text-only?
- 8 binary columns, one per format

### Keyword flags (checked on first 120 characters only!)
This is a neat design choice. LinkedIn shows roughly the first 120 characters before the "...more" fold. That's what a scrolling user actually reads before deciding to react. So keyword detection only looks at that "hook."

Six keyword categories:
- **Launch words**: "introducing", "new", "launch", "announcing"
- **Event words**: "IFA", "CES", "booth", "conference"
- **Award words**: "award", "winner", "best", "editor's choice"
- **Engagement asks**: "comment", "vote", "tell us"
- **People/culture**: "team", "intern", "colleagues"
- **Promo**: "offer", "deal", "sale", "discount"

These work in both English and German (for the TP-Link Deutschland page).

### Other
- **`is_german`** — detects German text by counting German stopwords and umlauts
- **`hour_utc`**, **`weekday`**, **`is_weekend`** — when the post was published

---

## Step 4: The Model

Here's where it gets interesting. The model is in `estimator/model.py`.

### What it predicts

The model does NOT predict raw like counts. It predicts the **log-space difference** between a post and its page's baseline:

```
target = log(1 + actual_likes) − log(1 + baseline_likes)
```

Why log space? Because engagement is multiplicative, not additive. A post doing "twice as well as usual" means +700 for Ubiquiti but +16 for TP-Link DE. In log space, "twice as well" is the same number regardless of scale.

Why relative to baseline? Because an Ubiquiti post getting 700 likes is average, but a TP-Link DE post getting 700 likes would be miraculous. The model needs to understand "above or below normal for this page," not raw numbers.

### The baseline: a rolling window

For each post, the baseline is the **median of the same page's previous 8 posts**. Not all-time median — a rolling window.

Why rolling? Because pages change. TP-Link Deutschland's engagement jumped around IFA 2026 (a big trade show). An all-time median would be dragged down by months of low-engagement posts and would underpredict their IFA posts. The rolling window tracks this drift.

Why median and not mean? Median is robust to outliers. If one post goes viral, it doesn't distort the baseline for the next 8 posts.

Important detail: the baseline only uses posts **strictly before** the current one in time. This means there's no data leakage — we only use information that would actually be available when the post goes live. Posts with fewer than 3 earlier posts on their page get no baseline and are dropped from training.

### The actual ML model

It's a `HistGradientBoostingRegressor` from scikit-learn. Think of it as a bunch of small decision trees that each correct the mistakes of the previous ones.

The settings are deliberately conservative:

```
max_depth=2          → very shallow trees (almost just "if X > threshold, go left/right")
max_iter=60          → only 60 rounds of boosting
min_samples_leaf=12  → each leaf must have at least 12 training examples
l2_regularization=3  → heavy penalty on large predictions
learning_rate=0.05   → learn slowly
```

Why so conservative? Because there are only ~130 usable training rows (after removing posts without enough history for a baseline). A bigger model would just memorise which page each post belongs to and overfit. The ablation study confirms this — a deeper model (depth 3, 150 iterations) actually performs worse on held-out data.

There are two separate models: one for likes, one for comments.

### The prediction range — this is the clever part

A point estimate like "this post will get 700 likes" isn't very useful by itself. What you actually want is: "somewhere between 500 and 900, with about 80% confidence."

Here's how the range is built. It uses a technique called **split-conformal prediction**, adapted with a twist:

**During training:**

1. Do 5-fold cross-validation on the training set. This gives you an "out-of-fold" prediction for every training post — a prediction made by a model that never saw that post.
2. Compute the residual for each post: `residual = actual_log_uplift − predicted_log_uplift`
3. **Here's the twist:** divide each residual by the page's **volatility** (standard deviation of log-likes over the page's recent posts). A miss of 0.3 log-units is a big deal for Ubiquiti (whose posts are super consistent) but totally normal for TP-Link DE (whose posts swing wildly). Normalising puts all pages on the same "how surprising is this miss?" scale.
4. Store the 10th and 90th percentiles of these normalised residuals. Call them `q10` and `q90`.

**At prediction time:**

```
range_low  = estimate × exp(q10 × this_page's_volatility)
range_high = estimate × exp(q90 × this_page's_volatility)
```

Because we multiply by the page's volatility, the range automatically becomes:
- **Tight for predictable pages** — Ubiquiti's range is about 1.8x (e.g., 510–937)
- **Wide for volatile pages** — TP-Link DE's range is about 10x (e.g., 18–131)

This is the single most important design choice in the whole project. The ablation study shows that switching from global ranges (same width for all pages) to per-page volatility-scaled ranges shrinks Ubiquiti's range from 7.8x to 1.8x without losing coverage for the volatile pages.

### Three ways to get a baseline

When you ask the model for a prediction, it needs a baseline for the page. There are three paths, in priority order:

1. **Known page** (e.g., `company="ubiquiti"`) — uses the stored median of the page's last 8 posts from training time. **Best accuracy.**
2. **User-provided baseline** (e.g., `baseline_likes=40`) — the user tells us what's typical for this page. **Decent accuracy.**
3. **Follower count only** (e.g., `company_followers=20000`) — estimates baseline from a rough log-log fit of median likes vs followers across the 4 known pages. **Bad accuracy** — the response flags this as "cold_start_followers_only" and the range uses the widest (most volatile) spread seen in training.

The cold start path exists because *something* is better than nothing, but follower count is genuinely a weak signal. In this dataset, Ubiquiti gets about 0.5% of its followers as likes per post, while NETGEAR gets about 0.04%. A 12x difference for a 1.7x difference in followers.

---

## Step 5: Training

`scripts/train.py` ties it all together:

1. Load and clean the data
2. Call `EngagementModel.fit(df)` — this trains both models (likes and comments), computes rolling baselines for every page, runs the cross-validation for conformal calibration, fits the cold-start regression, and stores per-page statistics
3. Save the model to `models/engagement_model.joblib` (~82 KB)
4. Write a `model_card.json` with training stats

The whole thing trains in seconds. It's a tiny model on tiny data.

---

## Step 6: Using It

Three ways to get predictions:

### Web form + API (`demo/app.py`)
A FastAPI app. Visit `http://127.0.0.1:8000` for a web form where you paste text, pick a format and page, and get back a likes range and comment estimate. Or POST to `/predict` with JSON for programmatic use.

### CLI (`demo/cli.py`)
```
python demo/cli.py --company ubiquiti --format video "Introducing UniFi Network 11"
→ Likes: 527 - 937  (estimate 710, 80% calibrated range)
```

### Python
```python
model = EngagementModel.load()
result = model.predict("Introducing our new router", media_type="video", company="ubiquiti")
```

---

## Step 7: Evaluating (The Honest Part)

The eval system (`eval/evaluate.py`) runs two tests:

### Test 1: Temporal holdout
For each page, hold out the newest 25% of posts. Train on the older 75%. Predict the holdout posts using walk-forward baselines (each post's baseline includes earlier holdout posts, since their numbers would be public by then).

**Results:**
- Overall: 76% of actual likes fall inside the predicted range (target was 80%)
- Ubiquiti: median error 10%, range 1.8x — great
- TP-Link DE: median error 63%, range 10.6x — wide but honest
- The model's MAE (43.4) basically ties the naive "just predict the baseline" (43.5)

### Test 2: Cold start (leave one page out)
Train on 3 pages, predict all posts of the 4th from follower count alone. Results are terrible (median errors of 47–877%). This proves that follower count alone doesn't work.

### Ablation study (`eval/ablation.py`)
Compares 5 model variants to justify the design choices. Key finding: **per-page volatility scaling on the range is what actually matters.** The choice between gradient boosting, linear regression, or no model at all barely changes the accuracy.

---

## The Uncomfortable Truth

This system is honest about its own limitations:

1. **The model barely beats "just guess the recent average."** With 165 posts, there's not enough data for post-level features (format, text, keywords) to show real lift. The eval always shows the naive baseline side by side so you can see this.

2. **Almost all the value comes from two things:**
   - Knowing the page's recent baseline (from its history)
   - The per-page volatility scaling on the range (making ranges tight for predictable pages, wide for volatile ones)

3. **The range is the real output, not the point estimate.** The point estimate is often wrong, but the range contains the real answer ~76–80% of the time. That's actually useful — it tells you "expect somewhere in this ballpark" rather than a false-precision number.

4. **For unknown pages, it's basically guessing.** The cold-start path exists to give you *something*, but the response explicitly says the confidence is low.

What would actually improve this? More data. 20+ pages, thousands of posts. Then text embeddings and content analysis could start pulling their weight. With 165 posts from 4 pages, you can't reliably learn that "video posts do 15% better" when the effect varies more between pages than between formats.

---

## Quick Reference: Data Flow

```
LinkedIn page (in browser)
    │
    ▼
extract_posts.js → data/raw/json/*.json     ← manual scraping
    │
    ▼
data.py → clean DataFrame (165 posts)        ← dedup, drop reposts, drop young posts
    │
    ▼
features.py → 27 features per post           ← text stats, keywords, format, baseline
    │
    ▼
model.py → EngagementModel                   ← gradient boosting on log-uplift
    │         ├── likes model
    │         ├── comments model
    │         ├── conformal range quantiles
    │         ├── per-page stats & volatility
    │         └── cold-start regression
    │
    ├──→ train.py → models/engagement_model.joblib (82 KB)
    ├──→ app.py   → web form + REST API
    ├──→ cli.py   → command line
    └──→ eval/    → accuracy report + ablation study
```
