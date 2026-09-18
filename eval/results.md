# Accuracy report

Dataset: 165 posts from 4 company pages (netgear: 57, tp-link-deutschland: 40, tp-link-systems: 31, ubiquiti: 37), posted 2025-09-18 to 2026-09-13.

Columns: **MAE** mean absolute error in likes. **median % error** typical miss relative to the true value. **within ±30%** share of posts where the point estimate was within 30% of reality. **in 80% range** share of posts whose real likes fell inside the predicted range (target: ~80%). **range high/low** how wide the range is (2.0x means e.g. 100-200).

## 1. Temporal holdout (latest 25% of each page's posts)

Trained on 124 older posts, tested on 41 newer ones. Each test post's baseline is the median of the page's previous 8 posts (walk-forward); the naive row predicts exactly that baseline, so the model has to beat "same as recent posts".

### Likes

| | n | MAE | median % error | within ±30% | in 80% range | range high/low |
|---|---|---|---|---|---|---|
| **All pages** model | 41 | 43.4 | 45% | 34% | 76% | 6.12x |
| All pages naive recent median | 41 | 43.5 | 42% | 46% | 73% | 6.15x |
| **netgear** model | 14 | 21.5 | 66% | 14% | 79% | 7.73x |
| netgear naive recent median | 14 | 21.5 | 57% | 36% | 79% | 7.77x |
| **tp-link-deutschland** model | 10 | 29.9 | 63% | 10% | 80% | 10.56x |
| tp-link-deutschland naive recent median | 10 | 27.6 | 54% | 20% | 70% | 10.63x |
| **tp-link-systems** model | 8 | 37.7 | 24% | 50% | 62% | 3.99x |
| tp-link-systems naive recent median | 8 | 37.0 | 22% | 62% | 62% | 4.01x |
| **ubiquiti** model | 9 | 97.7 | 10% | 78% | 78% | 1.83x |
| ubiquiti naive recent median | 9 | 101.1 | 9% | 78% | 78% | 1.84x |

### Comments

| | n | model MAE | naive MAE | model exact | naive exact |
|---|---|---|---|---|---|
| All pages | 41 | 1.78 | 1.80 | 15% | 17% |
| netgear | 14 | 0.93 | 1.00 | 36% | 14% |
| tp-link-deutschland | 10 | 2.20 | 2.00 | 0% | 10% |
| tp-link-systems | 8 | 1.25 | 1.00 | 12% | 38% |
| ubiquiti | 9 | 3.11 | 3.56 | 0% | 11% |

## 2. Cold start: leave one page out

Model trained on the other pages; the held-out page is described only by its follower count.

| | n | MAE | median % error | within ±30% | in 80% range | range high/low |
|---|---|---|---|---|---|---|
| netgear (77,000 followers) | 57 | 203.2 | 877% | 2% | 12% | 8.01x |
| tp-link-deutschland (4,710 followers) | 40 | 25.5 | 104% | 0% | 0% | 27.31x |
| tp-link-systems (45,000 followers) | 31 | 48.5 | 47% | 35% | 97% | 20.99x |
| ubiquiti (129,191 followers) | 37 | 622.8 | 91% | 0% | 3% | 23.81x |
