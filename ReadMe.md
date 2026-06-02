# NYCRentalRankings

Scrape StreetEasy rental listings, score their photos with Gemini, and use
regression models to identify mispriced apartments.

## Storage

All data lives in a single SQLite database, **`rentals.db`** (gitignored).
The schema is defined in [`db.py`](db.py); the relevant tables are:

| table | purpose |
|---|---|
| `listings` | one row per rental unit (rent, beds/baths, lat/lng, etc.) |
| `listing_amenities` | normalized amenity tags (one row per amenity) |
| `listing_images` | per-listing image refs (image_url + local path) |
| `image_scores` | Gemini score per unique image (keyed by image_url) |
| `listing_scores` | aggregated per-listing features (rebuildable) |

## Pipeline

```
scrape_listings.py   →   backfill_details.py   →   score_listings.py   →   features.py
   (search pages)            (detail pages,           (Gemini photo scoring)     (build training frame)
                              amenities, etc.)
```

The first three stages read from / write to `rentals.db` and can be re-run
idempotently: listings already present are skipped, only new work is done.

```bash
python scrape_listings.py --max-listings 100
python backfill_details.py --max-listings 50
python score_listings.py  --max-listings 50
```

Set `GOOGLE_API_KEY` in a `.env` file before running the scorer.

## Building the training frame

`features.py` joins `listings` ⨝ `listing_scores` ⨝ pivoted `listing_amenities`
into a single ML-ready DataFrame (target = `log_rent`).

`prepare_features()` returns a **base** frame with individual amenity one-hot
columns and all raw Gemini image-score columns. Two optional transforms can
be applied afterwards to evaluate their effect:

```python
from features import prepare_features, collapse_amenity_tiers, collapse_image_scores

df = prepare_features()                     # base frame (raw amenities + raw image scores)
y = df["log_rent"]
X = df.drop(columns=["listing_id", "log_rent"])

# optional compression for modeling
df_v2 = collapse_image_scores(df)           # 7 image cols → apt_quality + common_quality
df_v2 = collapse_amenity_tiers(df_v2)       # 40+ amen_* → amen_tier_a / tier_b / tier_c
```

`collapse_image_scores` averages correlated Gemini dimensions into two
composites (`apt_quality`, `common_quality`) while keeping `max_view_quality`,
`pct_bright_rooms`, and `has_good_view` as standalone features.

`collapse_amenity_tiers` groups amenities into three tiers:
- **Tier A** (premium): pool, gym, washer/dryer, doorman, central AC, parking, etc.
- **Tier B** (standard): elevator, dishwasher, laundry, package room, etc.
- **Tier C** (low-signal): fios_available, virtual, view tags, etc.

CLI usage:

```bash
python features.py                          # print summary (all transforms applied)
python features.py --output features.csv    # dump to disk
python features.py --raw                    # skip tier + composite transforms
python features.py --include-unscored       # include listings without image scores
```

## Modeling

[`NYC_Rental_Listing_Scores.ipynb`](NYC_Rental_Listing_Scores.ipynb) runs
Ridge and Lasso regression on both the raw and compressed feature sets,
compares results, and identifies underpriced listings.

Key design decisions:
- **GroupShuffleSplit** by building slug prevents data leakage (units in the
  same building don't appear in both train and test)
- **StandardScaler** inside the pipeline (refit per CV fold)
- Models are compared across raw (v1) and compressed (v2) feature sets

## To do

- Improving the Gemini prompt / scoring model for image reviews
- Add building table to the DB
- `listing_snapshots` table to track price and status changes over time
- Explore non-linear models (gradient boosted trees) once dataset grows
- Target encoding for neighborhoods as an alternative to one-hot
