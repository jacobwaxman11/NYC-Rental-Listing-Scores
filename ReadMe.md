# NYCRentalRankings

Scrape StreetEasy rental listings, score their photos with Gemini, and build a
training set for a rent-prediction model.

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

Use it from a notebook:

```python
from features import prepare_features
df = prepare_features()
y = df["log_rent"]
X = df.drop(columns=["listing_id", "log_rent"])
```

Or from the CLI for inspection:

```bash
python features.py                          # print summary
python features.py --output features.csv    # also dump to disk
python features.py --include-unscored       # use all listings (mean-impute scores)
```

By default, listings whose images haven't been scored yet are dropped. Once
`score_listings.py` has covered the whole DB, `prepare_features()` returns
all 314 rows.

## One-time migration from JSON

If you still have `listings.json`, `listings_scored.json`, and
`image_scores_cache.json` from an earlier run, import them into the DB once:

```bash
python migrate_to_sqlite.py
```

After that the JSON files are unused — they can stay around as a backup.

## To do

- Feature engineering required to flatten before training model
- Ridge Regression model
- Improving the ML model used to score images
- Add building table to the db
- listing_snapshots table to handle change in listing status and price