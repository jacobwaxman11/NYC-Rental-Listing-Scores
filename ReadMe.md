# NYCRentalRankings

Scrape StreetEasy rental listings, score their photos with Gemini **or Claude**,
and use regression models to identify mispriced apartments.

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

### Photo scoring provider

`score_listings.py` can score photos with either Gemini or Claude. Both produce
the **same** score shape (see `image_scores` table), so you can run one set of
images through each and compare. The provider/model that scored each image is
recorded in `image_scores.model`, and already-scored images are skipped
regardless of which provider produced them.

```bash
# Gemini (default)
python score_listings.py --provider gemini   --model gemini-2.5-flash

# Claude — default model is claude-opus-4-8; claude-haiku-4-5 is cheaper/faster
python score_listings.py --provider anthropic --model claude-opus-4-8
python score_listings.py --provider anthropic --model claude-haiku-4-5
```

Set the matching key in a `.env` file before running the scorer:
`GOOGLE_API_KEY` for `--provider gemini`, `ANTHROPIC_API_KEY` for
`--provider anthropic`.

In the same call, each apartment photo is also tagged with descriptive keywords
from a controlled ~80-term vocabulary (`hardwood_floors`, `exposed_brick`,
`duplex`, `high_ceilings`, `private_balcony`, `windowed_kitchen`, …; see
`TAG_GROUPS` in [`score_listings.py`](score_listings.py)). Tags are stored in the
`image_tags` table and rolled up per listing, powering tag chips and AI-search
filters in the web UI. No extra API call — the tags come back in the scoring
response.

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

## Web UI

[`web.py`](web.py) is a small Flask front-end that serves the listings the
model flags as underpriced. It builds the feature frame, predicts each
listing's "market" rent with **out-of-fold** Ridge predictions grouped by
building (the same leakage guard as the notebook), and ranks listings by how
far their actual rent sits below the prediction.

```bash
python web.py                       # http://127.0.0.1:5000
python web.py --db rentals.db --port 8000
```

Each card shows a representative photo (the brightest apartment shot), the
actual rent vs. the model's predicted rent, the % below/above market, and the
Gemini/Claude photo-quality score. Filter by area / beds, sort by discount or
quality, and toggle Underpriced / All / ❤ Liked / ✕ Passed views. You can ❤
listings and triage them in **Tinder mode** (swipe / arrow keys, with undo) —
both persist to the `listing_reactions` table. Suggestions are cached at
startup; hit `/?refresh=1` after re-running the scorer. Requires a populated
`rentals.db` (run the scrape → backfill → score pipeline first).

### AI search

Pass a model provider to enable natural-language search over the listings:

```bash
python web.py --provider anthropic --model claude-opus-4-8   # or --provider gemini
```

Ask things like *"like this but bigger"* (click ✨ on a card to set it as the
reference) or *"based on what I've liked, find similar ones I haven't reviewed"*.
**Cost is one small call per distinct question:** the model never sees the full
listing set — it only receives compact facets + the reference + a profile of
your likes, and returns a structured filter/ranking plan that the app executes
locally. Plans are cached per (query, reference, likes), so repeats are free.
Use `--model claude-haiku-4-5` (or `gemini-2.5-flash`) for the cheapest calls.
Needs `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` in `.env`; without it the UI runs
normally with AI search disabled.

### Semantic / taste matching (embeddings)

For description-aware "vibe" search, embed listings once with a local
sentence-transformers model (offline, no API cost):

```bash
python embed_listings.py            # embeds description + tags + basics
```

This unlocks two things in the web UI:

- **🧭 Match my likes** — ranks the listings you haven't reviewed by similarity
  to the centroid of your ❤ listings. Pure local vector math, **no model call**.
- **Free-text vibe** — the AI-search plan adds a `semantic_query` ("charming
  prewar with character and light"), which is embedded locally and blended into
  the ranking, so descriptive wants beyond the structured fields/tags count.

Embeddings are stored in `listing_embeddings` and re-run incrementally
(`--rebuild` to force). If `embed_listings.py` hasn't been run, Match-my-likes
is disabled and free-text search falls back to structured filters only.

## To do

- Improving the Gemini prompt / scoring model for image reviews
- Add building table to the DB
- `listing_snapshots` table to track price and status changes over time
- Explore non-linear models (gradient boosted trees) once dataset grows
- Target encoding for neighborhoods as an alternative to one-hot
