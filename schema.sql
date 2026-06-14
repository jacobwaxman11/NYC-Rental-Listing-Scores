PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS listings (
    listing_id              TEXT PRIMARY KEY,
    url                     TEXT,
    name                    TEXT,
    street                  TEXT,
    neighborhood            TEXT,
    zip                     TEXT,
    lat                     REAL,
    lng                     REAL,
    beds                    REAL,
    baths                   REAL,
    sqft                    INTEGER,
    rent                    INTEGER,
    lease_months            INTEGER,
    furnished               TEXT,
    building_type           TEXT,
    building_slug           TEXT,
    unit                    TEXT,
    description             TEXT,
    available_from          TEXT,
    floor_plan_url          TEXT,
    local_floor_plan_path   TEXT,
    scraped_at              TEXT,
    detail_fetched_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_listings_neighborhood ON listings(neighborhood);
CREATE INDEX IF NOT EXISTS idx_listings_building     ON listings(building_slug);

CREATE TABLE IF NOT EXISTS listing_amenities (
    listing_id  TEXT NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    amenity     TEXT NOT NULL,
    PRIMARY KEY (listing_id, amenity)
);
CREATE INDEX IF NOT EXISTS idx_amenity ON listing_amenities(amenity);

CREATE TABLE IF NOT EXISTS listing_images (
    listing_id        TEXT NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    position          INTEGER NOT NULL,
    image_url         TEXT NOT NULL,
    local_image_path  TEXT,
    PRIMARY KEY (listing_id, position)
);
CREATE INDEX IF NOT EXISTS idx_listing_images_url ON listing_images(image_url);

CREATE TABLE IF NOT EXISTS image_scores (
    image_url         TEXT PRIMARY KEY,
    image_category    TEXT,    -- 'apartment' | 'common_space' | 'irrelevant'
    -- apartment-only fields
    room_type         TEXT,
    natural_light     INTEGER,
    space_feeling     INTEGER,
    view_quality      INTEGER,
    -- common-space-only fields
    space_type        TEXT,
    appeal            INTEGER,
    -- shared between apartment + common_space
    finish_quality    INTEGER,
    condition_score   INTEGER,
    -- irrelevant-only
    irrelevant_reason TEXT,
    -- bookkeeping
    model             TEXT,
    scored_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_image_scores_category ON image_scores(image_category);
CREATE INDEX IF NOT EXISTS idx_image_scores_room     ON image_scores(room_type);
CREATE INDEX IF NOT EXISTS idx_image_scores_space    ON image_scores(space_type);

CREATE TABLE IF NOT EXISTS listing_scores (
    listing_id              TEXT PRIMARY KEY REFERENCES listings(listing_id) ON DELETE CASCADE,
    -- apartment aggregates
    avg_natural_light       REAL,
    avg_finish_quality      REAL,
    avg_space_feeling       REAL,
    avg_condition           REAL,
    max_view_quality        INTEGER,
    pct_bright_rooms        REAL,
    has_good_view           INTEGER,  -- 0/1
    -- common-space aggregates
    common_avg_finish_quality REAL,
    common_avg_condition      REAL,
    common_avg_appeal         REAL,
    -- counts
    photos_total            INTEGER,
    photos_scored           INTEGER,
    photos_apartment        INTEGER,
    photos_common           INTEGER,
    photos_irrelevant       INTEGER,
    -- nested distributions kept as JSON (small, not feature-y)
    room_type_counts_json   TEXT,
    space_type_counts_json  TEXT,
    aggregated_at           TEXT
);

CREATE TABLE IF NOT EXISTS image_tags (
    image_url  TEXT NOT NULL REFERENCES image_scores(image_url) ON DELETE CASCADE,
    tag        TEXT NOT NULL,
    PRIMARY KEY (image_url, tag)
);
CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag);

CREATE TABLE IF NOT EXISTS listing_reactions (
    listing_id  TEXT PRIMARY KEY REFERENCES listings(listing_id) ON DELETE CASCADE,
    reaction    TEXT NOT NULL,   -- 'liked' | 'passed'
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS listing_embeddings (
    listing_id   TEXT PRIMARY KEY REFERENCES listings(listing_id) ON DELETE CASCADE,
    model        TEXT,
    dim          INTEGER,
    text_hash    TEXT,            -- hash of (model, profile text) to skip re-embeds
    vector       TEXT,            -- JSON array of floats
    embedded_at  TEXT
);

-- Generic key/value store for small pipeline bookkeeping (e.g. last-poll times).
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT
);
