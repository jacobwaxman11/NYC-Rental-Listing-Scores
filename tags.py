"""Controlled descriptive-tag vocabulary for apartment photos.

Tags power search and indexing in the web UI, so they are constrained to this
fixed list: the scorer is told to choose only from these and drops anything
else. Kept in its own (dependency-free) module so the vocabulary can be shared —
by the scorer, the DB tag rollup, and the web filters — without importing the
heavy provider SDKs that ``score_listings`` pulls in.
"""

TAG_GROUPS = {
    "Light & windows": [
        "bright", "sun_filled", "lots_of_windows", "floor_to_ceiling_windows",
        "large_windows", "corner_unit", "dim", "no_windows",
    ],
    "Layout": [
        "duplex", "loft", "open_floor_plan", "railroad_layout", "split_bedroom",
        "internal_stairs", "high_ceilings", "low_ceilings", "sunken_living_room",
        "separate_dining_room", "home_office_space", "walk_in_closet",
        "ample_closets", "narrow_rooms", "spacious", "cramped",
        "windowed_kitchen", "windowed_bathroom",
    ],
    "Kitchen": [
        "open_kitchen", "galley_kitchen", "eat_in_kitchen", "renovated_kitchen",
        "dated_kitchen", "stainless_appliances", "dishwasher", "gas_stove",
        "kitchen_island", "breakfast_bar", "stone_counters", "ample_cabinets",
    ],
    "Bathroom": [
        "renovated_bathroom", "dated_bathroom", "double_vanity", "soaking_tub",
        "walk_in_shower", "marble_bathroom",
    ],
    "Floors & finishes": [
        "hardwood_floors", "tile_floors", "carpet", "exposed_brick",
        "crown_molding", "decorative_fireplace", "working_fireplace",
        "recessed_lighting", "exposed_beams", "original_prewar_details",
        "modern_finishes", "luxury_finishes", "builder_grade_finishes",
    ],
    "Private outdoor": [
        "private_balcony", "private_terrace", "private_backyard",
        "private_patio", "juliet_balcony", "roof_access",
    ],
    "Condition & style": [
        "newly_renovated", "gut_renovated", "well_maintained", "needs_renovation",
        "prewar_charm", "modern_building", "staged", "furnished", "empty",
    ],
    "Views": [
        "skyline_view", "water_view", "park_view", "open_city_view",
        "courtyard_view", "brick_wall_view", "obstructed_view",
    ],
}
APARTMENT_TAGS = tuple(t for group in TAG_GROUPS.values() for t in group)
VALID_TAGS = set(APARTMENT_TAGS)
TAG_TEXT = "\n".join(f"  {name}: {', '.join(tags)}" for name, tags in TAG_GROUPS.items())
