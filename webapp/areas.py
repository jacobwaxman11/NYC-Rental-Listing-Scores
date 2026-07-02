"""StreetEasy neighborhood slugs -> display names for the Pipeline UI's area
picker.

Slugs are the reliable, human-readable form (the ``<slug>`` in a
``streeteasy.com/for-rent/<slug>`` URL) and every one below was validated to
return listings. The slug is shown next to each name in the UI so it stays
verifiable. To add more, grab the slug from StreetEasy's URL.
"""

AREA_OPTIONS = [
    # Lower Manhattan
    ("financial-district", "Financial District"),
    ("battery-park-city", "Battery Park City"),
    ("fulton-seaport", "Fulton/Seaport"),
    ("civic-center", "Civic Center"),
    ("tribeca", "Tribeca"),
    ("soho", "SoHo"),
    ("nolita", "Nolita"),
    ("little-italy", "Little Italy"),
    ("chinatown", "Chinatown"),
    ("les", "Lower East Side"),
    # Greenwich Village / Midtown South
    ("west-village", "West Village"),
    ("greenwich-village", "Greenwich Village"),
    ("east-village", "East Village"),
    ("noho", "Noho"),
    ("chelsea", "Chelsea"),
    ("west-chelsea", "West Chelsea"),
    ("flatiron", "Flatiron"),
    ("nomad", "NoMad"),
    ("gramercy-park", "Gramercy Park"),
    ("murray-hill", "Murray Hill"),
    ("kips-bay", "Kips Bay"),
    ("hudson-yards", "Hudson Yards"),
    ("hells-kitchen", "Hell's Kitchen"),
    # Brooklyn
    ("williamsburg", "Williamsburg"),
    ("greenpoint", "Greenpoint"),
    ("dumbo", "DUMBO"),
    ("brooklyn-heights", "Brooklyn Heights"),
    ("cobble-hill", "Cobble Hill"),
    ("carroll-gardens", "Carroll Gardens"),
    ("boerum-hill", "Boerum Hill"),
    ("park-slope", "Park Slope"),
    ("fort-greene", "Fort Greene"),
]
