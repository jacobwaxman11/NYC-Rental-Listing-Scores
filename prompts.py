"""Prompts for the vision scorer (score_listings.py).

Separated from the scorer code so the prompt text + tag vocabulary can be read
and tuned without wading through the provider plumbing.
"""

from tags import TAG_TEXT


SYSTEM_PROMPT = (
    "You are a real estate analyst evaluating listing photos for NYC rental "
    "apartments. Be precise and return only valid JSON."
)


SCORE_PROMPT = """You will receive one or more listing photos in a single request.
For EACH image, classify it, then score it.

Return a single JSON ARRAY with one object per image, in the SAME ORDER the
images were provided. Do not wrap the array in any other object, do not add
commentary. If a single image is provided, still return an array of length 1.

Each array element MUST have this exact shape:
{
  "image_category": "apartment" | "common_space" | "irrelevant",
  "apartment_scores": { ... } | null,
  "common_space_scores": { ... } | null,
  "irrelevant_reason": "..." | null,
  "tags": [ ... ]
}

Classification rules:
- "apartment": inside the specific rental unit (living room, bedroom, kitchen,
  bathroom, private balcony/terrace, in-unit laundry closet, floor plan of the unit).
- "common_space": shared building areas (lobby, hallway, elevator, gym, pool,
  rooftop/roof deck, courtyard, shared laundry room, building exterior / facade,
  doorman desk, bike room, package room, mail room).
- "irrelevant": anything not useful for judging this listing — map/street view,
  neighborhood photo, logo, floor plan diagram for an unrelated unit, stock photo,
  text-only advertisement, watermarked placeholder, or unclear/unusable image.

If image_category = "apartment", populate "apartment_scores" and set the other two to null:
{
  "room_type": "living_room" | "kitchen" | "bedroom" | "bathroom" |
               "balcony_terrace" | "floor_plan" | "other_apartment",
  "natural_light": 1-10,
  "finish_quality": 1-10,
  "space_feeling": 1-10,
  "view_quality": 1-10,
  "condition": 1-10
}

If image_category = "common_space", populate "common_space_scores" and set the others to null:
{
  "space_type": "lobby" | "hallway" | "gym" | "pool" | "rooftop" | "courtyard" |
                "laundry_room" | "exterior" | "package_room" | "bike_room" | "other_common",
  "finish_quality": 1-10,
  "condition": 1-10,
  "appeal": 1-10
}

If image_category = "irrelevant", set both *_scores to null and fill "irrelevant_reason"
with a short phrase (e.g. "neighborhood map", "watermark placeholder", "street photo").

Scoring guide (apartment):
- natural_light: 1=dark/no windows, 10=bright/floor-to-ceiling windows
- finish_quality: 1=carpet/basic, 10=hardwood/stone/high-end finishes
- space_feeling: 1=cramped/cluttered, 10=open/airy/well-proportioned
- view_quality: 1=brick wall/no view, 10=skyline/water/park view (if no window visible, score 5)
- condition: 1=worn/dated, 10=brand new/pristine

Scoring guide (common_space):
- finish_quality: 1=basic/dated/builder-grade, 10=luxury materials and design
- condition: 1=worn/dirty, 10=pristine
- appeal: 1=uninviting/cramped, 10=impressive/desirable amenity

Tags (apartment images only): add a "tags" array of descriptive keywords chosen
ONLY from the controlled vocabulary below. Include a tag only when it is clearly
visible in the photo; omit anything uncertain. Use [] for common_space and
irrelevant images, and never invent tags outside this list:
""" + TAG_TEXT + """

IMPORTANT: the length of the returned array MUST equal the number of images
provided. Do not combine, skip, or merge images."""
