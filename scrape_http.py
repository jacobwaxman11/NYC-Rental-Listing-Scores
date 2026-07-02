"""Shared HTTP bits for the StreetEasy scrapers: the browser-ish request headers
and the generic file downloader.

``_download_one`` is deliberately client-agnostic — the search scraper passes an
``httpx.Client`` and the detail backfill passes a ``curl_cffi`` session — so
image and floor-plan downloads share one implementation.
"""

from __future__ import annotations

import os
from typing import Optional

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://streeteasy.com/",
}


def _download_one(url: str, dest_path: str, client) -> bool:
    """Download a single URL to ``dest_path``. Returns True on success or if the
    file already exists.

    ``client`` is any object with a ``.get(url, timeout=...)`` returning a
    response with ``.raise_for_status()`` and ``.content`` — an ``httpx.Client``
    (image downloads) or a ``curl_cffi`` session (floor-plan downloads)."""
    if os.path.exists(dest_path):
        return True
    try:
        r = client.get(url, timeout=15)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"    ✗ download failed ({url}): {e}")
        return False


def download_images(listing: dict, base_dir: str, client) -> list[Optional[str]]:
    """Download all listing images into images/<building>/<unit>/.

    Filenames use the Zillow CDN hash (last path segment of the URL).
    Returns a list of local paths in the same order as listing['image_urls'];
    failed downloads produce a None placeholder so positional alignment is
    preserved when we write to listing_images.
    """
    building = listing.get("building_slug") or listing["listing_id"]
    unit = listing.get("unit") or listing["listing_id"]
    unit_dir = os.path.join(base_dir, building, unit)
    os.makedirs(unit_dir, exist_ok=True)

    local_paths: list[Optional[str]] = []
    for img_url in listing.get("image_urls", []):
        filename = img_url.split("/")[-1]  # e.g. "b0c21bbf...-p_e.webp"
        filepath = os.path.join(unit_dir, filename)

        if _download_one(img_url, filepath, client):
            local_paths.append(filepath)
        else:
            local_paths.append(None)  # keep positional alignment

    return local_paths
