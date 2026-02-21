"""
ikman.lk Car Listings Scraper — v3 (built from real page HTML)
==============================================================

Key findings from actual page https://ikman.lk/en/ad/toyota-aqua-hybrid-2013-for-sale-colombo-251:

  1. Individual ad URLs use /en/ad/ (NOT /en/ads/)
       CORRECT:  https://ikman.lk/en/ad/toyota-aqua-hybrid-2013-for-sale-colombo-251

  2. The page is server-rendered plain HTML — no hashed CSS classes needed.
     All fields appear as simple <dt>/<dd> or label/value text pairs in the page body.

  3. Field extraction uses a DT→DD pattern or direct text parsing of the page.

  4. Listing index pages are at /en/ads/sri-lanka/cars?page=N
     Ad links within those pages contain /en/ad/ (singular).

Usage:
    pip install requests beautifulsoup4 pandas
    python scraper.py --test                   # 2 pages, verify output
    python scraper.py --pages 150              # full run ~3000 records
    python scraper.py --debug-url <URL>        # inspect one ad
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
import os
from datetime import datetime

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Config
BASE_URL       = "https://ikman.lk/en/ads/sri-lanka/cars"
OUTPUT_CSV     = "ikman_cars_raw.csv"
PAGES          = 150
MIN_DELAY      = 1.5
MAX_DELAY      = 3.5
PAGE_DELAY_MIN = 2.0
PAGE_DELAY_MAX = 5.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://ikman.lk/en/ads/sri-lanka/cars",
}

# HTTP helper

def safe_get(url, retries=3, backoff=5):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = backoff * attempt * 2
                log.warning(f"Rate limited. Waiting {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
            elif r.status_code == 403:
                log.warning(f"Blocked (403): {url}")
                return None
            else:
                log.warning(f"HTTP {r.status_code} on {url} (attempt {attempt}/{retries})")
                time.sleep(backoff)
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt}/{retries})")
            time.sleep(backoff * attempt)
    log.error(f"Failed after {retries} attempts: {url}")
    return None


# Step 1: Collect individual ad URLs from the listing page

def get_listing_urls(page: int) -> list:
    """
    Fetches the search results page and returns all individual ad URLs.

    Individual ads are at /en/ad/... (singular)
    Filter/brand pages are at /en/ads/... (plural) — these are EXCLUDED.

    Pattern confirmed from real page:
      <a href="/en/ad/toyota-aqua-hybrid-2013-for-sale-colombo-251">
    """
    url = f"{BASE_URL}?page={page}"
    r = safe_get(url)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    seen = set()
    urls = []

    for a in soup.find_all("a", href=True):
        href = a["href"].split("?")[0]

        # Must be /en/ad/ (singular) — individual ad pages only
        # Must end with a slug that has a numeric suffix (ad ID)
        if re.match(r"^/en/ad/[a-z0-9-]+-\d+$", href):
            full = "https://ikman.lk" + href
            if full not in seen:
                seen.add(full)
                urls.append(full)

    log.info(f"  Page {page}: found {len(urls)} ad URLs")

    if len(urls) == 0:
        log.warning("  Zero URLs found. Dumping page snippet for debugging:")
        log.warning(soup.get_text()[:400])

    return urls


# Step 2: Parse individual ad page

# Fields we want and the exact label text ikman uses
FIELD_LABELS = {
    "Brand":                "brand",
    "Model":                "model",
    "Trim / Edition":       "trim",
    "Year of Manufacture":  "year",
    "Condition":            "condition",
    "Transmission":         "transmission",
    "Body type":            "body_type",
    "Fuel type":            "fuel_type",
    "Engine capacity":      "engine_capacity",
    "Mileage":              "mileage",
}

def parse_listing(url: str) -> dict:
    """
    Parse one ad page.

    From the real page we confirmed:
      - Title is in <h1>
      - Price is plain text: "Rs 8,300,000"
      - Location appears after the title line: "Dehiwala, Colombo"
      - Fields appear as "Label:\nValue" text pairs in the page body
        e.g.  Brand:\nToyota    Model:\nAqua    Mileage:\n48,000 km

    Strategy: extract all text, then use a label-map regex scan.
    Also try <dt>/<dd> and <th>/<td> in case the page uses those.
    """
    r = safe_get(url)
    if r is None:
        return {}

    soup = BeautifulSoup(r.text, "html.parser")
    data = {"url": url, "scraped_at": datetime.now().isoformat(timespec="seconds")}

    # Title 
    h1 = soup.find("h1")
    data["title"] = h1.get_text(strip=True) if h1 else None

    # Price
    # Confirmed format: "Rs 8,300,000" as plain text somewhere near the top
    price_match = re.search(r"Rs[\s\u00a0]*([\d,]+)", soup.get_text())
    data["price_raw"] = "Rs " + price_match.group(1) if price_match else None

    # Location
    # From real page text we see the pattern:
    #   line N:   "Posted on"
    #   line N+1: "29 Jan 11:47 am"
    #   line N+2: ","
    #   line N+3: "Dehiwala"       ← area
    #   line N+4: ","
    #   line N+5: "Colombo"        ← district
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if line == "Posted on" and i + 5 < len(lines):
            # Skip timestamp (i+1), commas (i+2, i+4), grab area and district
            area     = lines[i + 3] if lines[i + 2] == "," else None
            district = lines[i + 5] if i + 5 < len(lines) and lines[i + 4] == "," else None
            parts = [p for p in [area, district] if p]
            if parts:
                data["location_raw"] = ", ".join(parts)
                data["area"]     = area
                data["district"] = district
            break

    # Fallback: find links like /en/ads/dehiwala/cars and /en/ads/colombo/cars
    if not data.get("location_raw"):
        loc_links = []
        for a in soup.find_all("a", href=True):
            m = re.match(r"^/en/ads/([a-z-]+)/cars$", a["href"])
            if m:
                loc_links.append(m.group(1).replace("-", " ").title())
        if loc_links:
            data["location_raw"] = ", ".join(loc_links[-2:])
            if len(loc_links) >= 1: data.setdefault("district", loc_links[-1])
            if len(loc_links) >= 2: data.setdefault("area",     loc_links[-2])

    # Field extraction: Method A — <dt>/<dd> pairs
    dt_tags = soup.find_all("dt")
    for dt in dt_tags:
        dd = dt.find_next_sibling("dd")
        if dd:
            label = dt.get_text(strip=True).rstrip(":")
            value = dd.get_text(strip=True)
            if label in FIELD_LABELS:
                data[FIELD_LABELS[label]] = value

    #  Field extraction: Method B — label text followed by value text
    # Works on pages where fields appear as plain "Label:\nValue" blocks
    # We look for each label string in the full page text
    if not any(data.get(v) for v in FIELD_LABELS.values()):
        page_text = soup.get_text(separator="\n")
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]

        for i, line in enumerate(lines):
            clean = line.rstrip(":")
            if clean in FIELD_LABELS and i + 1 < len(lines):
                col = FIELD_LABELS[clean]
                if not data.get(col):
                    data[col] = lines[i + 1]

    # Field extraction: Method C — inline "Label: Value" on same line 
    if not any(data.get(v) for v in FIELD_LABELS.values()):
        page_text = soup.get_text(separator=" ")
        for label, col in FIELD_LABELS.items():
            if not data.get(col):
                pattern = re.escape(label) + r"[:\s]+([^\n\|]{1,60})"
                m = re.search(pattern, page_text)
                if m:
                    data[col] = m.group(1).strip()

    #  Field extraction: Method D — th/td table rows
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).rstrip(":")
            value = cells[1].get_text(strip=True)
            if label in FIELD_LABELS and not data.get(FIELD_LABELS[label]):
                data[FIELD_LABELS[label]] = value

    # Description 
    # Confirmed: "Description" heading followed by text block
    desc_heading = soup.find(string=re.compile(r"^Description$", re.I))
    if desc_heading:
        parent = desc_heading.parent
        desc_el = parent.find_next_sibling() if parent else None
        if desc_el:
            data["description"] = desc_el.get_text(strip=True)[:500]

    return data


# Final column list 

FINAL_COLS = [
    "brand", "model", "trim", "year", "condition",
    "transmission", "body_type", "fuel_type", "engine_capacity", "mileage",
    "price_raw", "title", "location_raw", "description", "url", "scraped_at",
]

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    for col in FINAL_COLS:
        if col not in df.columns:
            df[col] = None
    return df[FINAL_COLS]


# Main scrape loop

def scrape(pages: int = PAGES, output: str = OUTPUT_CSV):
    all_records = []
    seen_urls: set = set()

    if os.path.exists(output):
        existing = pd.read_csv(output)
        seen_urls = set(existing["url"].dropna().tolist())
        all_records = existing.to_dict("records")
        log.info(f"Resuming — {len(seen_urls)} records already saved")

    try:
        for page in range(1, pages + 1):
            log.info(f"── Page {page}/{pages} ──")
            urls = get_listing_urls(page)

            if not urls:
                log.warning(f"No URLs on page {page}. Sleeping 15s.")
                time.sleep(15)
                continue

            for url in urls:
                if url in seen_urls:
                    log.info(f"  SKIP: {url}")
                    continue

                log.info(f"  Fetching: {url}")
                record = parse_listing(url)

                if record:
                    brand = record.get("brand", "?")
                    price = record.get("price_raw", "?")
                    year  = record.get("year", "?")
                    log.info(f"    ✓ {brand} | {year} | {price}")
                    all_records.append(record)
                    seen_urls.add(url)

                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            # Checkpoint after each page
            df = normalise(pd.DataFrame(all_records))
            df.to_csv(output, index=False)
            filled = df[["brand", "model", "year", "price_raw"]].notna().sum()
            log.info(
                f"  Checkpoint: {len(df)} total | "
                f"brand={filled['brand']} model={filled['model']} "
                f"year={filled['year']} price={filled['price_raw']}"
            )
            time.sleep(random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX))

    except KeyboardInterrupt:
        log.info("Interrupted — saving …")

    df = normalise(pd.DataFrame(all_records))
    df.to_csv(output, index=False)
    log.info(f"\n Done! {len(df)} records → '{output}'")
    return df


# Debug single URL

def debug_url(url: str):
    """Run: python scraper.py --debug-url "https://ikman.lk/en/ad/..."  """
    record = parse_listing(url)
    print("\n=== Extracted fields ===")
    for k, v in sorted(record.items()):
        print(f"  {k:25s}: {str(v)[:100]}")

    # Show what labels were actually found on the page
    r = safe_get(url)
    if r:
        soup = BeautifulSoup(r.text, "html.parser")
        lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
        print("\n=== Page text lines (first 80) ===")
        for i, line in enumerate(lines[:80]):
            print(f"  {i:3d}: {line}")


# Entry point 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ikman.lk car scraper v3")
    parser.add_argument("--pages",     type=int, default=PAGES)
    parser.add_argument("--output",    type=str, default=OUTPUT_CSV)
    parser.add_argument("--test",      action="store_true", help="Scrape 2 pages only")
    parser.add_argument("--debug-url", type=str, default=None)
    args = parser.parse_args()

    if args.debug_url:
        debug_url(args.debug_url)
    elif args.test:
        log.info("=== TEST MODE (2 pages) ===")
        df = scrape(pages=2, output="ikman_test.csv")
        print("\nSample output:")
        print(df.head(3).to_string())
        print("\nNon-null counts:")
        print(df.notna().sum().to_string())
    else:
        scrape(pages=args.pages, output=args.output)
