"""
Account Data Enrichment Script v3
─────────────────────────────────
Uses Gemini with Google Search grounding to enrich account data with >90% accuracy.

Flow:
  1. Verify website & account name → correct if needed
  2. Enrich: Description, Employees, HQ State, Region, Industry,
     Annual Revenue, Segment, Region Category

Key design decisions:
  - One API call per account with Google Search tool (model searches web per account)
  - Structured JSON output via response_json_schema (Pydantic model_json_schema)
  - Region / Region Category derived deterministically from country + state lookups
  - Segment derived from multiple signals (revenue, employees, age, funding)
  - Native async via client.aio.models.generate_content
  - Concurrency controlled via asyncio.Semaphore
  - Resume support: skips already-processed account IDs in output file
  - Handles 40K+ rows via streaming batch writes with flush after each batch
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION – update these for your environment
# ═══════════════════════════════════════════════════════════════
PROJECT_ID = "search-ahmed"
LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"
INPUT_CSV = Path("data/test_final_10.csv")
OUTPUT_CSV = Path("data/enriched_final_test.csv")

MAX_CONCURRENT = 30       # simultaneous API calls (native async, no OS threads — safe at 30+)
BATCH_SIZE     = MAX_CONCURRENT * 2  # coroutines queued per gather; semaphore limits execution to MAX_CONCURRENT
MAX_RETRIES    = 3        # retries per account on API failure
RETRY_BASE_DELAY = 2     # seconds; exponential back-off factor


# ═══════════════════════════════════════════════════════════════
# STATIC LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════
US_REGION_BY_STATE: Dict[str, set] = {
    "US east": {
        "ct", "de", "fl", "ga", "me", "md", "ma", "nh", "nj", "ny",
        "nc", "pa", "ri", "sc", "vt", "va", "wv",
    },
    "US west": {"ak", "az", "ca", "hi", "nv", "or", "wa"},
    "US north": {"id", "mt", "nd", "sd", "mn", "wi", "wy"},
    "US south": {"al", "ar", "ky", "la", "ms", "ok", "tn", "tx"},
    "US central": {
        "co", "il", "in", "ia", "ks", "mi", "mo", "ne", "nm", "oh", "ut",
    },
}

STATE_NAME_TO_CODE: Dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    # Canadian provinces
    "alberta": "ab", "british columbia": "bc", "manitoba": "mb",
    "new brunswick": "nb", "newfoundland and labrador": "nl",
    "nova scotia": "ns", "ontario": "on", "prince edward island": "pe",
    "quebec": "qc", "saskatchewan": "sk",
}

STATE_CODE_TO_FULL: Dict[str, str] = {v: k.title() for k, v in STATE_NAME_TO_CODE.items()}

# Country → GEO / Region Category
COUNTRY_TO_GEO: Dict[str, str] = {
    # NA
    "united states": "NA", "usa": "NA", "us": "NA", "u.s.": "NA",
    "u.s.a.": "NA", "canada": "NA",
    # LATAM
    "mexico": "LATAM", "brazil": "LATAM", "argentina": "LATAM",
    "chile": "LATAM", "colombia": "LATAM", "peru": "LATAM",
    "ecuador": "LATAM", "venezuela": "LATAM", "costa rica": "LATAM",
    "panama": "LATAM", "uruguay": "LATAM", "paraguay": "LATAM",
    "bolivia": "LATAM", "guatemala": "LATAM", "honduras": "LATAM",
    "el salvador": "LATAM", "dominican republic": "LATAM",
    "puerto rico": "LATAM", "cuba": "LATAM", "jamaica": "LATAM",
    "trinidad and tobago": "LATAM", "nicaragua": "LATAM",
    # EMEA
    "united kingdom": "EMEA", "uk": "EMEA", "great britain": "EMEA",
    "england": "EMEA", "scotland": "EMEA", "wales": "EMEA",
    "germany": "EMEA", "france": "EMEA", "italy": "EMEA", "spain": "EMEA",
    "netherlands": "EMEA", "belgium": "EMEA", "switzerland": "EMEA",
    "sweden": "EMEA", "norway": "EMEA", "denmark": "EMEA", "finland": "EMEA",
    "austria": "EMEA", "ireland": "EMEA", "portugal": "EMEA",
    "poland": "EMEA", "czech republic": "EMEA", "czechia": "EMEA",
    "romania": "EMEA", "hungary": "EMEA", "greece": "EMEA",
    "turkey": "EMEA", "south africa": "EMEA", "nigeria": "EMEA",
    "kenya": "EMEA", "egypt": "EMEA", "israel": "EMEA",
    "saudi arabia": "EMEA", "uae": "EMEA",
    "united arab emirates": "EMEA", "qatar": "EMEA", "bahrain": "EMEA",
    "kuwait": "EMEA", "oman": "EMEA", "russia": "EMEA", "ukraine": "EMEA",
    "morocco": "EMEA", "tunisia": "EMEA", "ghana": "EMEA",
    "ethiopia": "EMEA", "tanzania": "EMEA", "luxembourg": "EMEA",
    "iceland": "EMEA", "croatia": "EMEA", "serbia": "EMEA",
    "bulgaria": "EMEA", "slovakia": "EMEA", "slovenia": "EMEA",
    "lithuania": "EMEA", "latvia": "EMEA", "estonia": "EMEA",
    # APAC
    "india": "APAC", "china": "APAC", "japan": "APAC",
    "south korea": "APAC", "korea": "APAC", "australia": "APAC",
    "new zealand": "APAC", "singapore": "APAC", "malaysia": "APAC",
    "indonesia": "APAC", "thailand": "APAC", "vietnam": "APAC",
    "philippines": "APAC", "hong kong": "APAC", "taiwan": "APAC",
    "bangladesh": "APAC", "pakistan": "APAC", "sri lanka": "APAC",
    "myanmar": "APAC", "cambodia": "APAC", "nepal": "APAC",
}


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODEL  –  defines what the LLM must return
# ═══════════════════════════════════════════════════════════════
class EnrichmentResult(BaseModel):
    """Structured enrichment result for a single account."""

    verified_account_name: str = Field(
        description="Verified/corrected official company name."
    )
    verified_website: str = Field(
        description="Verified/corrected official company website URL."
    )
    description: str = Field(
        description=(
            "2-3 sentence description of the company's core business, "
            "products and services. Must NOT include physical addresses."
        )
    )
    employee_count: str = Field(
        description=(
            "Estimated total number of employees as a plain number string "
            "e.g. '250', '5000', '45000'."
        )
    )
    hq_state_or_province: str = Field(
        description=(
            "Full name of the headquarters state or province "
            "e.g. 'California', 'Ontario', 'Bavaria'."
        )
    )
    hq_country: str = Field(
        description=(
            "Full country name of headquarters "
            "e.g. 'United States', 'Canada', 'Germany'."
        )
    )
    industry: str = Field(
        description=(
            "Primary industry classification e.g. 'Information Technology', "
            "'Healthcare', 'Manufacturing', 'Financial Services'."
        )
    )
    annual_revenue_usd: str = Field(
        description=(
            "Estimated annual revenue in USD as a plain number string "
            "without commas or symbols e.g. '50000000' for $50M."
        )
    )
    company_age_years: str = Field(
        description="Approximate company age in years e.g. '5', '25'."
    )
    funding_stage: str = Field(
        description=(
            "One of: 'bootstrapped', 'seed', 'series_a', 'series_b', "
            "'series_c_plus', 'public', 'private'."
        )
    )


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def clean(val: Any) -> str:
    """Strip whitespace; return '' for None."""
    return str(val).strip() if val is not None else ""


def is_empty(val: str | None) -> bool:
    """Return True when the value carries no useful information."""
    v = clean(val).lower()
    return v in ("", "unknown", "n/a", "none", "null", "not available")


def parse_number(val: str | None) -> int | None:
    """Parse a human-readable number string to int (best effort)."""
    if not val:
        return None
    s = clean(val)
    # Handle abbreviations (50M, 1.5B, 200K)
    multiplier = 1
    upper = s.upper()
    if upper.endswith("B"):
        multiplier = 1_000_000_000
        s = s[:-1]
    elif upper.endswith("M"):
        multiplier = 1_000_000
        s = s[:-1]
    elif upper.endswith("K"):
        multiplier = 1_000
        s = s[:-1]
    # Remove currency symbols, commas, spaces
    s = re.sub(r"[^\d.]", "", s)
    if not s:
        return None
    try:
        return int(float(s) * multiplier)
    except (ValueError, OverflowError):
        return None


def bucket_employees(count: int | None) -> str:
    """Map employee count to a standard bucket string."""
    if count is None or count < 500:
        return "<500"
    if count < 1000:
        return "500-1000"
    if count < 2500:
        return "1000-2500"
    if count < 5000:
        return "2500-5000"
    if count < 10000:
        return "5000-10000"
    if count < 25000:
        return "10000-25000"
    if count < 50000:
        return "25000-50000"
    return ">50000"


def normalize_state(state_val: str) -> str:
    """Convert state code or name to full title-cased name."""
    s = clean(state_val)
    if not s:
        return ""
    sl = s.lower()
    if sl in STATE_NAME_TO_CODE:
        return s.title()
    if sl in STATE_CODE_TO_FULL:
        return STATE_CODE_TO_FULL[sl]
    return s


def normalize_revenue(val: str | None) -> str:
    """Turn any revenue representation into a plain digit string."""
    n = parse_number(val)
    return str(n) if n else ""


def derive_us_region(state: str) -> str:
    """Return 'US east|west|north|south|central' from a state name/code."""
    code = STATE_NAME_TO_CODE.get(state.lower().strip(), state.lower().strip())
    for region, codes in US_REGION_BY_STATE.items():
        if code in codes:
            return region
    return ""


def derive_region(state: str, country: str) -> str:
    """Return the Region value: US sub-region or country name."""
    cl = country.lower().strip()
    if cl in ("united states", "usa", "us", "u.s.", "u.s.a.", ""):
        us_region = derive_us_region(state)
        return us_region if us_region else "United States"
    return country if country else "Unknown"


def derive_region_category(country: str, state: str) -> str:
    """Deterministically derive GEO: NA, LATAM, EMEA, APAC, ROW."""
    cl = country.lower().strip()
    geo = COUNTRY_TO_GEO.get(cl)
    if geo:
        return geo
    # Fallback: if state maps to a US state it's NA
    if state:
        code = STATE_NAME_TO_CODE.get(state.lower().strip(), state.lower().strip())
        for codes in US_REGION_BY_STATE.values():
            if code in codes:
                return "NA"
    # No country and no recognisable state → default NA (most accounts are US)
    if not cl:
        return "NA"
    return "ROW"


def derive_segment(
    emp_count: int | None,
    revenue: int | None,
    age: int | None,
    funding: str | None,
) -> str:
    """Derive company segment from multiple signals."""
    # Revenue is the strongest signal
    if revenue is not None:
        if revenue >= 1_000_000_000:
            return "large enterprise"
        if revenue >= 100_000_000:
            return "Mid market"
        if revenue >= 10_000_000:
            return "SME"
        if age is not None and age < 10:
            return "startup"
        return "SME"

    # Employee count as secondary signal
    if emp_count is not None:
        if emp_count >= 10000:
            return "large enterprise"
        if emp_count >= 2500:
            return "Mid market"
        if emp_count >= 500:
            return "SME"
        if age is not None and age < 10:
            return "startup"
        return "SME"

    # Funding stage as tertiary signal
    if funding:
        fl = funding.lower().strip()
        if fl in ("seed", "bootstrapped"):
            return "startup"
        if fl in ("series_a", "series_b"):
            return "SME"
        if fl == "series_c_plus":
            return "Mid market"
        if fl == "public":
            return "large enterprise"

    return "SME"


def looks_like_address(text: str) -> bool:
    """Return True if text looks like a physical address (not a description)."""
    t = text.lower()
    patterns = [
        r"^\d+\s+\w+\s+(st|street|ave|avenue|rd|road|blvd|dr|drive|ln|lane)\b",
        r"\b(suite|ste|floor|unit)\s*#?\d+",
        r"\b\d{5}\b",
        r"\bp\.?o\.?\s*box\b",
    ]
    return sum(1 for p in patterns if re.search(p, t)) >= 2


# ═══════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════
def build_prompt(account_id: str, account_name: str, website: str) -> str:
    return f"""You are a business data research expert. Use Google Search to find accurate, up-to-date information about the company below.

COMPANY:
  Account ID   : {account_id}
  Account Name : {account_name}
  Website      : {website}

────── STEP 1 — VERIFY IDENTITY ──────
Search for the company and its website to confirm:
• Is the website URL correct and reachable? If not, find the real official website.
• Is the account name the correct legal/trade name? If it is an abbreviation, DBA,
  or subsidiary, provide the most commonly known official name.
• If the website is clearly invalid (e.g. all digits, placeholder), search the
  account name to discover the real website.

────── STEP 2 — ENRICH ──────
Using the company website, LinkedIn company page, Wikipedia, Crunchbase, ZoomInfo,
press releases, SEC/EDGAR filings, news articles, and any other publicly available
sources, provide:

  verified_account_name  — corrected official company name
  verified_website       — corrected official website URL
  description            — 2-3 sentences about what the company does (products,
                           services, business model). NO addresses or phone numbers.
  employee_count         — total employee count as a number string (e.g. "1500").
                           Search LinkedIn first, then Crunchbase, careers page.
  hq_state_or_province   — full state/province name of HQ (e.g. "California")
  hq_country             — full country name (e.g. "United States")
  industry               — single primary industry label (e.g. "Information Technology",
                           "Healthcare", "Manufacturing", "Financial Services",
                           "Retail", "Energy", "Consulting", "Government")
  annual_revenue_usd     — estimated annual revenue in USD as plain digits
                           (e.g. "50000000" for $50 M). If not publicly available,
                           estimate from: industry-average revenue per employee ×
                           employee count, or from funding/valuation data.
  company_age_years      — approximate number of years since founding (e.g. "12")
  funding_stage          — one of: bootstrapped, seed, series_a, series_b,
                           series_c_plus, public, private

RULES:
 • Every field MUST have a real value — never return "unknown", "N/A", or "".
 • For government or military entities set industry="Government",
   funding_stage="public".
 • Return ONLY a valid JSON object with these exact keys:
   verified_account_name, verified_website, description, employee_count,
   hq_state_or_province, hq_country, industry, annual_revenue_usd,
   company_age_years, funding_stage
 • Do NOT wrap the JSON in markdown code blocks or add any explanation."""


# ═══════════════════════════════════════════════════════════════
# API CALL
# ═══════════════════════════════════════════════════════════════
def _extract_json(text: str) -> dict:
    """Extract a JSON object from model output that may contain markdown fences."""
    # Try raw parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return json.loads(cleaned)


async def call_gemini(
    client: genai.Client,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Two-phase enrichment:
      Phase 1 – Google Search grounding call (no controlled generation)
                to get factual web-sourced data.
      Phase 2 – Structured output call (no search) to clean the data
                into strict JSON.
    Falls back to parsing JSON directly from phase 1 when possible.
    """
    # Phase 1: Google Search grounding — no schema enforcement
    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Phase 1: search-grounded call
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=search_config,
                )
                raw_text = response.text or ""

                # Try to parse JSON directly from search response
                try:
                    result = _extract_json(raw_text)
                except json.JSONDecodeError:
                    # Phase 2: ask Gemini to restructure the raw text into JSON
                    schema_str = json.dumps(
                        EnrichmentResult.model_json_schema(), indent=2
                    )
                    structure_prompt = (
                        "Convert the following company research data into a strict "
                        "JSON object matching this schema. Return ONLY the JSON "
                        "object, no markdown fences, no explanations.\n\n"
                        f"SCHEMA:\n{schema_str}\n\n"
                        f"DATA:\n{raw_text}"
                    )
                    structured_config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=EnrichmentResult.model_json_schema(),
                    )
                    struct_resp = await client.aio.models.generate_content(
                        model=MODEL_NAME,
                        contents=structure_prompt,
                        config=structured_config,
                    )
                    result = json.loads(struct_resp.text or "{}")

                # Validate: reject if most fields are empty
                values = [str(v).strip() for v in result.values()]
                empty_count = sum(1 for v in values if is_empty(v))
                if empty_count > len(values) // 2 and attempt < MAX_RETRIES:
                    logging.warning(
                        "Attempt %d: too many empty fields (%d/%d), retrying…",
                        attempt, empty_count, len(values),
                    )
                    await asyncio.sleep(RETRY_BASE_DELAY * attempt)
                    continue

                return result

            except json.JSONDecodeError:
                logging.warning("Attempt %d: invalid JSON, retrying…", attempt)
            except Exception as exc:
                logging.warning("Attempt %d: %s, retrying…", attempt, exc)

            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BASE_DELAY * attempt)

    return {}


# ═══════════════════════════════════════════════════════════════
# ROW ENRICHMENT
# ═══════════════════════════════════════════════════════════════
OUTPUT_FIELDS = [
    "Account ID 18 Digit",
    "Account Name",
    "Website",
    "Description",
    "Employees",
    "HQ State",
    "Region",
    "Industry",
    "Annual Revenue",
    "Segment",
    "Region Category",
]


def _empty_row(row: Dict[str, str]) -> Dict[str, str]:
    """Return a row with all enrichment fields blank (fallback on total failure)."""
    return {
        "Account ID 18 Digit": row.get("Account ID 18 Digit", ""),
        "Account Name": row.get("Account Name", ""),
        "Website": row.get("Website", ""),
        "Description": "",
        "Employees": "",
        "HQ State": "",
        "Region": "",
        "Industry": "",
        "Annual Revenue": "",
        "Segment": "",
        "Region Category": "",
    }


async def enrich_row(
    client: genai.Client,
    row: Dict[str, str],
    semaphore: asyncio.Semaphore,
) -> Dict[str, str]:
    """Enrich a single CSV row — one API call, then deterministic post-processing."""
    account_id = clean(row.get("Account ID 18 Digit"))
    account_name = clean(row.get("Account Name"))
    website = clean(row.get("Website"))

    data = await call_gemini(
        client, build_prompt(account_id, account_name, website), semaphore
    )
    if not data:
        logging.error("No enrichment data for '%s'", account_name)
        return _empty_row(row)

    # ── Extract raw values returned by the model ──
    v_name = clean(data.get("verified_account_name"))
    v_website = clean(data.get("verified_website"))
    description = clean(data.get("description"))
    hq_state_raw = clean(data.get("hq_state_or_province"))
    hq_country = clean(data.get("hq_country"))
    industry = clean(data.get("industry"))
    funding = clean(data.get("funding_stage"))

    # ── Post-process each field ──
    final_name = v_name if not is_empty(v_name) else account_name
    final_website = v_website if not is_empty(v_website) else website

    if is_empty(description) or looks_like_address(description):
        description = ""
    # Remove citation markers like [1], [2, 3], etc.
    description = re.sub(r"\s*\[[\d,\s]+\]", "", description).strip()

    emp_count = parse_number(data.get("employee_count"))
    employees = bucket_employees(emp_count)

    hq_state = normalize_state(hq_state_raw)
    if is_empty(hq_country):
        hq_country = "United States"  # default for this dataset

    if is_empty(industry):
        industry = ""

    revenue_num = parse_number(data.get("annual_revenue_usd"))
    annual_revenue = str(revenue_num) if revenue_num else ""

    age = parse_number(data.get("company_age_years"))

    # ── Deterministic derivations (not relying on LLM for these) ──
    region = derive_region(hq_state, hq_country)
    segment = derive_segment(emp_count, revenue_num, age, funding)
    region_category = derive_region_category(hq_country, hq_state)

    return {
        "Account ID 18 Digit": account_id,
        "Account Name": final_name,
        "Website": final_website,
        "Description": description,
        "Employees": employees,
        "HQ State": hq_state,
        "Region": region,
        "Industry": industry,
        "Annual Revenue": annual_revenue,
        "Segment": segment,
        "Region Category": region_category,
    }


# ═══════════════════════════════════════════════════════════════
# RESUME SUPPORT
# ═══════════════════════════════════════════════════════════════
def load_processed_ids(output_path: Path) -> set:
    """Read already-written account IDs from the output CSV."""
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    try:
        with output_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                aid = row.get("Account ID 18 Digit")
                if aid:
                    ids.add(aid)
    except Exception:
        return set()
    return ids


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Resume support
    processed_ids = load_processed_ids(OUTPUT_CSV)
    resume = bool(processed_ids)
    if resume:
        logging.info("Resume mode — %d accounts already done", len(processed_ids))

    # Read input CSV
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if r.get("Account ID 18 Digit") not in processed_ids]
    total = len(all_rows)

    if not rows:
        logging.info("All %d accounts already processed. Nothing to do.", total)
        return

    logging.info(
        "Starting enrichment: %d to process, %d already done, %d total",
        len(rows), len(processed_ids), total,
    )

    t0 = time.time()
    batch_size = BATCH_SIZE
    mode = "a" if resume else "w"
    done = 0

    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if not resume:
            writer.writeheader()

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            tasks = [enrich_row(client, row, semaphore) for row in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    name = batch[j].get("Account Name", "?")
                    logging.error("Failed '%s': %s", name, result)
                    writer.writerow(_empty_row(batch[j]))
                else:
                    writer.writerow(result)
                done += 1
                name = (
                    batch[j].get("Account Name", "?")
                    if isinstance(result, Exception)
                    else result.get("Account Name", "?")
                )
                logging.info(
                    "[%d/%d] ✓ %s", done + len(processed_ids), total, name,
                )

            f.flush()

            # Small pause between batches to respect rate limits
            if i + batch_size < len(rows):
                await asyncio.sleep(0.2)

    elapsed = time.time() - t0
    logging.info(
        "Done — %d accounts enriched in %.1fs (%.2f s/account). Output: %s",
        len(rows), elapsed, elapsed / len(rows) if rows else 0, OUTPUT_CSV,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
