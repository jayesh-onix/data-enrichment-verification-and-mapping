"""
Enrich accounts by querying Gemini with Google Search grounding.
The script reads `account_data.csv` and writes `account_data_enriched.csv`.

OPTIMIZED VERSION:
- Single comprehensive API call per account (all fields fetched at once)
- Async/await with concurrent processing for massive speed improvements
- Aggressive prompting to eliminate all "unknown" values
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


PROJECT_ID = "sales-excellence"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash"
INPUT_CSV = Path("Accounts SFA 21k.csv")
OUTPUT_CSV = Path("account_data_enriched.csv")

# Concurrency settings - adjust based on API rate limits
MAX_CONCURRENT_REQUESTS = 30
MAX_RETRIES = 2


# Regional groupings keep routing logic easy to read.
US_REGION_BY_STATE = {
    "east": {
        "ct", "de", "fl", "ga", "me", "md", "ma", "nh", "nj", "ny",
        "nc", "pa", "ri", "sc", "vt", "va", "wv",
    },
    "west": {
        "ak", "az", "ca", "hi", "nv", "or", "wa",
    },
    "north": {
        "id", "mt", "nd", "sd", "mn", "wi", "wy",
    },
    "south": {
        "al", "ar", "ky", "la", "ms", "ok", "tn", "tx",
    },
    "central": {
        "co", "il", "in", "ia", "ks", "mi", "mo", "ne", "nm", "oh", "ut",
    },
}

STATE_NAME_TO_CODE = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
}

STATE_CODE_TO_NAME = {
    "al": "Alabama",
    "ak": "Alaska",
    "az": "Arizona",
    "ar": "Arkansas",
    "ca": "California",
    "co": "Colorado",
    "ct": "Connecticut",
    "de": "Delaware",
    "fl": "Florida",
    "ga": "Georgia",
    "hi": "Hawaii",
    "id": "Idaho",
    "il": "Illinois",
    "in": "Indiana",
    "ia": "Iowa",
    "ks": "Kansas",
    "ky": "Kentucky",
    "la": "Louisiana",
    "me": "Maine",
    "md": "Maryland",
    "ma": "Massachusetts",
    "mi": "Michigan",
    "mn": "Minnesota",
    "ms": "Mississippi",
    "mo": "Missouri",
    "mt": "Montana",
    "ne": "Nebraska",
    "nv": "Nevada",
    "nh": "New Hampshire",
    "nj": "New Jersey",
    "nm": "New Mexico",
    "ny": "New York",
    "nc": "North Carolina",
    "nd": "North Dakota",
    "oh": "Ohio",
    "ok": "Oklahoma",
    "or": "Oregon",
    "pa": "Pennsylvania",
    "ri": "Rhode Island",
    "sc": "South Carolina",
    "sd": "South Dakota",
    "tn": "Tennessee",
    "tx": "Texas",
    "ut": "Utah",
    "vt": "Vermont",
    "va": "Virginia",
    "wa": "Washington",
    "wv": "West Virginia",
    "wi": "Wisconsin",
    "wy": "Wyoming",
    # Canadian provinces
    "ab": "Alberta",
    "bc": "British Columbia",
    "mb": "Manitoba",
    "nb": "New Brunswick",
    "nl": "Newfoundland and Labrador",
    "ns": "Nova Scotia",
    "on": "Ontario",
    "pe": "Prince Edward Island",
    "qc": "Quebec",
    "sk": "Saskatchewan",
}

EMPLOYEE_BUCKETS = [
    ("<500", lambda n: n < 500),
    ("500-1000", lambda n: 500 <= n < 1000),
    ("1000-2500", lambda n: 1000 <= n < 2500),
    ("2500-5000", lambda n: 2500 <= n < 5000),
    ("5000-10000", lambda n: 5000 <= n < 10000),
    ("10000-25000", lambda n: 10000 <= n < 25000),
    ("25000-50000", lambda n: 25000 <= n < 50000),
    (">50000", lambda n: n >= 50000),
]


# Pydantic model for comprehensive structured output from Gemini
class ComprehensiveEnrichmentResponse(BaseModel):
    """
    Complete enrichment data for a single account.
    All fields must be populated with best estimates - no 'unknown' values allowed.
    """
    website: str = Field(description="Official company website URL")
    description: str = Field(description="Two-sentence description of what the company does")
    employees_bucket: str = Field(description="Employee count bucket: <500, 500-1000, 1000-2500, 2500-5000, 5000-10000, 10000-25000, 25000-50000, >50000")
    hq_state: str = Field(description="Full state or province name (e.g., 'California', not 'CA')")
    region: str = Field(description="Geographic region: US east/west/north/south/central, or country name")
    industry: str = Field(description="Primary industry classification")
    annual_revenue_usd: str = Field(description="Annual revenue in USD as plain numeric string (no formatting, no currency symbols)")
    segment: str = Field(description="Company segment: startup, SME, Mid market, or large enterprise")


def normalize_string(value: str | None) -> str:
    if not value:
        return ""
    return value.strip()


def normalize_or_unknown(value: str | None) -> str:
    cleaned = normalize_string(value)
    return cleaned if cleaned else "unknown"


def convert_to_full_state_name(state_value: str | None) -> str:
    """
    Convert state code or partial state name to full state name.
    """
    if not state_value:
        return ""
    
    cleaned = normalize_string(state_value)
    cleaned_lower = cleaned.lower()
    
    # Check if it's already a full state name (capitalized)
    if cleaned.title() in STATE_NAME_TO_CODE.keys():
        return cleaned.title()
    
    # Check if it's a state code
    if cleaned_lower in STATE_CODE_TO_NAME:
        return STATE_CODE_TO_NAME[cleaned_lower]
    
    # Return as-is if not found (might be a province or international)
    return cleaned


def looks_like_address(text: str | None) -> bool:
    """
    Check if text appears to be a physical address rather than a description.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Common address indicators - improved patterns
    address_patterns = [
        # Standard street addresses with numbers
        r'^\d+\s+[a-z0-9\s]+(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|pl|place|way|pkwy|parkway|cir|circle)\b',
        # Suite/floor/unit patterns
        r'\b(suite|ste|floor|fl|unit|apt|apartment)\s*\d+',
        # US zip code (5 digits)
        r'\b\d{5}\b',
        # State code + zip code
        r'\b[a-z]{2}\s+\d{5}\b',
        # PO Box patterns
        r'\b(p\.?o\.?\s*box|po\s*box)\s*\d+',
    ]
    
    has_address_pattern = any(re.search(pattern, text_lower) for pattern in address_patterns)
    
    # Check for address keywords
    address_keywords = [
        'suite', 'floor', 'building', 'street', 'avenue', 'road', 'boulevard',
        'drive', 'lane', 'court', 'place', 'parkway', 'circle', 'overland park',
        'p.o. box', 'po box'
    ]
    has_address_keywords = any(keyword in text_lower for keyword in address_keywords)
    
    # Check if it starts with a street number (common for addresses)
    starts_with_number = bool(re.match(r'^\d+\s+[a-z]', text_lower))
    
    # Check for city, state, zip pattern (e.g., "CITY, ST 12345")
    has_city_state_zip = bool(re.search(r',\s*[a-z]{2}\s+\d{5}', text_lower))
    
    # If text is short and starts with number, it's very likely an address
    is_short_and_numbered = len(text.split()) <= 10 and starts_with_number
    
    # Count indicators
    indicators = [
        has_address_pattern,
        has_address_keywords,
        starts_with_number,
        has_city_state_zip,
        is_short_and_numbered
    ]
    indicator_count = sum(indicators)
    
    # If it has 2+ address indicators, it's likely an address
    return indicator_count >= 2


def parse_int(value: str | None) -> int | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def normalize_revenue(value: str | None) -> str:
    """
    Normalize revenue value to plain numeric string without currency symbols or formatting.
    Examples:
        "USD 598,222,000" -> "598222000"
        "$50M" -> "50000000"
        "1.5B" -> "1500000000"
    """
    if not value:
        return ""
    
    cleaned = normalize_string(value)
    if not cleaned or cleaned.lower() == "unknown":
        return "unknown"
    
    # Remove common currency symbols and prefixes
    cleaned = cleaned.replace("USD", "").replace("$", "").replace("€", "").replace("£", "")
    cleaned = cleaned.replace(",", "").replace(" ", "").strip()
    
    # Handle abbreviations like M (million), B (billion), K (thousand)
    multiplier = 1
    cleaned_upper = cleaned.upper()
    
    if cleaned_upper.endswith("B"):
        multiplier = 1_000_000_000
        cleaned = cleaned[:-1]
    elif cleaned_upper.endswith("M"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    elif cleaned_upper.endswith("K"):
        multiplier = 1_000
        cleaned = cleaned[:-1]
    
    # Try to parse as float then convert to int
    try:
        numeric_value = float(cleaned)
        final_value = int(numeric_value * multiplier)
        return str(final_value)
    except (ValueError, TypeError):
        # If parsing fails, just return digits only
        digits_only = "".join(ch for ch in value if ch.isdigit())
        return digits_only if digits_only else "unknown"


def bucket_employees(raw_value: str | None, suggested_bucket: str | None) -> str:
    cleaned = normalize_string(raw_value)
    if suggested_bucket:
        return suggested_bucket

    numeric_guess = parse_int(cleaned)
    if numeric_guess is None:
        return ""

    for label, matcher in EMPLOYEE_BUCKETS:
        if matcher(numeric_guess):
            return label
    return ""


def derive_region(hq_state: str | None, country: str | None) -> str:
    state_raw = normalize_string(hq_state).lower()
    state_code = STATE_NAME_TO_CODE.get(state_raw, state_raw)
    country_name = normalize_string(country)

    if not state_code:
        return country_name or "Unknown"

    for region, states in US_REGION_BY_STATE.items():
        if state_code in states:
            return f"US {region}"
    return country_name or "Unknown"


def derive_segment(bucket: str, fallback: str | None) -> str:
    explicit = normalize_string(fallback)
    if explicit:
        return explicit

    if not bucket:
        return ""

    if bucket == "<500":
        return "startup"
    if bucket in {"500-1000", "1000-2500"}:
        return "SME"
    if bucket in {"2500-5000", "5000-10000"}:
        return "Mid market"
    return "large enterprise"


def build_comprehensive_enrichment_prompt(row: Dict[str, str]) -> str:
    """
    Build a single comprehensive prompt that requests ALL enrichment fields at once.
    This dramatically reduces API calls and latency.
    """
    account_name = row.get('Account Name', '')
    existing_website = row.get('Website', '')
    existing_description = row.get('Description', '')
    existing_employees = row.get('Number of Employees') or row.get('Employees', '')
    existing_hq_state = row.get('HQ State') or row.get('Billing State/Province', '')
    existing_country = row.get('Billing Country', '')
    existing_industry = row.get('Industry', '')
    existing_revenue = row.get('Annual Revenue (converted)', '')
    
    return f"""You are a business intelligence expert with access to comprehensive Google Search.
Your task is to enrich the following company's data by finding ALL requested fields.

COMPANY INFORMATION:
- Account Name: {account_name}
- Existing Website: {existing_website}
- Existing Description: {existing_description}
- Existing Employees: {existing_employees}
- Existing HQ State: {existing_hq_state}
- Existing Country: {existing_country}
- Existing Industry: {existing_industry}
- Existing Revenue: {existing_revenue}

CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. **MANDATORY REQUIREMENT**: You MUST provide COMPLETE, ACCURATE values for ALL 8 fields below.
2. **NO "unknown" VALUES ALLOWED**: Make your best educated estimate for every field. Use industry benchmarks, typical company patterns, and logical inference when exact data is unavailable.
3. **USE COMPREHENSIVE GOOGLE SEARCH**: Search multiple sources for each data point - company websites, LinkedIn, Crunchbase, news articles, SEC filings, industry databases, press releases.

FIELDS TO POPULATE (return as strict JSON):

1. **website**: 
   - Find the official primary company website URL
   - Prefer official corporate site over social profiles
   - Try company name variations, check LinkedIn profile links, news mentions
   - If no official site exists, provide LinkedIn company page URL as fallback
   
2. **description**: 
   - Provide a concise 2-sentence description of the company's core business
   - Focus ONLY on what they do, their products/services, their business model
   - Do NOT include addresses, contact info, or location details
   - Search company About pages, LinkedIn, news articles, press releases
   
3. **employees_bucket**: 
   - Return ONE of these exact values: <500, 500-1000, 1000-2500, 2500-5000, 5000-10000, 10000-25000, 25000-50000, >50000
   - Search LinkedIn company page, Crunchbase, careers pages, news articles
   - If exact count unavailable, estimate based on: office size, funding rounds, market presence, job postings
   
4. **hq_state**: 
   - Return the FULL state or province name (e.g., "California" not "CA", "Ontario" not "ON")
   - Search company website footer, contact pages, LinkedIn, news articles, business registries
   - For US companies, use full state names; for Canadian, use full province names
   - If multiple offices, prioritize headquarters
   
5. **region**: 
   - If US company, map the state to ONE of: "US east", "US west", "US north", "US south", "US central"
   - US east: CT, DE, FL, GA, ME, MD, MA, NH, NJ, NY, NC, PA, RI, SC, VT, VA, WV
   - US west: AK, AZ, CA, HI, NV, OR, WA
   - US north: ID, MT, ND, SD, MN, WI, WY
   - US south: AL, AR, KY, LA, MS, OK, TN, TX
   - US central: CO, IL, IN, IA, KS, MI, MO, NE, NM, OH, UT
   - If non-US, return the country name (e.g., "Canada", "United Kingdom", "Germany")
   
6. **industry**: 
   - Return a clear, single industry classification
   - Common examples: Software & Internet, Healthcare, Manufacturing, Financial Services, Retail, Technology, Biotechnology, Consulting, Education, Energy, Real Estate, Media & Entertainment
   - Search company website, LinkedIn, Crunchbase, analyze their products/services
   - Use your best judgment to select the MOST ACCURATE industry
   
7. **annual_revenue_usd**: 
   - Return ONLY a plain numeric string with NO formatting, NO currency symbols, NO commas
   - Examples: "50000000" for $50M, "1500000000" for $1.5B, "500000" for $500K
   - Search: company website (Investor Relations), LinkedIn, Crunchbase, SEC filings, news, press releases
   - If exact revenue unavailable, ESTIMATE using:
     * Industry average revenue per employee (multiply by employee count)
     * Company size indicators (funding, market cap, office locations)
     * Competitor revenue comparisons
     * Industry benchmarks for similar-sized companies
   - Provide your BEST EDUCATED ESTIMATE - this is required
   
8. **segment**: 
   - Return ONE of: "startup", "SME", "Mid market", "large enterprise"
   - Guidelines:
     * startup: <500 employees, typically <$10M revenue, often venture-backed, <10 years old
     * SME: 500-2500 employees, typically $10M-$100M revenue
     * Mid market: 2500-10000 employees, typically $100M-$1B revenue
     * large enterprise: >10000 employees, typically >$1B revenue
   - Consider: employee count, revenue, market presence, company age, funding history

REMEMBER: 
- Return ONLY valid JSON matching the ComprehensiveEnrichmentResponse schema
- ALL fields must be populated with real values - NO "unknown" or "N/A" values
- Use your best judgment and industry knowledge when exact data is unavailable
- Make informed estimates rather than leaving any field empty

Return the JSON response now."""


async def call_gemini_comprehensive(
    client: genai.Client,
    prompt: str,
    attempt: int = 1
) -> Dict[str, Any]:
    """
    Call Gemini with comprehensive enrichment prompt using Google Search grounding.
    Returns structured data matching ComprehensiveEnrichmentResponse schema.
    
    This async function allows multiple accounts to be processed concurrently.
    """
    config_params = {
        "tools": [types.Tool(google_search=types.GoogleSearch())],
        "response_mime_type": "application/json",
        "response_schema": ComprehensiveEnrichmentResponse,
    }
    
    try:
        # Note: google.genai client doesn't have native async support yet
        # We use asyncio.to_thread to avoid blocking
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(**config_params)
        )
        
        result = json.loads(response.text or "{}")
        
        # Validate that no field contains "unknown" or empty values
        has_unknown_values = any(
            not str(value).strip() or str(value).lower() == "unknown"
            for value in result.values()
        )
        
        if has_unknown_values and attempt < MAX_RETRIES:
            logging.warning(
                "Attempt %d returned incomplete data. Retrying with more aggressive prompt...",
                attempt
            )
            # Add more aggressive instruction for retry
            aggressive_prompt = prompt + "\n\nIMPORTANT: Your previous attempt included 'unknown' values. This is NOT ACCEPTABLE. You MUST provide educated estimates for ALL fields based on industry patterns, similar companies, and logical inference. Try again with complete data."
            return await call_gemini_comprehensive(client, aggressive_prompt, attempt + 1)
        
        return result
        
    except json.JSONDecodeError as e:
        logging.error("JSON decode error on attempt %d: %s", attempt, e)
        if attempt < MAX_RETRIES:
            return await call_gemini_comprehensive(client, prompt, attempt + 1)
        return {}
    except Exception as e:
        logging.error("API error on attempt %d: %s", attempt, e)
        if attempt < MAX_RETRIES:
            # Wait a bit before retrying
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            return await call_gemini_comprehensive(client, prompt, attempt + 1)
        return {}


async def enrich_row(client: genai.Client, row: Dict[str, str]) -> Dict[str, str]:
    """
    Enrich a single account row using ONE comprehensive API call.
    This is dramatically faster than making 8+ separate calls per account.
    
    All post-processing and validation is applied to ensure data quality.
    """
    account_name = row.get("Account Name", "Unknown")
    
    # Build and execute the comprehensive prompt
    prompt = build_comprehensive_enrichment_prompt(row)
    enrichment_data = await call_gemini_comprehensive(client, prompt)
    
    # Extract and validate all fields
    website = normalize_string(enrichment_data.get("website") or row.get("Website"))
    if not website or website.lower() == "unknown":
        website = "No website found"  # Better than "unknown"
    
    raw_description = normalize_string(enrichment_data.get("description") or row.get("Description"))
    
    # Validate description is not an address
    if looks_like_address(raw_description):
        logging.warning(
            "Description for '%s' looks like an address, attempting to get proper description",
            account_name
        )
        raw_description = "Business information not available"
    
    description = raw_description if raw_description and raw_description.lower() != "unknown" else "Business information not available"
    
    # Process employees bucket
    employees_bucket_raw = normalize_string(enrichment_data.get("employees_bucket"))
    employees_bucket = bucket_employees(
        row.get("Number of Employees") or row.get("Employees"),
        employees_bucket_raw if employees_bucket_raw and employees_bucket_raw.lower() != "unknown" else None
    )
    
    # If still empty, use a reasonable default
    if not employees_bucket or employees_bucket.lower() == "unknown":
        employees_bucket = "<500"  # Conservative default for unknown companies
        logging.info("Using default employee bucket '<500' for '%s'", account_name)
    
    # Process HQ state
    raw_hq_state = normalize_string(enrichment_data.get("hq_state"))
    if not raw_hq_state or raw_hq_state.lower() == "unknown":
        raw_hq_state = row.get("HQ State") or row.get("Billing State/Province")
    
    hq_state = convert_to_full_state_name(raw_hq_state)
    if not hq_state or hq_state.lower() == "unknown":
        # Use country as fallback
        hq_state = row.get("Billing Country", "United States")
    
    # Process region
    region = normalize_string(enrichment_data.get("region"))
    if not region or region.lower() == "unknown":
        # Derive from HQ state and country
        region = derive_region(hq_state, row.get("Billing Country"))
    
    if not region or region.lower() == "unknown":
        region = row.get("Billing Country", "United States")
    
    # Process industry
    industry = normalize_string(enrichment_data.get("industry") or row.get("Industry"))
    if not industry or industry.lower() == "unknown":
        industry = "Technology & Services"  # Generic fallback
        logging.info("Using fallback industry 'Technology & Services' for '%s'", account_name)
    
    # Process revenue
    raw_revenue = enrichment_data.get("annual_revenue_usd") or row.get("Annual Revenue (converted)")
    annual_revenue = normalize_revenue(raw_revenue)
    
    # If still unknown, try to estimate based on employee bucket
    if not annual_revenue or annual_revenue == "unknown":
        # Industry average revenue per employee is roughly $200K-$500K
        # Estimate based on employee bucket midpoint
        employee_to_revenue_estimates = {
            "<500": "25000000",  # ~$25M (250 employees * $100K avg)
            "500-1000": "150000000",  # ~$150M (750 * $200K)
            "1000-2500": "400000000",  # ~$400M (1750 * $230K)
            "2500-5000": "1000000000",  # ~$1B (3750 * $270K)
            "5000-10000": "2250000000",  # ~$2.25B (7500 * $300K)
            "10000-25000": "5000000000",  # ~$5B (17500 * $290K)
            "25000-50000": "12000000000",  # ~$12B (37500 * $320K)
            ">50000": "25000000000",  # ~$25B (75000 * $330K)
        }
        annual_revenue = employee_to_revenue_estimates.get(employees_bucket, "50000000")
        logging.info("Estimated revenue for '%s' based on employee bucket: $%s", account_name, annual_revenue)
    
    # Process segment
    segment = normalize_string(enrichment_data.get("segment"))
    if not segment or segment.lower() == "unknown":
        segment = derive_segment(employees_bucket, None)
    
    if not segment or segment.lower() == "unknown":
        segment = "SME"  # Default fallback

    return {
        "Account ID 18 Digit": normalize_string(row.get("Account ID 18 Digit")) or "N/A",
        "Account Name": account_name,
        "Website": website,
        "Description": description,
        "Employees": employees_bucket,
        "HQ State": hq_state,
        "Region": region,
        "Industry": industry,
        "Annual Revenue": annual_revenue,
        "Segment": segment,
    }


def load_processed_account_ids() -> set[str]:
    """
    Load already processed account IDs from the output file to enable resume functionality.
    """
    if not OUTPUT_CSV.exists():
        return set()
    
    processed_ids = set()
    try:
        with OUTPUT_CSV.open(newline="", encoding="utf-8") as outfile:
            reader = csv.DictReader(outfile)
            for row in reader:
                account_id = row.get("Account ID 18 Digit")
                if account_id and account_id != "unknown":
                    processed_ids.add(account_id)
    except Exception as e:
        logging.warning("Could not read existing output file: %s", e)
        return set()
    
    return processed_ids


async def process_batch(
    client: genai.Client,
    rows_batch: list[Dict[str, str]],
    batch_num: int,
    total_batches: int
) -> list[Dict[str, str]]:
    """
    Process a batch of rows concurrently.
    This is where the massive speed improvement happens.
    """
    batch_start_time = time.time()
    
    logging.info(
        "Processing batch %d/%d with %d accounts...",
        batch_num,
        total_batches,
        len(rows_batch)
    )
    
    # Create tasks for all rows in the batch
    tasks = [enrich_row(client, row) for row in rows_batch]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    enriched_rows = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(
                "Error processing account '%s': %s",
                rows_batch[idx].get("Account Name", "Unknown"),
                result
            )
            # Create a fallback row with minimal data
            enriched_rows.append({
                "Account ID 18 Digit": rows_batch[idx].get("Account ID 18 Digit", "N/A"),
                "Account Name": rows_batch[idx].get("Account Name", "Unknown"),
                "Website": "Error",
                "Description": "Error during processing",
                "Employees": "<500",
                "HQ State": rows_batch[idx].get("Billing State/Province", "Unknown"),
                "Region": rows_batch[idx].get("Billing Country", "Unknown"),
                "Industry": "Unknown",
                "Annual Revenue": "0",
                "Segment": "SME",
            })
        else:
            enriched_rows.append(result)
    
    batch_duration = time.time() - batch_start_time
    accounts_per_second = len(rows_batch) / batch_duration if batch_duration > 0 else 0
    
    logging.info(
        "⏱️  Batch %d/%d completed in %.2f seconds (%.2f accounts/sec, %.2f sec/account)",
        batch_num,
        total_batches,
        batch_duration,
        accounts_per_second,
        batch_duration / len(rows_batch) if len(rows_batch) > 0 else 0
    )
    
    return enriched_rows


async def async_main() -> None:
    """
    Async main function that orchestrates parallel processing.
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # Load already processed accounts for resume functionality
    processed_ids = load_processed_account_ids()
    resume_mode = len(processed_ids) > 0
    
    if resume_mode:
        logging.info("Resume mode: Found %d already processed accounts", len(processed_ids))

    # Read all rows from input
    with INPUT_CSV.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        all_rows = list(reader)
    
    # Filter out already processed rows
    rows_to_process = [
        row for row in all_rows
        if row.get("Account ID 18 Digit") not in processed_ids
    ]
    
    total_rows = len(all_rows)
    rows_to_process_count = len(rows_to_process)
    already_processed_count = total_rows - rows_to_process_count

    if rows_to_process_count == 0:
        logging.info("All %d accounts already processed. Nothing to do.", total_rows)
        return

    logging.info(
        "Starting PARALLEL enrichment: %d total accounts, %d already done, %d to process",
        total_rows,
        already_processed_count,
        rows_to_process_count
    )
    logging.info(
        "Using %d concurrent requests per batch for maximum speed",
        MAX_CONCURRENT_REQUESTS
    )
    
    # Start timing the entire enrichment process
    total_start_time = time.time()

    fieldnames = [
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
    ]

    # Open output file in append mode if resuming, otherwise write mode
    file_mode = "a" if resume_mode else "w"
    
    # Process in batches to respect rate limits
    batch_size = MAX_CONCURRENT_REQUESTS
    num_batches = (rows_to_process_count + batch_size - 1) // batch_size
    
    with OUTPUT_CSV.open(file_mode, newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write header only if starting fresh
        if not resume_mode:
            writer.writeheader()

        processed_count = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, rows_to_process_count)
            batch_rows = rows_to_process[start_idx:end_idx]
            
            # Process this batch concurrently
            enriched_batch = await process_batch(
                client,
                batch_rows,
                batch_idx + 1,
                num_batches
            )
            
            # Write all enriched rows from this batch
            for enriched in enriched_batch:
                writer.writerow(enriched)
                processed_count += 1
                
                account_name = enriched.get("Account Name", "Unknown")
                total_processed = already_processed_count + processed_count
                remaining = rows_to_process_count - processed_count
                
                logging.info(
                    "✓ Enriched '%s' (%d/%d total, %d remaining)",
                    account_name,
                    total_processed,
                    total_rows,
                    remaining,
                )
            
            outfile.flush()  # Ensure data is written immediately after each batch
            
            # Small delay between batches to be respectful to API rate limits
            if batch_idx < num_batches - 1:
                await asyncio.sleep(1)

        total_duration = time.time() - total_start_time
        total_minutes = total_duration / 60
        total_hours = total_duration / 3600
        avg_time_per_account = total_duration / rows_to_process_count if rows_to_process_count > 0 else 0
        
        logging.info(
            "✅ Completed enrichment for all %d accounts! Output: %s",
            total_rows,
            OUTPUT_CSV
        )
        logging.info(
            "⏱️  Total time: %.2f seconds (%.2f minutes / %.2f hours)",
            total_duration,
            total_minutes,
            total_hours
        )
        logging.info(
            "⚡ Performance: %.2f accounts/sec | %.2f sec/account average",
            rows_to_process_count / total_duration if total_duration > 0 else 0,
            avg_time_per_account
        )
        logging.info(
            "🚀 Speedup vs sequential (est. 25 sec/account): %.1fx faster",
            (rows_to_process_count * 25) / total_duration if total_duration > 0 else 0
        )


def main() -> None:
    """
    Synchronous entry point that runs the async main function.
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

