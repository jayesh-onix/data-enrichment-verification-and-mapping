"""
Shared utilities for account enrichment + verification scripts.

Design goals:
- Extremely readable, predictable transformations.
- Conservative overwrites: only normalize/clean when we have high confidence.
- Keep business rules (state->region mapping, URL normalization) in one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


# --- Constants -----------------------------------------------------------------

# Regional groupings keep routing logic easy to read.
US_REGION_BY_STATE_CODE: dict[str, set[str]] = {
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

# US states
US_STATE_NAME_TO_CODE: dict[str, str] = {
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

# Canadian provinces (for normalization only; region stays "Canada")
CA_PROVINCE_CODE_TO_NAME: dict[str, str] = {
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

CA_PROVINCE_NAME_TO_CODE: dict[str, str] = {
    name.lower(): code for code, name in CA_PROVINCE_CODE_TO_NAME.items()
}

UNKNOWN_SENTINELS = {
    "",
    "unknown",
    "n/a",
    "na",
    "null",
    "none",
}

CANONICAL_US_REGIONS = {f"US {name}" for name in US_REGION_BY_STATE_CODE.keys()}


# --- Basic normalization --------------------------------------------------------


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().split())


def is_unknown(value: str | None) -> bool:
    cleaned = normalize_whitespace(value).lower()
    return cleaned in UNKNOWN_SENTINELS


def is_probably_invalid_website(value: str | None) -> bool:
    cleaned = normalize_whitespace(value)
    if is_unknown(cleaned):
        return True
    if cleaned.lower() == "no website found":
        return True
    return False


def normalize_region(value: str | None) -> str:
    """
    Normalize region casing for known regions (e.g. 'US East' -> 'US east').
    """
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return ""

    cleaned_lower = cleaned.lower()
    if cleaned_lower.startswith("us "):
        suffix = cleaned_lower.replace("us ", "", 1).strip()
        if suffix in US_REGION_BY_STATE_CODE:
            return f"US {suffix}"
        return f"US {suffix}"  # keep normalized casing even if suffix is unusual

    # Country names: just title-case common ones? Too risky—leave as-is.
    return cleaned


_NON_LETTER_CHARS = re.compile(r"[^a-zA-Z\s\-]", re.UNICODE)


def clean_state_like_text(value: str | None) -> str:
    """
    Clean obvious garbage from HQ State values (e.g. 'New Jersey}state{{').

    Conservative: only strips non-letter characters and normalizes whitespace.
    """
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return ""
    cleaned = _NON_LETTER_CHARS.sub(" ", cleaned)
    return normalize_whitespace(cleaned)


def normalize_hq_state(value: str | None) -> str:
    """
    Normalize US state / Canadian province to full name where possible.
    Otherwise return the cleaned input as-is.
    """
    cleaned = clean_state_like_text(value)
    if not cleaned:
        return ""

    cleaned_lower = cleaned.lower()

    # US state code like "CA"
    if len(cleaned_lower) == 2 and cleaned_lower.isalpha():
        code = cleaned_lower
        for full_name, full_code in US_STATE_NAME_TO_CODE.items():
            if full_code == code:
                return full_name.title()
        if code in CA_PROVINCE_CODE_TO_NAME:
            return CA_PROVINCE_CODE_TO_NAME[code]

    # Full name normalization
    if cleaned_lower in US_STATE_NAME_TO_CODE:
        return cleaned_lower.title()
    if cleaned_lower in CA_PROVINCE_NAME_TO_CODE:
        return CA_PROVINCE_CODE_TO_NAME[CA_PROVINCE_NAME_TO_CODE[cleaned_lower]]

    return cleaned


def derive_region_from_hq_state_and_existing_region(
    *,
    hq_state: str | None,
    existing_region: str | None,
) -> str:
    """
    If we can confidently map the state to a US region, do so.
    Otherwise keep existing region (normalized) as-is.
    """
    normalized_existing = normalize_region(existing_region)

    state_name = normalize_hq_state(hq_state)
    state_code = US_STATE_NAME_TO_CODE.get(state_name.lower())
    if not state_code:
        return normalized_existing

    for region_suffix, state_codes in US_REGION_BY_STATE_CODE.items():
        if state_code in state_codes:
            return f"US {region_suffix}"

    return normalized_existing


def normalize_website_url(value: str | None) -> str:
    """
    Normalize a website field into a usable URL.

    Rules:
    - If empty/unknown -> ""
    - If it has no scheme, assume https://
    - If it parses poorly, return cleaned original (caller may treat as invalid)
    """
    cleaned = normalize_whitespace(value)
    if is_probably_invalid_website(cleaned):
        return ""

    # Allow common 'example.com' or 'www.example.com'
    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", cleaned))
    url_candidate = cleaned if has_scheme else f"https://{cleaned}"

    parsed = urlparse(url_candidate)
    if parsed.netloc:
        return url_candidate

    # Fallback: sometimes value includes scheme-less URL with path that confuses parsing
    return cleaned


def normalize_annual_revenue_usd(value: str | None) -> str:
    """
    Convert revenue to a plain numeric string. If not possible, return "".
    """
    cleaned = normalize_whitespace(value)
    if is_unknown(cleaned):
        return ""
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return digits


def estimate_revenue_from_segment(segment: str | None) -> str:
    """
    Conservative fallback estimates (USD), used only when we still can't get revenue.
    """
    cleaned = normalize_whitespace(segment).lower()
    if cleaned == "startup":
        return "25000000"
    if cleaned == "sme":
        return "100000000"
    if cleaned == "mid market":
        return "500000000"
    if cleaned == "large enterprise":
        return "5000000000"
    return "50000000"


@dataclass(frozen=True)
class UnknownFixTargets:
    """
    Which columns we consider in-scope for fixing in the 'unknown fixer' pass.
    """

    website_col: str = "Website"
    description_col: str = "Description"
    hq_state_col: str = "HQ State"
    region_col: str = "Region"
    annual_revenue_col: str = "Annual Revenue"

    def all_columns(self) -> tuple[str, ...]:
        return (
            self.website_col,
            self.description_col,
            self.hq_state_col,
            self.region_col,
            self.annual_revenue_col,
        )


def any_unknown_in_columns(row: dict[str, str], columns: Iterable[str]) -> bool:
    return any(is_unknown(row.get(col)) for col in columns)



