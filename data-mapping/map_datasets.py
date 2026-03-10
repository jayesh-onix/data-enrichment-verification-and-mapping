#!/usr/bin/env python3
"""
Dataset Mapping Script
======================
Maps an Onix account CSV and a Google account CSV into a single merged file.

Matching strategy
-----------------
1. Primary   - website URL (when BOTH sides have a non-"NOT FOUND" URL).
2. Secondary - account_name (when at least ONE side has website = "NOT FOUND"
               and the row was not already matched in the primary pass).

Only matched rows are written to the output (inner-join semantics).

Output schema
-------------
  matched_by        - 'website' or 'account_name'
  website           - canonical URL  (shared, not duplicated)
  account_name      - canonical name (shared, not duplicated)
  onix_<field>      - 18 Onix-specific columns (all except website & account_name)
  google_<field>    - 18 Google-specific columns (all except website & account_name)

Total: 3 header cols + 18 onix cols + 18 google cols = 39 columns.

How to run
----------
Option 1 – Edit the CONFIG block below and run:
    python3 map_datasets.py

Option 2 – Pass paths via CLI (overrides CONFIG):
    python3 map_datasets.py \\
        --onix   path/to/onix.csv   \\
        --google path/to/google.csv \\
        --output path/to/output.csv \\
        [--chunk-size 5000]
"""

# ===========================================================================
# CONFIG – Set your file paths here and just run: python3 map_datasets.py
# ===========================================================================

onix_file_path   = ""   # e.g. "data/onix_accounts.csv"
google_file_path = ""   # e.g. "data/google_accounts.csv"
output_file_path = ""   # e.g. "data/mapped_output.csv"

# Number of rows written per chunk when saving the output (default: 5000).
# Increase for faster writes on fast storage, decrease to reduce peak memory.
chunk_size = 5000

# ===========================================================================

import os
import argparse
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "account_id",
    "account_name",
    "website",
    "Correct_Account_Name",
    "headcount",
    "annual_revenue",
    "industry",
    "hq_location",
    "region",
    "geo",
    "description",
    "cloud_stack",
    "legacy_debt",
    "strategic_priorities_2026",
    "business_triggers",
    "sales_hook_2026",
    "sources",
    "confidence",
    "enrichment_status",
    "enriched_at",
]

# These two appear once (no prefix) in the output – they are the mapping keys.
SHARED_COLS = ["website", "account_name"]

# All remaining columns get an onix_ / google_ prefix.
DATA_COLS = [c for c in EXPECTED_COLUMNS if c not in SHARED_COLS]

NOT_FOUND = "NOT FOUND"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _norm_url(raw: str) -> Optional[str]:
    """
    Normalise a website URL for comparison.
    Returns None when the value is blank or 'NOT FOUND'.
    Strips protocol, leading 'www.', and trailing slashes.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.upper() == NOT_FOUND:
        return None
    s = s.lower()
    for scheme in ("https://", "http://"):
        if s.startswith(scheme):
            s = s[len(scheme):]
            break
    if s.startswith("www."):
        s = s[4:]
    return s.rstrip("/")


def _norm_name(raw: str) -> Optional[str]:
    """
    Normalise an account name for comparison.
    Returns None when the value is blank.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    return s.lower() if s else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(df: pd.DataFrame, label: str) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------


def _to_output(merged: pd.DataFrame, matched_by: str) -> pd.DataFrame:
    """
    Convert a merged DataFrame (with _onix / _google column suffixes) into
    the standardised output format.
    """
    if merged.empty:
        return pd.DataFrame()

    out: dict = {"matched_by": [matched_by] * len(merged)}

    if matched_by == "website":
        # Both sides had a valid URL.  Use the Onix raw value as canonical.
        out["website"] = merged["website_onix"].tolist()
        # account_name may differ; use Onix value as canonical.
        out["account_name"] = merged["account_name_onix"].tolist()
    else:
        # account_name was the join key; Onix value is canonical.
        out["account_name"] = merged["account_name_onix"].tolist()
        # For website: prefer a real URL over NOT FOUND.
        onix_sites = merged["website_onix"].tolist()
        google_sites = merged["website_google"].tolist()
        out["website"] = [
            g if o.strip().upper() == NOT_FOUND else o
            for o, g in zip(onix_sites, google_sites)
        ]

    for col in DATA_COLS:
        out[f"onix_{col}"] = merged[f"{col}_onix"].tolist()
        out[f"google_{col}"] = merged[f"{col}_google"].tolist()

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Core mapping logic
# ---------------------------------------------------------------------------


def map_datasets(
    onix_file: str,
    google_file: str,
    output_file: str,
    chunk_size: int = 5000,
) -> None:
    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading Onix   : {onix_file}")
    onix = pd.read_csv(onix_file, dtype=str, keep_default_na=False)

    print(f"Loading Google : {google_file}")
    google = pd.read_csv(google_file, dtype=str, keep_default_na=False)

    _validate(onix, "Onix")
    _validate(google, "Google")

    print(f"Onix rows: {len(onix):,}  |  Google rows: {len(google):,}")

    # ── Pre-process ───────────────────────────────────────────────────────────
    onix = onix.reset_index(drop=True)
    google = google.reset_index(drop=True)

    # Normalised keys for matching
    onix["_url_key"] = onix["website"].apply(_norm_url)
    onix["_name_key"] = onix["account_name"].apply(_norm_name)
    onix["_oid"] = onix.index  # unique row id (used to track matched rows)

    google["_url_key"] = google["website"].apply(_norm_url)
    google["_name_key"] = google["account_name"].apply(_norm_name)
    google["_gid"] = google.index  # unique row id

    # ── 1. Primary match: website URL ────────────────────────────────────────
    o_url = onix[onix["_url_key"].notna()].copy()
    g_url = google[google["_url_key"].notna()].copy()

    primary = pd.merge(
        o_url,
        g_url,
        on="_url_key",
        suffixes=("_onix", "_google"),
    )

    # _oid comes only from the left df; _gid only from the right → no suffix conflict.
    matched_o: set = set(primary["_oid"])
    matched_g: set = set(primary["_gid"])

    print(f"Primary matches   (website)      : {len(primary):,}")

    # ── 2. Secondary match: account_name ─────────────────────────────────────
    # Use only rows that were NOT already matched in the primary pass.
    o_rem = onix[~onix["_oid"].isin(matched_o)].copy()
    g_rem = google[~google["_gid"].isin(matched_g)].copy()

    # Partition remaining rows by whether they have a valid URL.
    # Both sides of a secondary join must have a valid name key.
    o_no_url = o_rem[(o_rem["_url_key"].isna()) & (o_rem["_name_key"].notna())]
    o_has_url = o_rem[(o_rem["_url_key"].notna()) & (o_rem["_name_key"].notna())]
    g_no_url = g_rem[(g_rem["_url_key"].isna()) & (g_rem["_name_key"].notna())]
    g_has_url = g_rem[(g_rem["_url_key"].notna()) & (g_rem["_name_key"].notna())]

    # Three sub-cases (at least ONE side must have NOT FOUND website):
    #   a) onix NOT FOUND  <-> google NOT FOUND
    #   b) onix NOT FOUND  <-> google has URL  (google URL was unmatched)
    #   c) onix has URL    <-> google NOT FOUND (onix URL was unmatched)
    sec_parts = []
    for o_side, g_side in [
        (o_no_url, g_no_url),
        (o_no_url, g_has_url),
        (o_has_url, g_no_url),
    ]:
        if o_side.empty or g_side.empty:
            continue
        part = pd.merge(
            o_side,
            g_side,
            on="_name_key",
            suffixes=("_onix", "_google"),
        )
        if not part.empty:
            sec_parts.append(part)

    if sec_parts:
        secondary = pd.concat(sec_parts, ignore_index=True)
        # If a row matched in multiple sub-cases, keep the first hit only
        # to maintain a clean 1-to-1 mapping.
        secondary = secondary.drop_duplicates(subset=["_oid"])
        secondary = secondary.drop_duplicates(subset=["_gid"])
    else:
        secondary = pd.DataFrame()

    print(f"Secondary matches (account_name) : {len(secondary):,}")

    # ── 3. Build the final output ─────────────────────────────────────────────
    result = pd.concat(
        [_to_output(primary, "website"), _to_output(secondary, "account_name")],
        ignore_index=True,
    )

    print(f"Total matched rows : {len(result):,}")

    if result.empty:
        print("No matches found – output file not created.")
        return

    # ── 4. Write output in chunks ─────────────────────────────────────────────
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    first_chunk = True
    total_rows = len(result)
    for start in range(0, total_rows, chunk_size):
        chunk = result.iloc[start : start + chunk_size]
        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )
        first_chunk = False
        written = min(start + chunk_size, total_rows)
        print(f"  Written {written:,} / {total_rows:,} rows …", end="\r")

    print()  # newline after progress

    # ── 5. Summary ────────────────────────────────────────────────────────────
    url_count = int((result["matched_by"] == "website").sum())
    name_count = int((result["matched_by"] == "account_name").sum())

    print(f"\nOutput written to : {output_file}")
    print(f"\nMatch summary:")
    print(f"  Matched by website      : {url_count:,}")
    print(f"  Matched by account_name : {name_count:,}")
    print(f"  Total                   : {len(result):,}")
    print(f"\nOutput columns ({len(result.columns)}): {list(result.columns)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map Onix and Google account datasets into a single merged CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--onix",
        required=False,
        default="",
        metavar="FILE",
        help="Path to the Onix source CSV file. Overrides onix_file_path in CONFIG.",
    )
    parser.add_argument(
        "--google",
        required=False,
        default="",
        metavar="FILE",
        help="Path to the Google source CSV file. Overrides google_file_path in CONFIG.",
    )
    parser.add_argument(
        "--output",
        required=False,
        default="",
        metavar="FILE",
        help="Path for the output mapped CSV file. Overrides output_file_path in CONFIG.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        metavar="N",
        help="Number of rows written per chunk (default: 5000).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    # CONFIG values take priority when set; fall back to CLI args otherwise.
    resolved_onix   = onix_file_path.strip()   or args.onix
    resolved_google = google_file_path.strip() or args.google
    resolved_output = output_file_path.strip() or args.output
    resolved_chunk  = chunk_size if chunk_size != 5000 else args.chunk_size

    # Validate that all required paths are available.
    missing = []
    if not resolved_onix:   missing.append("--onix   (or set onix_file_path in CONFIG)")
    if not resolved_google: missing.append("--google (or set google_file_path in CONFIG)")
    if not resolved_output: missing.append("--output (or set output_file_path in CONFIG)")
    if missing:
        print("ERROR: The following required paths are not set:")
        for m in missing:
            print(f"  {m}")
        raise SystemExit(1)

    map_datasets(
        onix_file=resolved_onix,
        google_file=resolved_google,
        output_file=resolved_output,
        chunk_size=resolved_chunk,
    )


if __name__ == "__main__":
    main()
