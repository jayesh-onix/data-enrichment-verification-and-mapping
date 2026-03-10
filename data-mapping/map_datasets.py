#!/usr/bin/env python3
"""
Dataset Mapping Script
======================
Maps an Onix account CSV and a Google account CSV into a single merged file.

Required columns (both files must have these):
  website, account_name

All other columns are discovered dynamically from each file and prefixed
with "onix_" or "google_" in the output.

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
  onix_<field>      - all Onix columns except website & account_name
  google_<field>    - all Google columns except website & account_name

How to run
----------
Option 1 - Edit the CONFIG block below and run:
    python3 map_datasets.py

Option 2 - Pass paths via CLI (overrides CONFIG):
    python3 map_datasets.py \\
        --onix   path/to/onix.csv   \\
        --google path/to/google.csv \\
        --output path/to/output.csv \\
        [--chunk-size 5000]
"""

# ===========================================================================
# CONFIG - Set your file paths here and just run: python3 map_datasets.py
# ===========================================================================

onix_file_path   = ""   # e.g. "data/onix_accounts.csv"
google_file_path = ""   # e.g. "data/google_accounts.csv"
output_file_path = ""   # e.g. "data/mapped_output.csv"

# Number of rows written per chunk when saving the output (default: 5000).
# Increase for faster writes on fast storage, decrease to reduce peak memory.
chunk_size = 5000

# ===========================================================================

import os
import re
import time
import logging
import argparse
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup  (timestamps + level shown on every line)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only these two columns are mandatory in both input files.
REQUIRED_COLS = {"website", "account_name"}

NOT_FOUND = "NOT FOUND"

# Legal / corporate suffixes stripped during name normalisation.
_LEGAL_RE = re.compile(
    r"\b("
    r"inc|incorporated|llc|ltd|limited|corp|corporation|co|company|"
    r"group|holdings|holding|international|intl|"
    r"technologies|technology|tech|"
    r"solutions|services|enterprises|enterprise|"
    r"global|systems|system|"
    r"partners|partnership|associates|consulting|consultants|"
    r"management|industries|industry|ventures|venture|"
    r"networks|network|labs|laboratory|laboratories"
    r")\.?",
    re.IGNORECASE,
)

# Common domain extensions that appear in account names (e.g. "Amazon.com").
_DOMAIN_EXT_RE = re.compile(
    r"\.(com|org|net|io|co|gov|edu|us|uk|biz|info)$",
    re.IGNORECASE,
)

_NON_ALNUM_RE  = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _norm_url(raw: str) -> Optional[str]:
    """
    Normalise a website URL for comparison.
    Returns None when the value is blank or 'NOT FOUND'.
    Strips protocol, leading 'www.', and trailing slashes.

    Examples:
        "https://www.Example.com/"  ->  "example.com"
        "http://Example.com"        ->  "example.com"
        "NOT FOUND"                 ->  None
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
    return s.rstrip("/") or None


def _norm_name(raw: str) -> Optional[str]:
    """
    Normalise an account name for robust comparison.

    Steps applied in order:
      1. Lowercase
      2. Strip trailing domain extension  (e.g. ".com", ".org")
      3. Strip legal / corporate suffixes (Inc, LLC, Corp, Ltd, Co, …)
      4. Strip all remaining punctuation
      5. Collapse whitespace

    Examples:
        "Amazon Inc"          ->  "amazon"
        "Amazon"              ->  "amazon"
        "Amazon.com"          ->  "amazon"
        "Golf Stix, Inc."     ->  "golf stix"
        "1-800-Flowers.com"   ->  "1 800 flowers"
        "2-1-1 San Diego"     ->  "2 1 1 san diego"
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    s = s.lower()
    s = _DOMAIN_EXT_RE.sub("", s)          # "amazon.com"   -> "amazon"
    s = _LEGAL_RE.sub(" ", s)              # "Amazon Inc"   -> "amazon "
    s = _NON_ALNUM_RE.sub(" ", s)          # strip punctuation / special chars
    s = _WHITESPACE_RE.sub(" ", s).strip() # collapse spaces
    return s if s else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(df: pd.DataFrame, label: str) -> None:
    """Raise ValueError if any required mapping column is missing."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{label}] Missing required mapping columns: {missing}. "
            f"These two columns must be present in every input file."
        )


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------


def _to_output(
    merged: pd.DataFrame,
    matched_by: str,
    onix_cols: List[str],
    google_cols: List[str],
) -> pd.DataFrame:
    """
    Assemble the final output rows from a post-merge DataFrame.

    Parameters
    ----------
    merged      : result of pd.merge() with suffixes ("_onix", "_google")
    matched_by  : "website" or "account_name"
    onix_cols   : pre-renamed onix data column names  (e.g. ["onix_account_id", ...])
    google_cols : pre-renamed google data column names (e.g. ["google_account_id", ...])
    """
    if merged.empty:
        return pd.DataFrame()

    out: dict = {"matched_by": [matched_by] * len(merged)}

    if matched_by == "website":
        # Both sides had a valid URL. Use Onix raw value as canonical.
        out["website"]      = merged["website_onix"].tolist()
        # account_name may differ between datasets; Onix is canonical.
        out["account_name"] = merged["account_name_onix"].tolist()
    else:
        # account_name was the join key; Onix value is canonical.
        out["account_name"] = merged["account_name_onix"].tolist()
        # For website: prefer whichever side has a real URL over NOT FOUND.
        onix_sites   = merged["website_onix"].tolist()
        google_sites = merged["website_google"].tolist()
        out["website"] = [
            g if o.strip().upper() == NOT_FOUND else o
            for o, g in zip(onix_sites, google_sites)
        ]

    # Add all data columns (pre-renamed, so no suffix conflicts in merged df).
    # Fill with empty string when a column is absent (e.g. column only in one dataset).
    for col in onix_cols:
        out[col] = merged[col].tolist() if col in merged.columns else [""] * len(merged)
    for col in google_cols:
        out[col] = merged[col].tolist() if col in merged.columns else [""] * len(merged)

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
    t_start = time.time()

    # ── STEP 1: Load source files ─────────────────────────────────────────────
    log.info("=" * 65)
    log.info("STEP 1/6  Loading source files")
    log.info(f"  Onix   : {onix_file}")
    onix = pd.read_csv(onix_file, dtype=str, keep_default_na=False)
    log.info(f"  Loaded {len(onix):,} rows | {len(onix.columns)} columns")

    log.info(f"  Google : {google_file}")
    google = pd.read_csv(google_file, dtype=str, keep_default_na=False)
    log.info(f"  Loaded {len(google):,} rows | {len(google.columns)} columns")

    # ── STEP 2: Validate and discover columns ─────────────────────────────────
    log.info("-" * 65)
    log.info("STEP 2/6  Validating inputs and discovering columns")

    _validate(onix,   "Onix")
    _validate(google, "Google")

    onix   = onix.reset_index(drop=True)
    google = google.reset_index(drop=True)

    # Discover data columns dynamically — everything except the two mapping keys.
    onix_data_cols   = [c for c in onix.columns   if c not in REQUIRED_COLS]
    google_data_cols = [c for c in google.columns if c not in REQUIRED_COLS]

    log.info(f"  Onix   data columns ({len(onix_data_cols)}): {onix_data_cols}")
    log.info(f"  Google data columns ({len(google_data_cols)}): {google_data_cols}")

    # Pre-rename data columns with their dataset prefix BEFORE the merge.
    # This ensures no column name collision in the merged result for data cols.
    # Only 'website' and 'account_name' will receive "_onix"/"_google" suffixes
    # from the merge since they are the only shared non-key columns left.
    onix   = onix.rename(columns={c: f"onix_{c}"   for c in onix_data_cols})
    google = google.rename(columns={c: f"google_{c}" for c in google_data_cols})

    onix_prefixed   = [f"onix_{c}"   for c in onix_data_cols]
    google_prefixed = [f"google_{c}" for c in google_data_cols]

    expected_out_cols = 3 + len(onix_prefixed) + len(google_prefixed)
    log.info(
        f"  Output schema: matched_by + website + account_name "
        f"+ {len(onix_prefixed)} onix cols + {len(google_prefixed)} google cols "
        f"= {expected_out_cols} columns total"
    )

    # ── STEP 3: Compute normalised match keys ─────────────────────────────────
    log.info("-" * 65)
    log.info("STEP 3/6  Computing normalised match keys")

    onix["_url_key"]  = onix["website"].apply(_norm_url)
    onix["_name_key"] = onix["account_name"].apply(_norm_name)
    onix["_oid"]      = onix.index   # stable row id

    google["_url_key"]  = google["website"].apply(_norm_url)
    google["_name_key"] = google["account_name"].apply(_norm_name)
    google["_gid"]      = google.index  # stable row id

    o_url_cnt  = int(onix["_url_key"].notna().sum())
    o_nurl_cnt = len(onix) - o_url_cnt
    g_url_cnt  = int(google["_url_key"].notna().sum())
    g_nurl_cnt = len(google) - g_url_cnt

    log.info(f"  Onix   - valid URL: {o_url_cnt:,}  |  NOT FOUND / blank: {o_nurl_cnt:,}")
    log.info(f"  Google - valid URL: {g_url_cnt:,}  |  NOT FOUND / blank: {g_nurl_cnt:,}")

    o_name_cnt  = int(onix["_name_key"].notna().sum())
    g_name_cnt  = int(google["_name_key"].notna().sum())
    log.info(f"  Onix   - rows with a valid account_name key: {o_name_cnt:,}")
    log.info(f"  Google - rows with a valid account_name key: {g_name_cnt:,}")

    # ── STEP 4: Primary match — website URL ───────────────────────────────────
    log.info("-" * 65)
    log.info("STEP 4/6  Primary match  (website URL)")

    o_url = onix[onix["_url_key"].notna()].copy()
    g_url = google[google["_url_key"].notna()].copy()

    # Deduplicate on URL key before merging to prevent Cartesian explosion.
    # (2 Onix rows with same URL × 2 Google rows = 4 output rows without this.)
    o_url_raw, g_url_raw = len(o_url), len(g_url)
    o_url = o_url.drop_duplicates(subset=["_url_key"], keep="first")
    g_url = g_url.drop_duplicates(subset=["_url_key"], keep="first")

    o_dropped = o_url_raw - len(o_url)
    g_dropped = g_url_raw - len(g_url)
    if o_dropped:
        log.warning(f"  Onix  : {o_dropped:,} duplicate URL row(s) dropped before primary match (kept first)")
    if g_dropped:
        log.warning(f"  Google: {g_dropped:,} duplicate URL row(s) dropped before primary match (kept first)")

    log.info(f"  Merging {len(o_url):,} Onix URL rows with {len(g_url):,} Google URL rows ...")
    primary = pd.merge(o_url, g_url, on="_url_key", suffixes=("_onix", "_google"))

    matched_o: set = set(primary["_oid"])
    matched_g: set = set(primary["_gid"])

    log.info(f"  Primary matches (website) : {len(primary):,}")

    # ── STEP 5: Secondary match — account_name fallback ───────────────────────
    log.info("-" * 65)
    log.info("STEP 5/6  Secondary match  (account_name fallback)")

    # Pool: rows NOT already matched in the primary pass.
    o_rem = onix[~onix["_oid"].isin(matched_o)].copy()
    g_rem = google[~google["_gid"].isin(matched_g)].copy()

    log.info(f"  Onix   unmatched rows available: {len(o_rem):,}")
    log.info(f"  Google unmatched rows available: {len(g_rem):,}")

    # Partition the remaining pool by URL availability.
    o_no_url  = o_rem[(o_rem["_url_key"].isna())  & (o_rem["_name_key"].notna())]
    o_has_url = o_rem[(o_rem["_url_key"].notna()) & (o_rem["_name_key"].notna())]
    g_no_url  = g_rem[(g_rem["_url_key"].isna())  & (g_rem["_name_key"].notna())]
    g_has_url = g_rem[(g_rem["_url_key"].notna()) & (g_rem["_name_key"].notna())]

    log.info(
        f"  Onix   unmatched - no URL: {len(o_no_url):,}"
        f"  |  has URL but unmatched: {len(o_has_url):,}"
    )
    log.info(
        f"  Google unmatched - no URL: {len(g_no_url):,}"
        f"  |  has URL but unmatched: {len(g_has_url):,}"
    )

    # Three sub-cases (at least ONE side must have NOT FOUND website).
    #   a) Onix NOT FOUND  <-> Google NOT FOUND
    #   b) Onix NOT FOUND  <-> Google has URL  (Google URL was unmatched in primary)
    #   c) Onix has URL    <-> Google NOT FOUND (Onix URL was unmatched in primary)
    sub_cases = [
        ("both NOT FOUND",                   o_no_url,  g_no_url),
        ("Onix NOT FOUND / Google has URL",  o_no_url,  g_has_url),
        ("Onix has URL   / Google NOT FOUND", o_has_url, g_no_url),
    ]

    sec_parts = []
    for label, o_side, g_side in sub_cases:
        if o_side.empty or g_side.empty:
            log.info(f"  Sub-case '{label}': skipped (one side is empty)")
            continue
        part = pd.merge(o_side, g_side, on="_name_key", suffixes=("_onix", "_google"))
        log.info(f"  Sub-case '{label}': {len(part):,} match(es)")
        if not part.empty:
            sec_parts.append(part)

    if sec_parts:
        secondary = pd.concat(sec_parts, ignore_index=True)
        pre_dedup = len(secondary)
        # Keep the first match per Onix row and per Google row to maintain
        # a clean 1-to-1 mapping across sub-cases.
        secondary = secondary.drop_duplicates(subset=["_oid"])
        secondary = secondary.drop_duplicates(subset=["_gid"])
        removed = pre_dedup - len(secondary)
        if removed:
            log.info(f"  Secondary dedup: removed {removed:,} cross-case duplicate(s)")
    else:
        secondary = pd.DataFrame()

    log.info(f"  Secondary matches (account_name) : {len(secondary):,}")

    # ── STEP 6: Assemble result and write output ───────────────────────────────
    log.info("-" * 65)
    log.info("STEP 6/6  Assembling output and writing to disk")

    result = pd.concat(
        [
            _to_output(primary,   "website",      onix_prefixed, google_prefixed),
            _to_output(secondary, "account_name", onix_prefixed, google_prefixed),
        ],
        ignore_index=True,
    )

    log.info(f"  Total matched rows : {len(result):,}")

    if result.empty:
        log.warning("  No matches found — output file not created.")
        return

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    first_chunk = True
    total_rows  = len(result)
    chunks_written = 0

    for start in range(0, total_rows, chunk_size):
        chunk = result.iloc[start: start + chunk_size]
        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )
        first_chunk = False
        chunks_written += 1
        written = min(start + chunk_size, total_rows)
        log.info(f"  Chunk {chunks_written} written — {written:,} / {total_rows:,} rows")

    elapsed = time.time() - t_start
    url_count  = int((result["matched_by"] == "website").sum())
    name_count = int((result["matched_by"] == "account_name").sum())

    log.info("=" * 65)
    log.info("DONE")
    log.info(f"  Output file          : {output_file}")
    log.info(f"  Total rows written   : {len(result):,}")
    log.info(f"  Matched by website   : {url_count:,}")
    log.info(f"  Matched by name      : {name_count:,}")
    log.info(f"  Output columns       : {len(result.columns)}")
    log.info(f"  Elapsed time         : {elapsed:.1f}s")
    log.info("=" * 65)


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
        log.error("The following required paths are not set:")
        for m in missing:
            log.error(f"  {m}")
        raise SystemExit(1)

    map_datasets(
        onix_file=resolved_onix,
        google_file=resolved_google,
        output_file=resolved_output,
        chunk_size=resolved_chunk,
    )


if __name__ == "__main__":
    main()
