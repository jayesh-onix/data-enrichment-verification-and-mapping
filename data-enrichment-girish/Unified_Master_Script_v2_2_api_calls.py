"""
Unified Master Enrichment & Intelligence Script v2
Production-Ready | 40k+ Rows | Two-Pass Pipeline

Improvements over v1:
  Phase 1 – Critical Bug Fixes
    • Global semaphore (one shared instance, not per-task)
    • Retry queue with exponential back-off for 429 / transient errors
    • Header written reliably (seek-to-start / empty-file check, not post-open stat)

  Phase 2 – Data Accuracy
    • "NOT FOUND" enforcement in prompts – no hallucination guesses
    • Two-pass pipeline: Flash (fast firmographics) + Pro (grounded intelligence)
    • Robust JSON parsing – strips markdown fences, catches JSONDecodeError

  Phase 3 – Performance
    • MAX_CONCURRENCY raised to 30 (tune to your quota)
    • Static batch sleep removed; rate-limit back-off via retry queue
    • In-memory company cache to skip duplicate API calls
    • Per-call asyncio timeout (120 s) to prevent hung tasks
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ID = "search-ahmed"       # GCP project
LOCATION = "global"                   # Vertex AI location

# Pass-1: fast model for verifiable firmographics
MODEL_FLASH = "gemini-2.5-flash"

# Pass-2: grounded reasoning model for sales intelligence
MODEL_PRO = "gemini-3.1-pro-preview"

INPUT_FILE = Path("data/test_final_10.csv")
OUTPUT_REPORT = Path("data/test_final_10_output_v3.csv")

# Concurrency – one global semaphore shared by ALL tasks
MAX_CONCURRENCY = 30

# Retry settings for 429 / 5xx errors
MAX_RETRIES = 5
RETRY_INITIAL_DELAY = 5.0    # seconds
RETRY_MAX_DELAY = 120.0      # seconds
RETRY_EXP_BASE = 2.0

# Per-call API timeout
API_TIMEOUT_S  = 120.0       # asyncio.wait_for timeout (seconds)
API_TIMEOUT_MS = 110_000     # httpx-layer fallback timeout (milliseconds, slightly less than above)

# ---------------------------------------------------------------------------
# OUTPUT SCHEMA – Pydantic models for structured responses
# ---------------------------------------------------------------------------

class FirmographicsData(BaseModel):
    """Pass-1: Basic, verifiable firmographic facts only."""
    headcount: str = Field(
        description=(
            "Current employee count range (e.g. '5,000–10,000'). "
            "Return 'NOT FOUND' if not verifiable from public sources."
        )
    )
    annual_revenue: str = Field(
        description=(
            "Latest annual revenue (e.g. '$2.5B'). "
            "Return 'NOT FOUND' if not publicly disclosed."
        )
    )
    industry: str = Field(
        description=(
            "Primary industry classification (e.g. 'Financial Services'). "
            "Return 'NOT FOUND' if unclear."
        )
    )
    hq_location: str = Field(
        description=(
            "HQ City, State, Country (e.g. 'Austin, TX, USA'). "
            "Return 'NOT FOUND' if not verifiable."
        )
    )
    region_geo: str = Field(
        description=(
            "US Region (e.g. 'Southeast') and Global GEO (e.g. 'NA'). "
            "Return 'NOT FOUND' if not verifiable."
        )
    )
    description: str = Field(
        description=(
            "Brief Wikipedia-style overview (2–3 sentences) of the organisation. "
            "Return 'NOT FOUND' if no reliable information exists."
        )
    )
    website: str = Field(
        description=(
            "Verified official website URL. "
            "Return 'NOT FOUND' if unverifiable."
        )
    )
    confidence: float = Field(
        description="Confidence in the above data (0.0 to 1.0)."
    )


class SalesIntelligenceData(BaseModel):
    """Pass-2: Deep sales intelligence requiring search-grounded reasoning."""
    cloud_stack: str = Field(
        description=(
            "Current cloud usage (AWS / Azure / GCP) and Workspace or M365 footprint. "
            "Return 'NOT FOUND' if not verifiable."
        )
    )
    legacy_debt: str = Field(
        description=(
            "Legacy tech displacement targets (e.g. Snowflake, Teradata, Oracle, "
            "Hadoop, on-prem DCs). You may deduce likely targets based on industry and scale if unknown."
        )
    )
    strategic_priorities_2026: str = Field(
        description=(
            "Top 3 board-level priorities or investment areas for 2026. "
            "Forecast based on industry trends if direct signals are unavailable."
        )
    )
    business_triggers: str = Field(
        description=(
            "Recent M&A, DC exits, or leadership changes impacting IT. "
            "Identify or forecast business drivers that would trigger cloud investment."
        )
    )
    sales_hook_2026: str = Field(
        description=(
            "The 'Why Now?' — the best, specific reason for a Google AE to reach out "
            "today in 2026. Use research and deductive reasoning to craft a compelling, specific hook. "
            "Do NOT return 'NOT FOUND'."
        )
    )
    sources: str = Field(
        description="Key URLs used for verification (LinkedIn, Wikipedia, news articles)."
    )
    confidence: float = Field(
        description="Confidence in the above intelligence (0.0 to 1.0)."
    )


# ---------------------------------------------------------------------------
# OUTPUT CSV FIELD ORDER
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "account_id", "account_name",
    # Pass-1 fields
    "website", "headcount", "annual_revenue", "industry",
    "hq_location", "region_geo", "description",
    "firmographics_confidence",
    # Pass-2 fields
    "cloud_stack", "legacy_debt", "strategic_priorities_2026",
    "business_triggers", "sales_hook_2026",
    "sources", "intelligence_confidence",
    # Metadata
    "enrichment_status", "enriched_at",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences that some models add around JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json(raw: str) -> dict[str, Any] | None:
    """Parse JSON safely; return None on failure."""
    try:
        return json.loads(_strip_json_fences(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("JSON parse error: %s — raw snippet: %.120s", exc, raw)
        return None


async def _call_with_retry(
    coro_factory,
    label: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """
    Execute `coro_factory()` under the global semaphore.
    Retries on 429 (rate limit) and 5xx (transient server errors) with
    exponential back-off + full jitter.  Returns None after all attempts fail.

    Key design: the semaphore is RELEASED before any sleep so that other
    tasks are not starved while this one waits out a rate-limit or backoff.
    asyncio.wait_for() is used to guarantee a real asyncio.TimeoutError is
    raised (httpx timeouts raise httpx exceptions, not asyncio.TimeoutError).
    """
    delay = RETRY_INITIAL_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        # wait_after is set to a positive number only when we should retry.
        # It is checked OUTSIDE the semaphore so the slot is not held during sleep.
        wait_after: float | None = None

        async with semaphore:
            try:
                # asyncio.wait_for raises asyncio.TimeoutError on expiry,
                # regardless of what the underlying httpx client would raise.
                response = await asyncio.wait_for(
                    coro_factory(), timeout=API_TIMEOUT_S
                )
                return _safe_json(response.text or "")
            except ClientError as exc:
                if exc.code == 429:
                    jitter = random.uniform(0, delay)
                    wait_after = min(delay + jitter, RETRY_MAX_DELAY)
                    logging.warning(
                        "[%s] Rate-limited (attempt %d/%d). Waiting %.1fs …",
                        label, attempt, MAX_RETRIES, wait_after,
                    )
                    delay = min(delay * RETRY_EXP_BASE, RETRY_MAX_DELAY)
                else:
                    logging.error("[%s] Client error (code %d): %s", label, exc.code, exc)
                    return None
            except ServerError as exc:
                jitter = random.uniform(0, delay)
                wait_after = min(delay + jitter, RETRY_MAX_DELAY)
                logging.warning(
                    "[%s] Server error %d (attempt %d/%d). Waiting %.1fs …",
                    label, exc.code, attempt, MAX_RETRIES, wait_after,
                )
                delay = min(delay * RETRY_EXP_BASE, RETRY_MAX_DELAY)
            except asyncio.TimeoutError:
                wait_after = min(delay, RETRY_MAX_DELAY)
                logging.error(
                    "[%s] API call timed out (attempt %d/%d). Waiting %.1fs …",
                    label, attempt, MAX_RETRIES, wait_after,
                )
                delay = min(delay * RETRY_EXP_BASE, RETRY_MAX_DELAY)
            except Exception as exc:
                logging.error("[%s] Unexpected error: %s", label, exc, exc_info=True)
                return None

        # Semaphore released — now safe to sleep without blocking other tasks.
        if wait_after is not None:
            await asyncio.sleep(wait_after)

    logging.error("[%s] All %d retry attempts exhausted. Skipping.", label, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# API CALLS
# ---------------------------------------------------------------------------

_HTTP_OPTIONS = types.HttpOptions(timeout=API_TIMEOUT_MS)


async def _pass1_firmographics(
    aio_client,
    name: str,
    url: str,
    semaphore: asyncio.Semaphore,
    label: str,
) -> dict[str, Any] | None:
    """Call 1 – Flash model with Google Search: extract verifiable firmographics.

    NOTE: Vertex AI does not support controlled generation (response_schema /
    response_mime_type) combined with Google Search for flash models.
    We therefore rely on prompt-based JSON formatting and parse with _safe_json().
    """
    prompt = (
        f"You are a meticulous data researcher.\n"
        f"Company: {name}\n"
        f"Provided URL: {url}\n\n"
        f"Task: Use Google Search to find and extract ONLY verifiable facts about this "
        f"company from public sources (official website, LinkedIn, Wikipedia, Crunchbase, "
        f"SEC filings, press releases).\n\n"
        f"CRITICAL RULES:\n"
        f"  - You MUST use Google Search to verify each data point.\n"
        f"  - If a value cannot be verified from a reliable public source after searching, return exactly "
        f"'NOT FOUND' — do NOT estimate or guess for headcount, revenue, or location.\n"
        f"  - For industry and description, you may synthesize based on available information.\n"
        f"  - Prefer the most recent publicly available data (2024–2026).\n"
        f"  - For website, verify the official domain via search.\n\n"
        f"Return ONLY a single valid JSON object with exactly these keys:\n"
        f'{{"headcount": "...", "annual_revenue": "...", "industry": "...", '
        f'"hq_location": "...", "region_geo": "...", "description": "...", '
        f'"website": "...", "confidence": 0.0}}\n'
        f"No markdown, no explanation — just the JSON object."
    )

    def factory():
        return aio_client.models.generate_content(
            model=MODEL_FLASH,
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=_HTTP_OPTIONS,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0,
            ),
        )

    return await _call_with_retry(factory, label, semaphore)


async def _pass2_intelligence(
    aio_client,
    name: str,
    url: str,
    firmographics: dict[str, Any],
    semaphore: asyncio.Semaphore,
    label: str,
) -> dict[str, Any] | None:
    """Call 2 – Pro model with Google Search: deep sales intelligence."""
    firm_summary = (
        f"Industry: {firmographics.get('industry', 'Unknown')}, "
        f"Headcount: {firmographics.get('headcount', 'Unknown')}, "
        f"Revenue: {firmographics.get('annual_revenue', 'Unknown')}, "
        f"HQ: {firmographics.get('hq_location', 'Unknown')}"
    )
    prompt = (
        f"You are a Senior Sales Intelligence Analyst for Google Cloud in 2026.\n"
        f"Company: {name} (URL: {url})\n"
        f"Known firmographics: {firm_summary}\n\n"
        f"Task: Using Google Search, research the CURRENT tech stack, strategic priorities, "
        f"business triggers, and craft a compelling sales hook for this company.\n\n"
        f"CRITICAL RULES:\n"
        f"  - For Tech Stack and Legacy Debt: Report signals from real sources. If none are found, you may deduce likely legacy debt targets based on their industry and scale, but clearly state it is deduced.\n"
        f"  - For Strategic Priorities and Business Triggers: Forecast their likely 2026 goals based on recent industry trends, their business model, and any recent news. If absolutely no information exists, return 'NOT FOUND'.\n"
        f"  - For the Sales Hook: You MUST write a compelling, specific reason for a Google Cloud AE to reach out today. Do NOT return 'NOT FOUND' for the sales hook; use your research and reasoning to craft the best possible hook.\n"
        f"  - Include source URLs in the 'sources' field.\n"
        f"Return ONLY valid JSON matching the required schema."
    )

    def factory():
        return aio_client.models.generate_content(
            model=MODEL_PRO,
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=_HTTP_OPTIONS,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=SalesIntelligenceData,
                temperature=0,
            ),
        )

    return await _call_with_retry(factory, label, semaphore)


# ---------------------------------------------------------------------------
# PER-ROW ENRICHMENT
# ---------------------------------------------------------------------------

async def enrich_row(
    aio_client,
    row: dict[str, str],
    semaphore: asyncio.Semaphore,
    company_cache: dict[str, asyncio.Future],
) -> dict[str, Any]:
    """Enrich a single CSV row using the two-pass pipeline."""
    account_id = row.get("account_id", "")
    name = row.get("account_name", "").strip()
    url = row.get("website", "").strip()
    label = f"{name}|{account_id}"

    # --- Duplicate company cache ---
    # Use account_id as tiebreaker for blank names to prevent cross-row cache collision.
    cache_key = name.lower() if name else f"__no_name__|{account_id}"
    if cache_key in company_cache:
        fut = company_cache[cache_key]
        # Capture done-state BEFORE awaiting; after await it is always True.
        was_already_done = fut.done()
        if not was_already_done:
            logging.info("[%s] Cache dedup — waiting for concurrent enrichment to complete.", label)
        try:
            firm_data, intel_data = await fut
            if was_already_done:
                logging.info("[%s] Cache hit — reusing previous result.", label)
            else:
                logging.info("[%s] Cache dedup — enrichment complete, result shared.", label)
        except Exception:
            logging.error("[%s] Cached enrichment failed; row will output NOT FOUND.", label)
            firm_data, intel_data = None, None
    else:
        fut = asyncio.Future()
        company_cache[cache_key] = fut
        try:
            # Pass 1 – Firmographics
            firm_data = await _pass1_firmographics(
                aio_client, name, url, semaphore, label + "|pass1"
            )

            # Pass 2 – Intelligence (only if pass-1 succeeded; use empty dict as fallback)
            intel_data = await _pass2_intelligence(
                aio_client, name, url, firm_data or {}, semaphore, label + "|pass2"
            )
            fut.set_result((firm_data, intel_data))
        except Exception as e:
            logging.error("[%s] Unexpected enrichment error: %s", label, e, exc_info=True)
            firm_data, intel_data = None, None
            # Always resolve the future so any waiting dedup tasks are unblocked.
            if not fut.done():
                fut.set_result((None, None))

    # --- Build output row ---
    enrichment_ok = bool(firm_data or intel_data)
    return {
        "account_id": account_id,
        "account_name": name,
        # Pass-1
        "website": (firm_data or {}).get("website") or url or "NOT FOUND",
        "headcount": (firm_data or {}).get("headcount", "NOT FOUND"),
        "annual_revenue": (firm_data or {}).get("annual_revenue", "NOT FOUND"),
        "industry": (firm_data or {}).get("industry", "NOT FOUND"),
        "hq_location": (firm_data or {}).get("hq_location", "NOT FOUND"),
        "region_geo": (firm_data or {}).get("region_geo", "NOT FOUND"),
        "description": (firm_data or {}).get("description", "NOT FOUND"),
        "firmographics_confidence": (firm_data or {}).get("confidence", 0.0),
        # Pass-2
        "cloud_stack": (intel_data or {}).get("cloud_stack", "NOT FOUND"),
        "legacy_debt": (intel_data or {}).get("legacy_debt", "NOT FOUND"),
        "strategic_priorities_2026": (intel_data or {}).get("strategic_priorities_2026", "NOT FOUND"),
        "business_triggers": (intel_data or {}).get("business_triggers", "NOT FOUND"),
        "sales_hook_2026": (intel_data or {}).get("sales_hook_2026", "NOT FOUND"),
        "sources": (intel_data or {}).get("sources", "NOT FOUND"),
        "intelligence_confidence": (intel_data or {}).get("confidence", 0.0),
        # Metadata
        "enrichment_status": "success" if enrichment_ok else "failed",
        "enriched_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # --- Input validation ---
    if not INPUT_FILE.exists():
        logging.error("Input file not found: %s", INPUT_FILE)
        return

    with INPUT_FILE.open(newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    if not all_rows:
        logging.error("Input file is empty: %s", INPUT_FILE)
        return

    logging.info("Loaded %d rows from %s", len(all_rows), INPUT_FILE)

    # --- Validate required input columns ---
    required_columns = {"account_id", "account_name"}
    actual_columns = set(all_rows[0].keys())
    missing_columns = required_columns - actual_columns
    if missing_columns:
        logging.error(
            "Input CSV is missing required columns: %s. Found columns: %s",
            sorted(missing_columns), sorted(actual_columns),
        )
        return

    # --- Resume support: collect already-processed account_ids ---
    done_ids: set[str] = set()
    if OUTPUT_REPORT.exists() and OUTPUT_REPORT.stat().st_size > 0:
        with OUTPUT_REPORT.open(newline="", encoding="utf-8") as f:
            done_ids = {
                r["account_id"]
                for r in csv.DictReader(f)
                if r.get("account_id") and r.get("enrichment_status") == "success"
            }
        logging.info("Resuming: %d successfully enriched rows will be skipped.", len(done_ids))

    to_process = [r for r in all_rows if r.get("account_id") not in done_ids]
    total = len(to_process)

    if total == 0:
        logging.info("All rows already processed. Nothing to do.")
        return

    logging.info("Rows to enrich: %d", total)

    # --- Shared state ---
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)   # ONE global semaphore
    company_cache: dict[str, asyncio.Future] = {}    # keyed by lowercase company name

    # --- Vertex AI client ---
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    # --- Output file: open in append mode; write header only if file is empty ---
    write_header = (not OUTPUT_REPORT.exists()) or (OUTPUT_REPORT.stat().st_size == 0)

    completed = 0
    failed = 0
    start_time = time.monotonic()

    with OUTPUT_REPORT.open("a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            out_f.flush()

        # Process using a bounded worker pool to stream results continuously
        pending = set()
        row_iter = iter(to_process)
        processed_so_far = 0

        try:
            # Start initial batch (queue size slightly larger than concurrency to keep workers busy)
            for _ in range(MAX_CONCURRENCY * 2):
                try:
                    row = next(row_iter)
                    task = asyncio.create_task(enrich_row(client.aio, row, semaphore, company_cache))
                    pending.add(task)
                except StopIteration:
                    break

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        result = task.result()
                        writer.writerow(result)
                        if result.get("enrichment_status") == "success":
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logging.error("Unexpected task exception: %s", e, exc_info=e)
                        failed += 1

                    processed_so_far += 1

                    # Top up the worker pool
                    try:
                        row = next(row_iter)
                        new_task = asyncio.create_task(enrich_row(client.aio, row, semaphore, company_cache))
                        pending.add(new_task)
                    except StopIteration:
                        pass

                out_f.flush()

                # --- Progress reporting ---
                if (processed_so_far > 0 and processed_so_far % 10 == 0) or not pending:
                    elapsed = time.monotonic() - start_time
                    rate = processed_so_far / elapsed if elapsed > 0 else 0
                    remaining = total - processed_so_far
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_str = (
                        (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).strftime("%H:%M:%S UTC")
                        if eta_seconds > 0 else "calculating…"
                    )
                    logging.info(
                        "Progress: %d/%d | Success: %d | Failed: %d | Rate: %.1f rows/s | ETA: %s",
                        len(done_ids) + processed_so_far, len(all_rows),
                        completed, failed, rate, eta_str,
                    )
        except KeyboardInterrupt:
            logging.warning("Received KeyboardInterrupt! Cancelling %d pending tasks safely...", len(pending))
            for task in pending:
                task.cancel()
            
            # Wait for all cancelled tasks to actually finish cancelling so we don't drop logs/resources
            await asyncio.gather(*pending, return_exceptions=True)
            out_f.flush()
            logging.info("Shutdown cleanly.")

    logging.info(
        "Done. Total processed: %d | Success: %d | Failed: %d | Output: %s",
        completed + failed, completed, failed, OUTPUT_REPORT,
    )


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
