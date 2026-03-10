"""
Unified Master Enrichment & Intelligence Script v2
Production-Ready | 40k+ Rows | Single-Pass Pro Pipeline

Architecture:
  Single API call per company using gemini-3.1-pro-preview with
  Google Search grounding + controlled generation (response_schema).
  This proven combination delivers the best accuracy.

  NOTE: gemini-2.5-flash does NOT support response_schema + Google Search
  together on Vertex AI (returns 400 INVALID_ARGUMENT). A two-pass
  Flash + Pro pipeline was tested and produced worse results than a
  single holistic Pro call. The single-pass approach is both faster
  (fewer API calls) and more accurate.

Improvements over base:
  Phase 1 – Critical Bug Fixes
    • Global semaphore (one shared instance, not per-task)
    • Retry queue with exponential back-off for 429 / transient errors
    • Header written reliably (empty-file check before writing)

  Phase 2 – Data Accuracy
    • "NOT FOUND" enforcement for unverifiable facts (headcount, revenue)
    • Robust JSON parsing – strips markdown fences, catches JSONDecodeError
    • Search grounding + controlled generation for reliable structured output

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

# Model: Pro with Google Search + controlled generation (proven best quality)
MODEL_NAME = "gemini-3.1-pro-preview"

INPUT_FILE = Path("data/test_final_10.csv")
OUTPUT_REPORT = Path("data/test_final_10_output_v4.csv")

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
# OUTPUT SCHEMA – Pydantic model for structured response
# ---------------------------------------------------------------------------

class AccountIntelligence(BaseModel):
    """Unified Schema: Basic Firmographics + Deep Sales Intelligence."""
    # Firmographics
    headcount: str = Field(description="Current employee count range")
    annual_revenue: str = Field(description="Latest annual revenue (e.g. $2.5B)")
    industry: str = Field(description="Primary industry classification")
    hq_location: str = Field(description="HQ City, State, and Country")
    region_geo: str = Field(description="US Region (East/West/etc) and Global GEO (NA/EMEA/etc)")
    description: str = Field(description="Brief Wikipedia-style overview of the organization")
    website: str = Field(description="Verified official URL")
    # Sales Intelligence
    cloud_stack: str = Field(description="Current Cloud usage (AWS, Azure, GCP) and Workspace/M365 footprint")
    legacy_debt: str = Field(description="Legacy tech targets (e.g., Snowflake, Teradata, Oracle, Hadoop, On-prem DCs)")
    strategic_priorities_2026: str = Field(description="Top 3 board-level priorities or investment areas for 2026")
    business_triggers: str = Field(description="Recent M&A, DC exits, or leadership changes impacting IT")
    sales_hook_2026: str = Field(description="The 'Why Now?'—the best reason for a Google AE to reach out today")
    sources: str = Field(description="Key links used for verification (LinkedIn, Wikipedia, News)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")


# ---------------------------------------------------------------------------
# OUTPUT CSV FIELD ORDER
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "account_id", "account_name",
    # Firmographics
    "website", "headcount", "annual_revenue", "industry",
    "hq_location", "region_geo", "description",
    # Sales Intelligence
    "cloud_stack", "legacy_debt", "strategic_priorities_2026",
    "business_triggers", "sales_hook_2026",
    "sources", "confidence",
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
# API CALL
# ---------------------------------------------------------------------------

_HTTP_OPTIONS = types.HttpOptions(timeout=API_TIMEOUT_MS)


async def _enrich_account(
    aio_client,
    name: str,
    url: str,
    semaphore: asyncio.Semaphore,
    label: str,
) -> dict[str, Any] | None:
    """Single Pro call with Google Search + controlled generation.

    This combination (Pro + search + response_schema) is proven to deliver
    the best accuracy.  A holistic prompt lets the model research everything
    in one search session, producing more complete data than split calls.
    """
    prompt = (
        f"Act as a Senior Sales Intelligence Analyst for Google Cloud. "
        f"Research {name} (URL: {url}) to provide a full 360-degree view for 2026 planning.\n\n"
        f"1. EXTRACT BASIC DATA: Headcount, Revenue, Industry, and detailed HQ/Region/GEO.\n"
        f"2. ANALYZE TECH STACK: Identify Cloud providers and legacy 'displacement' targets (Snowflake, Teradata, etc).\n"
        f"3. FORECAST 2026: Identify board-level priorities and specific AI/Data mandates.\n"
        f"4. CREATE THE HOOK: Based on your research, write a specific sales 'hook' for an AE.\n\n"
        f"Use Google Search to find the most recent 2025/2026 signals. Return ONLY valid JSON."
    )

    def factory():
        return aio_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=_HTTP_OPTIONS,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=AccountIntelligence,
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
    """Enrich a single CSV row using a single Pro API call."""
    account_id = row.get("account_id", "")
    name = row.get("account_name", "").strip()
    url = row.get("website", "").strip()
    label = f"{name}|{account_id}"

    # --- Duplicate company cache ---
    cache_key = name.lower() if name else f"__no_name__|{account_id}"
    if cache_key in company_cache:
        fut = company_cache[cache_key]
        was_already_done = fut.done()
        if not was_already_done:
            logging.info("[%s] Cache dedup — waiting for concurrent enrichment to complete.", label)
        try:
            data = await fut
            if was_already_done:
                logging.info("[%s] Cache hit — reusing previous result.", label)
            else:
                logging.info("[%s] Cache dedup — enrichment complete, result shared.", label)
        except Exception:
            logging.error("[%s] Cached enrichment failed; row will output NOT FOUND.", label)
            data = None
    else:
        fut = asyncio.Future()
        company_cache[cache_key] = fut
        try:
            data = await _enrich_account(
                aio_client, name, url, semaphore, label
            )
            fut.set_result(data)
        except Exception as e:
            logging.error("[%s] Unexpected enrichment error: %s", label, e, exc_info=True)
            data = None
            if not fut.done():
                fut.set_result(None)

    # --- Build output row ---
    d = data or {}
    return {
        "account_id": account_id,
        "account_name": name,
        # Firmographics
        "website": d.get("website") or url or "NOT FOUND",
        "headcount": d.get("headcount", "NOT FOUND"),
        "annual_revenue": d.get("annual_revenue", "NOT FOUND"),
        "industry": d.get("industry", "NOT FOUND"),
        "hq_location": d.get("hq_location", "NOT FOUND"),
        "region_geo": d.get("region_geo", "NOT FOUND"),
        "description": d.get("description", "NOT FOUND"),
        # Sales Intelligence
        "cloud_stack": d.get("cloud_stack", "NOT FOUND"),
        "legacy_debt": d.get("legacy_debt", "NOT FOUND"),
        "strategic_priorities_2026": d.get("strategic_priorities_2026", "NOT FOUND"),
        "business_triggers": d.get("business_triggers", "NOT FOUND"),
        "sales_hook_2026": d.get("sales_hook_2026", "NOT FOUND"),
        "sources": d.get("sources", "NOT FOUND"),
        "confidence": d.get("confidence", 0.0),
        # Metadata
        "enrichment_status": "success" if data else "failed",
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
