"""
Unified Master Enrichment & Intelligence Script v2
Production-Ready | 40k+ Rows | Single-Pass Pipeline

Architecture:
  Single API call per company using gemini-3.1-pro-preview with
  Google Search grounding + controlled generation (response_schema).

  Why single-pass?
    • gemini-2.5-flash does NOT support response_schema + Google Search
      together on Vertex AI (returns 400 INVALID_ARGUMENT).
    • A two-pass Flash (no search) + Pro approach causes two problems:
        1. Pass-1 validates website / firmographics WITHOUT real search — unreliable.
        2. Pass-1’s unverified output biases Pass-2 when passed as “known context”.
    • A single Pro + Search call grounds EVERY field in real web data,
      eliminates the bias chain, and halves the total API calls (40k vs 80k).

  Operational features:
    • Global semaphore for quota-safe concurrency
    • Retry with exponential back-off + Retry-After header for 429 / 499 / 5xx
    • In-memory company cache to skip duplicate API calls
    • Resume support: skips already-enriched rows on restart
    • Per-call asyncio timeout (180 s) to accommodate long search-grounded calls
    • Streaming CSV writes with flush after each batch
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

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ID = "search-ahmed"       # GCP project
LOCATION = "global"                   # Vertex AI location

# Single model: Pro with Google Search + controlled generation
MODEL_NAME = "gemini-3.1-pro-preview"

INPUT_FILE = Path("data/test_final_10.csv")
OUTPUT_REPORT = Path("data/test_final_10_output_v10.csv")

# Concurrency – one global semaphore (tune to your GCP project’s Pro QPM quota)
MAX_CONCURRENCY = 30

# How many asyncio tasks to pre-spawn into the pending queue.
INITIAL_QUEUE_DEPTH = MAX_CONCURRENCY * 2

# Retry settings for 429 / 5xx errors
MAX_RETRIES = 5
RETRY_INITIAL_DELAY = 5.0    # seconds
RETRY_MAX_DELAY = 120.0      # seconds
RETRY_EXP_BASE = 2.0

# Per-call API timeout
# Pro + Google Search with AFC can legitimately take 60–90s for complex
# companies (multiple search rounds).  Set generous timeouts so the client
# does NOT cancel in-flight requests (which causes 499 CANCELLED errors).
API_TIMEOUT_S  = 180.0       # asyncio.wait_for timeout (seconds)
API_TIMEOUT_MS = 170_000     # httpx-layer fallback timeout (milliseconds, slightly less)

# ---------------------------------------------------------------------------
# OUTPUT SCHEMA – Unified Pydantic model for structured response
# ---------------------------------------------------------------------------

class AccountEnrichment(BaseModel):
    """Unified schema: website validation + firmographics + sales intelligence."""
    # Website validation & account name
    website: str = Field(
        description=(
            "Verified official website URL for this company. "
            "Return 'NOT FOUND' if unverifiable."
        )
    )
    correct_account_name: str = Field(
        description=(
            "The correct official company name as found on the verified website. "
            "If the provided website was invalid, write the correct company name. "
            "If valid, keep the original account name unchanged."
        )
    )
    # Firmographics
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
    region: str = Field(
        description=(
            "US Region. Use ONLY: 'US East', 'US West', 'US North', 'US South', or 'US Central'. "
            "Return 'NOT FOUND' if the company is non-US or sub-region is unclear."
        )
    )
    geo: str = Field(
        description=(
            "Global Geography. Use ONLY: 'NA', 'LATAM', 'EMEA', 'APAC', or 'ROW'. "
            "Return 'NOT FOUND' if unverifiable."
        )
    )
    description: str = Field(
        description=(
            "Brief Wikipedia-style overview (2–3 sentences) of the organisation. "
            "Return 'NOT FOUND' if no reliable information exists."
        )
    )
    # Sales intelligence
    cloud_stack: str = Field(
        description=(
            "Current cloud usage (AWS / Azure / GCP) and Workspace or M365 footprint. "
            "Return 'NOT FOUND' if not verifiable."
        )
    )
    legacy_debt: str = Field(
        description=(
            "Legacy tech displacement targets (e.g. Snowflake, Teradata, Oracle, "
            "Hadoop, on-prem DCs). Return 'NOT FOUND' if unknown."
        )
    )
    strategic_priorities_2026: str = Field(
        description=(
            "Top 3 board-level priorities or investment areas for 2026. "
            "Return 'NOT FOUND' if no reliable signals exist."
        )
    )
    business_triggers: str = Field(
        description=(
            "Recent M&A, DC exits, or leadership changes impacting IT. "
            "Return 'NOT FOUND' if no recent events found."
        )
    )
    sales_hook_2026: str = Field(
        description=(
            "The 'Why Now?' — the best, specific reason for a Google AE to reach out "
            "today in 2026. Must be grounded in real signals. "
            "Return 'NOT FOUND' if no strong hook is identified."
        )
    )
    sources: str = Field(
        description="Key URLs used for verification (LinkedIn, Wikipedia, news articles)."
    )
    confidence: float = Field(
        description="Overall confidence in the enrichment data (0.0 to 1.0)."
    )


# ---------------------------------------------------------------------------
# OUTPUT CSV FIELD ORDER
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "account_id", "account_name",
    # Website validation
    "website", "Correct_Account_Name",
    # Firmographics
    "headcount", "annual_revenue", "industry",
    "hq_location", "region", "geo", "description",
    # Sales intelligence
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
    Retries on 429, 499, 5xx, and transient timeout errors with
    exponential back-off + full jitter.  Returns None after all attempts fail.

    Key design: the semaphore is RELEASED before any sleep so that other
    tasks are not starved while this one waits out a rate-limit or backoff.
    """
    delay = RETRY_INITIAL_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        wait_after: float | None = None

        async with semaphore:
            try:
                response = await asyncio.wait_for(
                    coro_factory(), timeout=API_TIMEOUT_S
                )
                return _safe_json(response.text or "")
            except ClientError as exc:
                if exc.code in (429, 499):
                    # 429 = rate-limited; 499 = the server or our asyncio.wait_for
                    # cancelled the request (Client Closed Request).  Both are transient.
                    retry_after_hdr: str | None = None
                    if exc.code == 429 and exc.response is not None and isinstance(
                        exc.response, httpx.Response
                    ):
                        retry_after_hdr = exc.response.headers.get("retry-after")
                    if retry_after_hdr is not None:
                        try:
                            wait_after = min(float(retry_after_hdr), RETRY_MAX_DELAY)
                        except ValueError:
                            jitter = random.uniform(0, delay)
                            wait_after = min(delay + jitter, RETRY_MAX_DELAY)
                    else:
                        jitter = random.uniform(0, delay)
                        wait_after = min(delay + jitter, RETRY_MAX_DELAY)
                    logging.warning(
                        "[%s] %s (attempt %d/%d). Waiting %.1fs …%s",
                        label,
                        "Rate-limited" if exc.code == 429 else "Request cancelled (499)",
                        attempt, MAX_RETRIES, wait_after,
                        " (from Retry-After header)" if retry_after_hdr else "",
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
            except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
                wait_after = min(delay, RETRY_MAX_DELAY)
                logging.warning(
                    "[%s] Timeout (%s, attempt %d/%d). Waiting %.1fs …",
                    label, type(exc).__name__, attempt, MAX_RETRIES, wait_after,
                )
                delay = min(delay * RETRY_EXP_BASE, RETRY_MAX_DELAY)
            except Exception as exc:
                logging.error("[%s] Unexpected error: %s", label, exc, exc_info=True)
                return None

        # Semaphore released — sleep without blocking other tasks.
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

    A holistic prompt lets the model research everything in one search session.
    Website validation, firmographics, and sales intelligence are all
    grounded in real web data — no blind guesses.
    """
    prompt = (
        f"You are a Senior Sales Intelligence Analyst for Google Cloud.\n"
        f"Company: {name!r}\n"
        f"Provided URL: {url!r}\n\n"
        f"STEP 1 – WEBSITE VALIDATION (do this first):\n"
        f"  Use Google Search to determine whether the Provided URL is the real,\n"
        f"  official website for '{name}'.\n"
        f"  • If valid: set 'website' to the provided URL, set 'correct_account_name'\n"
        f"    to the original account name (unchanged).\n"
        f"  • If invalid or belongs to a different entity: set 'website' to the correct\n"
        f"    verified official URL, set 'correct_account_name' to the correct official\n"
        f"    company name.\n"
        f"  • If unverifiable: set both to 'NOT FOUND'.\n\n"
        f"STEP 2 – FIRMOGRAPHIC DATA:\n"
        f"  Extract verifiable facts from public sources (Wikipedia, official IR pages,\n"
        f"  LinkedIn, Crunchbase, Bloomberg). Use the most recent data (prefer 2024–2026).\n\n"
        f"STEP 3 – SALES INTELLIGENCE:\n"
        f"  Research the CURRENT tech stack, strategic priorities, business triggers,\n"
        f"  and create a specific sales hook for a Google Cloud AE in 2026.\n\n"
        f"FIELD CONSTRAINTS:\n"
        f"  'region'  – US-headquartered companies ONLY. Choose EXACTLY one of:\n"
        f"              'US East', 'US West', 'US North', 'US South', 'US Central'.\n"
        f"              Return 'NOT FOUND' if non-US or sub-region is unclear.\n"
        f"  'geo'     – Choose EXACTLY one of: 'NA', 'LATAM', 'EMEA', 'APAC', 'ROW'.\n"
        f"              Must always be populated if the company's country is known.\n"
        f"  All other fields: return exactly 'NOT FOUND' if not verifiable — no guesses.\n"
        f"  Include source URLs in the 'sources' field.\n"
        f"  The sales_hook_2026 must be grounded in a specific, real signal — not generic.\n\n"
        f"Return ONLY valid JSON matching the required schema."
    )

    def factory():
        return aio_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=_HTTP_OPTIONS,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=AccountEnrichment,
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
    """Enrich a single CSV row using a single Pro + Search API call."""
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
        data = None
        try:
            data = await _enrich_account(aio_client, name, url, semaphore, label)
        except Exception as e:
            logging.error("[%s] Unexpected enrichment error: %s", label, e, exc_info=True)
        finally:
            # Guarantee the future always resolves so dedup waiters are never blocked.
            if not fut.done():
                fut.set_result(data)

    # --- Build output row ---
    d = data or {}
    return {
        "account_id": account_id,
        "account_name": name,
        # Website validation
        "website": d.get("website") or url or "NOT FOUND",
        "Correct_Account_Name": d.get("correct_account_name") or name or "NOT FOUND",
        # Firmographics
        "headcount": d.get("headcount", "NOT FOUND"),
        "annual_revenue": d.get("annual_revenue", "NOT FOUND"),
        "industry": d.get("industry", "NOT FOUND"),
        "hq_location": d.get("hq_location", "NOT FOUND"),
        "region": d.get("region", "NOT FOUND"),
        "geo": d.get("geo", "NOT FOUND"),
        "description": d.get("description", "NOT FOUND"),
        # Sales intelligence
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
    # Suppress noisy per-request HTTP and AFC logs from the genai SDK.
    # Our own retry/progress logging provides all needed operational visibility.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)

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
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)       # ONE global semaphore
    company_cache: dict[str, asyncio.Future] = {}        # dedup by company name

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
            for _ in range(INITIAL_QUEUE_DEPTH):
                try:
                    row = next(row_iter)
                    task = asyncio.create_task(
                        enrich_row(client.aio, row, semaphore, company_cache)
                    )
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
                        new_task = asyncio.create_task(
                            enrich_row(client.aio, row, semaphore, company_cache)
                        )
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
