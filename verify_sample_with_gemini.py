"""
Sample-based verification of enriched account data using Gemini + Google Search grounding.

Goal:
- Provide a statistically defensible sample size for a given confidence level
  (default: 95%) and margin of error (default: 5%).
- For each sampled row, ask Gemini to verify key fields against web evidence.
- Output a CSV report with per-row pass/fail and suggested corrections.

Scope of verification (as requested):
- Website
- Description
- HQ State
- Plus additional columns: Region, Industry, Annual Revenue, Segment (light check)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from account_enrichment_common import (
    UnknownFixTargets,
    is_unknown,
    normalize_region,
    normalize_whitespace,
)


PROJECT_ID = "search-ahmed"
LOCATION = "global"
MODEL_NAME = "gemini-2.5-flash"

DEFAULT_INPUT = Path("onix_enriched_data_trimmed.csv")
DEFAULT_REPORT = Path("verification_sample_report.csv")

TARGETS = UnknownFixTargets()


class VerificationResponse(BaseModel):
    """
    Model-driven verification response for one row.
    """

    website_correct: bool = Field(description="Is the provided website the official primary website?")
    hq_state_correct: bool = Field(description="Is the HQ State correct?")
    description_correct: bool = Field(description="Does the description accurately describe the company?")
    region_correct: bool = Field(description="Is the region consistent with HQ State/country?")
    industry_reasonable: bool = Field(description="Is the industry classification reasonable?")
    annual_revenue_reasonable: bool = Field(description="Is the annual revenue plausible/reasonable?")

    corrected_website: str | None = Field(default=None, description="If incorrect, the official website to use")
    corrected_hq_state: str | None = Field(default=None, description="If incorrect, corrected full state/province")
    corrected_region: str | None = Field(default=None, description="If incorrect, corrected region")
    notes: str = Field(description="Brief evidence-based notes (include sources if possible)")


def z_value_for_confidence(confidence: float) -> float:
    """
    Z values for common confidence levels.
    We intentionally keep this table-based to avoid pulling heavy dependencies.
    """
    confidence_to_z = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    if confidence in confidence_to_z:
        return confidence_to_z[confidence]

    raise ValueError(
        "Unsupported confidence level. Use one of: 0.90, 0.95, 0.99"
    )


def sample_size_for_proportion(
    *,
    population_size: int,
    confidence: float,
    margin_of_error: float,
    assumed_proportion: float = 0.5,
) -> int:
    """
    Standard sample size for estimating a proportion with finite population correction.

    - assumed_proportion=0.5 yields the largest (most conservative) sample size.
    """
    if population_size <= 0:
        raise ValueError("population_size must be > 0")
    if not (0 < margin_of_error < 1):
        raise ValueError("margin_of_error must be between 0 and 1")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1")

    z = z_value_for_confidence(confidence)
    p = assumed_proportion

    n0 = (z * z * p * (1 - p)) / (margin_of_error * margin_of_error)
    # Finite population correction
    n = n0 / (1 + ((n0 - 1) / population_size))
    return max(1, math.ceil(n))


def build_verification_prompt(row: dict[str, str]) -> str:
    account_name = normalize_whitespace(row.get("Account Name"))

    website = normalize_whitespace(row.get(TARGETS.website_col))
    description = normalize_whitespace(row.get(TARGETS.description_col))
    hq_state = normalize_whitespace(row.get(TARGETS.hq_state_col))
    region = normalize_region(row.get(TARGETS.region_col))
    industry = normalize_whitespace(row.get("Industry"))
    annual_revenue = normalize_whitespace(row.get(TARGETS.annual_revenue_col))
    segment = normalize_whitespace(row.get("Segment"))

    return f"""You are a data quality auditor with access to Google Search.
Verify the provided company fields against web evidence (official site, LinkedIn, reputable sources).

Company:
- Account Name: {account_name}

Provided fields to verify:
- Website: {website}
- HQ State: {hq_state}
- Region: {region}
- Industry: {industry}
- Annual Revenue (USD digits): {annual_revenue}
- Segment: {segment}
- Description: {description}

Rules:
1) Use Google Search grounding. Prefer official website and LinkedIn company page.
2) Mark each field as correct/incorrect/reasonable using booleans in the JSON schema.
3) If a field is incorrect, provide a corrected value (when possible).
4) If uncertain, be conservative: set the boolean to false and explain why in notes.
5) Notes must be short and evidence-based; include sources if you can.

Return ONLY valid JSON matching the schema.
"""


async def call_gemini_verify(
    *,
    client: genai.Client,
    prompt: str,
) -> dict[str, Any]:
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        response_mime_type="application/json",
        response_schema=VerificationResponse,
        temperature=0,
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )
    return json.loads(response.text or "{}")


@dataclass(frozen=True)
class VerificationRowResult:
    account_id: str
    account_name: str
    website: str
    hq_state: str
    region: str
    description: str
    website_correct: bool
    hq_state_correct: bool
    description_correct: bool
    region_correct: bool
    industry_reasonable: bool
    annual_revenue_reasonable: bool
    corrected_website: str
    corrected_hq_state: str
    corrected_region: str
    notes: str


async def verify_rows(
    *,
    rows: list[dict[str, str]],
    max_concurrency: int,
) -> list[VerificationRowResult]:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def verify_one(row: dict[str, str]) -> VerificationRowResult:
        prompt = build_verification_prompt(row)
        async with semaphore:
            verdict = await call_gemini_verify(client=client, prompt=prompt)

        account_id = normalize_whitespace(row.get("Account ID 18 Digit"))
        account_name = normalize_whitespace(row.get("Account Name"))

        return VerificationRowResult(
            account_id=account_id,
            account_name=account_name,
            website=normalize_whitespace(row.get(TARGETS.website_col)),
            hq_state=normalize_whitespace(row.get(TARGETS.hq_state_col)),
            region=normalize_region(row.get(TARGETS.region_col)),
            description=normalize_whitespace(row.get(TARGETS.description_col)),
            website_correct=bool(verdict.get("website_correct", False)),
            hq_state_correct=bool(verdict.get("hq_state_correct", False)),
            description_correct=bool(verdict.get("description_correct", False)),
            region_correct=bool(verdict.get("region_correct", False)),
            industry_reasonable=bool(verdict.get("industry_reasonable", False)),
            annual_revenue_reasonable=bool(verdict.get("annual_revenue_reasonable", False)),
            corrected_website=normalize_whitespace(verdict.get("corrected_website")) or "",
            corrected_hq_state=normalize_whitespace(verdict.get("corrected_hq_state")) or "",
            corrected_region=normalize_whitespace(verdict.get("corrected_region")) or "",
            notes=normalize_whitespace(verdict.get("notes")) or "",
        )

    tasks = [verify_one(row) for row in rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    finalized: list[VerificationRowResult] = []
    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            finalized.append(
                VerificationRowResult(
                    account_id=normalize_whitespace(row.get("Account ID 18 Digit")),
                    account_name=normalize_whitespace(row.get("Account Name")),
                    website=normalize_whitespace(row.get(TARGETS.website_col)),
                    hq_state=normalize_whitespace(row.get(TARGETS.hq_state_col)),
                    region=normalize_region(row.get(TARGETS.region_col)),
                    description=normalize_whitespace(row.get(TARGETS.description_col)),
                    website_correct=False,
                    hq_state_correct=False,
                    description_correct=False,
                    region_correct=False,
                    industry_reasonable=False,
                    annual_revenue_reasonable=False,
                    corrected_website="",
                    corrected_hq_state="",
                    corrected_region="",
                    notes=f"Verification failed with exception: {result}",
                )
            )
        else:
            finalized.append(result)

    return finalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.90, 0.95, 0.99])
    parser.add_argument("--margin-of-error", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-concurrency", type=int, default=5)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    population_size = len(rows)
    n = sample_size_for_proportion(
        population_size=population_size,
        confidence=args.confidence,
        margin_of_error=args.margin_of_error,
        assumed_proportion=0.5,
    )

    rng = random.Random(args.seed)
    sample = rng.sample(rows, k=min(n, population_size))

    # Guard: we should not be verifying unknowns at this stage.
    unknowns_in_sample = sum(
        1
        for row in sample
        if is_unknown(row.get(TARGETS.website_col))
        or is_unknown(row.get(TARGETS.description_col))
        or is_unknown(row.get(TARGETS.hq_state_col))
    )
    if unknowns_in_sample > 0:
        logging.warning(
            "Sample still contains %d rows with unknown Website/Description/HQ State. "
            "You may want to run fix_unknowns.py again before verifying.",
            unknowns_in_sample,
        )

    logging.info(
        "Sampling %d/%d rows (confidence=%.2f, margin_of_error=%.2f, seed=%d)",
        len(sample),
        population_size,
        args.confidence,
        args.margin_of_error,
        args.seed,
    )

    start = time.time()
    results = await verify_rows(rows=sample, max_concurrency=args.max_concurrency)
    elapsed = time.time() - start

    fieldnames = [
        "Account ID 18 Digit",
        "Account Name",
        "Website",
        "HQ State",
        "Region",
        "Description",
        "website_correct",
        "hq_state_correct",
        "description_correct",
        "region_correct",
        "industry_reasonable",
        "annual_revenue_reasonable",
        "corrected_website",
        "corrected_hq_state",
        "corrected_region",
        "notes",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "Account ID 18 Digit": r.account_id,
                    "Account Name": r.account_name,
                    "Website": r.website,
                    "HQ State": r.hq_state,
                    "Region": r.region,
                    "Description": r.description,
                    "website_correct": r.website_correct,
                    "hq_state_correct": r.hq_state_correct,
                    "description_correct": r.description_correct,
                    "region_correct": r.region_correct,
                    "industry_reasonable": r.industry_reasonable,
                    "annual_revenue_reasonable": r.annual_revenue_reasonable,
                    "corrected_website": r.corrected_website,
                    "corrected_hq_state": r.corrected_hq_state,
                    "corrected_region": r.corrected_region,
                    "notes": r.notes,
                }
            )

    failed = sum(
        1
        for r in results
        if not (r.website_correct and r.hq_state_correct and r.description_correct)
    )
    logging.info(
        "✅ Verification complete in %.2fs. Report: %s. Failures (website+state+desc): %d/%d",
        elapsed,
        args.output,
        failed,
        len(results),
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()


