"""
Unified Master Enrichment & Intelligence Script v4
Combines: Basic Firmographics + Deep Sales Intelligence + 2026 Strategic Hooks.
Target: Google Data Enrichsing first lot - Sheet1.csv
"""

from __future__ import annotations
import asyncio
import csv
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
PROJECT_ID = "search-ahmed"  # GCP project for Vertex AI access
LOCATION = "global"
MODEL_NAME = "gemini-3.1-pro-preview"

INPUT_FILE = Path("data/test_final_10.csv")
OUTPUT_REPORT = Path("data/test_final_10_output_v1.csv")

# Performance Settings (Adjust to your Quota)
MAX_CONCURRENCY = 5 
BATCH_SLEEP = 1.0

class FullAccountIntelligence(BaseModel):
    """Unified Schema: Basic Data + Deep Intelligence"""
    # Section 1: Basic Firmographics
    headcount: str = Field(description="Current employee count range")
    annual_revenue: str = Field(description="Latest annual revenue (e.g. $2.5B)")
    industry: str = Field(description="Primary industry classification")
    hq_location: str = Field(description="HQ City, State, and Country")
    region_geo: str = Field(description="US Region (East/West/etc) and Global GEO (NA/EMEA/etc)")
    description: str = Field(description="Brief Wikipedia-style overview of the organization")
    
    # Section 2: Deep Sales Intelligence
    cloud_stack: str = Field(description="Current Cloud usage (AWS, Azure, GCP) and Workspace/M365 footprint")
    legacy_debt: str = Field(description="Legacy tech targets (e.g., Snowflake, Teradata, Oracle, Hadoop, On-prem DCs)")
    strategic_priorities_2026: str = Field(description="Top 3 board-level priorities or investment areas for 2026")
    business_triggers: str = Field(description="Recent M&A, DC exits, or leadership changes impacting IT")
    sales_hook_2026: str = Field(description="The 'Why Now?'—the best reason for a Google AE to reach out today")
    
    # Section 3: Metadata
    website: str = Field(description="Verified official URL")
    sources: str = Field(description="Key links used for verification (LinkedIn, Wikipedia, News)")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")

async def call_gemini_master(async_client, name, url):
    prompt = f"""Act as a Senior Sales Intelligence Analyst for Google Cloud. 
    Research {name} (URL: {url}) to provide a full 360-degree view for 2026 planning.
    
    1. EXTRACT BASIC DATA: Headcount, Revenue, Industry, and detailed HQ/Region/GEO.
    2. ANALYZE TECH STACK: Identify Cloud providers and legacy 'displacement' targets (Snowflake, Teradata, etc).
    3. FORECAST 2026: Identify board-level priorities and specific AI/Data mandates.
    4. CREATE THE HOOK: Based on your research, write a specific sales 'hook' for an AE.
    
    Use Google Search to find the most recent 2025/2026 signals. Return ONLY valid JSON."""
    
    try:
        response = await async_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                response_mime_type="application/json",
                response_schema=FullAccountIntelligence,
                temperature=0,
            ),
        )
        return json.loads(response.text or "{}")
    except Exception as e:
        if "429" in str(e): return "RATE_LIMIT"
        logging.error(f"Error {name}: {e}")
        return None

async def async_main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    if not INPUT_FILE.exists():
        logging.error(f"File {INPUT_FILE} not found!")
        return

    with INPUT_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done_ids = set()
    if OUTPUT_REPORT.exists():
        with OUTPUT_REPORT.open("r", encoding="utf-8") as f:
            done_ids = {r["account_id"] for r in csv.DictReader(f) if r.get("account_id")}

    to_process = [r for r in rows if r.get("account_id") not in done_ids]
    total_to_do = len(to_process)
    start_time = time.time()
    
    with OUTPUT_REPORT.open("a", newline="", encoding="utf-8") as out:
        fieldnames = [
            "account_id", "account_name", "website", "headcount", "annual_revenue", 
            "industry", "hq_location", "region_geo", "description", "cloud_stack", 
            "legacy_debt", "strategic_priorities_2026", "business_triggers", 
            "sales_hook_2026", "sources", "confidence"
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        if not OUTPUT_REPORT.exists() or OUTPUT_REPORT.stat().st_size == 0:
            writer.writeheader()

        for i in range(0, total_to_do, MAX_CONCURRENCY):
            batch = to_process[i : i + MAX_CONCURRENCY]
            tasks = []
            for row in batch:
                async def t(r):
                    async with asyncio.Semaphore(MAX_CONCURRENCY):
                        return r, await call_gemini_master(client.aio, r["account_name"], r.get("website", ""))
                tasks.append(t(row))

            results = await asyncio.gather(*tasks)

            for orig, data in results:
                if data == "RATE_LIMIT" or not data: continue
                writer.writerow({
                    "account_id": orig.get("account_id"),
                    "account_name": orig.get("account_name"),
                    "website": data.get("website", orig.get("website")),
                    "headcount": data.get("headcount"),
                    "annual_revenue": data.get("annual_revenue"),
                    "industry": data.get("industry"),
                    "hq_location": data.get("hq_location"),
                    "region_geo": data.get("region_geo"),
                    "description": data.get("description"),
                    "cloud_stack": data.get("cloud_stack"),
                    "legacy_debt": data.get("legacy_debt"),
                    "strategic_priorities_2026": data.get("strategic_priorities_2026"),
                    "business_triggers": data.get("business_triggers"),
                    "sales_hook_2026": data.get("sales_hook_2026"),
                    "sources": data.get("sources"),
                    "confidence": data.get("confidence")
                })
            out.flush()
            
            elapsed = time.time() - start_time
            processed_now = i + len(batch)
            eta = datetime.now() + timedelta(seconds=(total_to_do - processed_now) * (elapsed / processed_now))
            logging.info(f"Progress: {len(done_ids) + processed_now} / {len(rows)} | ETA: {eta.strftime('%H:%M:%S')}")
            await asyncio.sleep(BATCH_SLEEP)

if __name__ == "__main__":
    asyncio.run(async_main())