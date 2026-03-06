"""
Offline unit tests for verify_sample_with_gemini.py.

These tests validate the non-API logic: sampling, prompt building, CSV handling,
column validation, result construction, and the _extract_evidence_text helper.
No Gemini API calls are made.
"""

from __future__ import annotations

import csv
import io
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Import the modules under test
from account_enrichment_common import UnknownFixTargets, normalize_whitespace
from verify_sample_with_gemini import (
    VerificationResponse,
    VerificationRowResult,
    _extract_evidence_text,
    build_verification_prompt,
    sample_size_for_proportion,
    z_value_for_confidence,
)


# ---------------------------------------------------------------------------
# Fixtures: dummy CSV rows
# ---------------------------------------------------------------------------

def make_row(
    account_name="Acme Corp",
    website="https://acme.com",
    hq_state="California",
    region="US west",
    industry="Technology",
    annual_revenue="50000000",
    description="Acme Corp builds widgets.",
    segment="SME",
    account_id="001TESTACCOUNT00001",
) -> dict[str, str]:
    """Create a realistic CSV row dict matching the expected column schema."""
    return {
        "Account ID 18 Digit": account_id,
        "Account Name": account_name,
        "Website": website,
        "Description": description,
        "HQ State": hq_state,
        "Region": region,
        "Industry": industry,
        "Annual Revenue": annual_revenue,
        "Segment": segment,
        "Employees": "<500",
    }


# ---------------------------------------------------------------------------
# 1. Statistical sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_z_value_known(self):
        assert z_value_for_confidence(0.95) == 1.96
        assert z_value_for_confidence(0.90) == 1.645
        assert z_value_for_confidence(0.99) == 2.576

    def test_z_value_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            z_value_for_confidence(0.85)

    def test_sample_size_small_population(self):
        """For a tiny population, sample size == population."""
        n = sample_size_for_proportion(
            population_size=10,
            confidence=0.95,
            margin_of_error=0.05,
        )
        assert n == 10  # can't sample more than pop

    def test_sample_size_large_population(self):
        n = sample_size_for_proportion(
            population_size=10000,
            confidence=0.95,
            margin_of_error=0.05,
        )
        # Cochran formula: n0 = 384.16, FPC → ~370
        assert 360 <= n <= 400

    def test_sample_size_59_rows(self):
        """Match our actual CSV: 59 rows → should sample ~52."""
        n = sample_size_for_proportion(
            population_size=59,
            confidence=0.95,
            margin_of_error=0.05,
        )
        assert n == 52

    def test_sample_size_invalid_inputs(self):
        with pytest.raises(ValueError):
            sample_size_for_proportion(population_size=0, confidence=0.95, margin_of_error=0.05)
        with pytest.raises(ValueError):
            sample_size_for_proportion(population_size=100, confidence=0.95, margin_of_error=0)
        with pytest.raises(ValueError):
            sample_size_for_proportion(population_size=100, confidence=1.5, margin_of_error=0.05)


# ---------------------------------------------------------------------------
# 2. Prompt building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_prompt_contains_all_fields(self):
        row = make_row()
        prompt = build_verification_prompt(row)
        assert "Acme Corp" in prompt
        assert "https://acme.com" in prompt
        assert "California" in prompt
        assert "US west" in prompt
        assert "Technology" in prompt
        assert "50000000" in prompt
        assert "SME" in prompt
        assert "Acme Corp builds widgets." in prompt

    def test_prompt_handles_empty_fields(self):
        row = make_row(website="", hq_state="", description="")
        prompt = build_verification_prompt(row)
        # Should not crash, just contain empty values
        assert "Account Name: Acme Corp" in prompt

    def test_prompt_normalizes_whitespace(self):
        row = make_row(account_name="  Acme   Corp  ")
        prompt = build_verification_prompt(row)
        assert "Account Name: Acme Corp" in prompt


# ---------------------------------------------------------------------------
# 3. _extract_evidence_text
# ---------------------------------------------------------------------------

class TestExtractEvidenceText:
    def _make_response(self, text="Some evidence", sources=None, blocked=False):
        """Build a mock GenerateContentResponse."""
        response = MagicMock()
        if blocked:
            response.text = property(lambda self: (_ for _ in ()).throw(ValueError("blocked")))
            # Make .text raise ValueError
            type(response).text = property(lambda self: (_ for _ in ()).throw(ValueError("blocked")))
        else:
            response.text = text

        if sources:
            chunks = []
            for uri in sources:
                chunk = MagicMock()
                chunk.web = MagicMock()
                chunk.web.uri = uri
                chunks.append(chunk)
            gm = MagicMock()
            gm.grounding_chunks = chunks
            candidate = MagicMock()
            candidate.grounding_metadata = gm
            response.candidates = [candidate]
        else:
            response.candidates = None

        return response

    def test_plain_text_no_grounding(self):
        resp = self._make_response(text="Hello world", sources=None)
        result = _extract_evidence_text(resp)
        assert result == "Hello world"
        assert "GROUNDING SOURCES" not in result

    def test_text_with_grounding_sources(self):
        resp = self._make_response(
            text="Evidence text",
            sources=["https://example.com", "https://linkedin.com/company/test"],
        )
        result = _extract_evidence_text(resp)
        assert "Evidence text" in result
        assert "=== GROUNDING SOURCES ===" in result
        assert "[1] https://example.com" in result
        assert "[2] https://linkedin.com/company/test" in result

    def test_empty_text_no_sources(self):
        resp = self._make_response(text="", sources=None)
        result = _extract_evidence_text(resp)
        assert result == ""

    def test_blocked_response_returns_empty(self):
        resp = self._make_response(blocked=True, sources=None)
        result = _extract_evidence_text(resp)
        assert result == ""

    def test_grounding_with_no_web_uri(self):
        """If chunks exist but web.uri is None, no sources appended."""
        response = MagicMock()
        response.text = "Some text"
        chunk = MagicMock()
        chunk.web = MagicMock()
        chunk.web.uri = None
        gm = MagicMock()
        gm.grounding_chunks = [chunk]
        candidate = MagicMock()
        candidate.grounding_metadata = gm
        response.candidates = [candidate]

        result = _extract_evidence_text(response)
        assert result == "Some text"
        assert "GROUNDING SOURCES" not in result


# ---------------------------------------------------------------------------
# 4. VerificationResponse Pydantic schema
# ---------------------------------------------------------------------------

class TestVerificationSchema:
    def test_schema_accepts_all_true(self):
        data = {
            "website_correct": True,
            "hq_state_correct": True,
            "description_correct": True,
            "region_correct": True,
            "industry_reasonable": True,
            "annual_revenue_reasonable": True,
            "notes": "All good",
        }
        v = VerificationResponse(**data)
        assert v.website_correct is True
        assert v.corrected_website is None

    def test_schema_with_corrections(self):
        data = {
            "website_correct": False,
            "hq_state_correct": False,
            "description_correct": True,
            "region_correct": False,
            "industry_reasonable": True,
            "annual_revenue_reasonable": False,
            "corrected_website": "https://correct.com",
            "corrected_hq_state": "Texas",
            "corrected_region": "US south",
            "notes": "Fixed fields",
        }
        v = VerificationResponse(**data)
        assert v.corrected_website == "https://correct.com"
        assert v.corrected_hq_state == "Texas"

    def test_schema_json_round_trip(self):
        """Ensure the schema can round-trip through JSON (as the API returns)."""
        data = {
            "website_correct": True,
            "hq_state_correct": False,
            "description_correct": True,
            "region_correct": True,
            "industry_reasonable": True,
            "annual_revenue_reasonable": False,
            "corrected_hq_state": "Ohio",
            "notes": "Check hq",
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        v = VerificationResponse(**parsed)
        assert v.hq_state_correct is False
        assert v.corrected_hq_state == "Ohio"


# ---------------------------------------------------------------------------
# 5. CSV column validation (offline)
# ---------------------------------------------------------------------------

class TestCSVValidation:
    def test_all_expected_columns_present(self):
        row = make_row()
        targets = UnknownFixTargets()
        expected = {
            "Account ID 18 Digit",
            "Account Name",
            "Industry",
            "Segment",
            targets.website_col,
            targets.description_col,
            targets.hq_state_col,
            targets.region_col,
            targets.annual_revenue_col,
        }
        actual = set(row.keys())
        missing = expected - actual
        assert missing == set(), f"Missing columns: {missing}"

    def test_missing_column_detected(self):
        row = make_row()
        del row["Industry"]
        targets = UnknownFixTargets()
        expected = {
            "Account ID 18 Digit",
            "Account Name",
            "Industry",
            "Segment",
            targets.website_col,
            targets.description_col,
            targets.hq_state_col,
            targets.region_col,
            targets.annual_revenue_col,
        }
        actual = set(row.keys())
        missing = expected - actual
        assert "Industry" in missing

    def test_duplicate_columns_in_csv(self):
        """DictReader with duplicate headers keeps the last value."""
        csv_text = "Name,Website,Value,Website\nAcme,old.com,100,new.com\n"
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        # Python's csv.DictReader uses the last value for duplicates
        assert rows[0]["Website"] == "new.com"


# ---------------------------------------------------------------------------
# 6. VerificationRowResult construction
# ---------------------------------------------------------------------------

class TestRowResult:
    def test_result_from_dict(self):
        verdict = {
            "website_correct": True,
            "hq_state_correct": False,
            "description_correct": True,
            "region_correct": True,
            "industry_reasonable": True,
            "annual_revenue_reasonable": False,
            "corrected_hq_state": "Texas",
            "notes": "HQ is Texas not California",
        }
        row = make_row()
        result = VerificationRowResult(
            account_id=row["Account ID 18 Digit"],
            account_name=row["Account Name"],
            website=row["Website"],
            hq_state=row["HQ State"],
            region=row["Region"],
            description=row["Description"],
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
        assert result.website_correct is True
        assert result.hq_state_correct is False
        assert result.corrected_hq_state == "Texas"

    def test_result_from_empty_verdict(self):
        """When Gemini returns {}, all booleans should be False."""
        verdict = {}
        row = make_row()
        result = VerificationRowResult(
            account_id=row["Account ID 18 Digit"],
            account_name=row["Account Name"],
            website=row["Website"],
            hq_state=row["HQ State"],
            region=row["Region"],
            description=row["Description"],
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
        assert result.website_correct is False
        assert result.hq_state_correct is False
        assert result.description_correct is False
        assert result.notes == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
