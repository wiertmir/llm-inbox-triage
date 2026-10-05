"""Offline integrity checks for the labeled samples for issue #8."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage  # noqa: E402

EVALS = ROOT / "evals"
MANIFEST = json.loads((EVALS / "expected.json").read_text(encoding="utf-8"))
SAMPLES = MANIFEST["samples"]


def test_manifest_has_twenty_unique_samples():
    assert set(MANIFEST) == {"schema_version", "samples"}
    assert MANIFEST["schema_version"] == 1
    assert len(SAMPLES) == 20
    assert len({sample["id"] for sample in SAMPLES}) == 20
    assert len({sample["file"] for sample in SAMPLES}) == 20


def test_manifest_covers_every_sample_file():
    labeled = {EVALS / sample["file"] for sample in SAMPLES}
    assert labeled == set((EVALS / "samples").glob("*.txt"))


def test_samples_cover_all_categories():
    assert {sample["expected"]["category"] for sample in SAMPLES} == {
        category.value for category in triage.Category
    }


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample["id"] for sample in SAMPLES])
def test_sample_and_expected_fields_are_valid(sample):
    assert set(sample) == {"id", "file", "expected"}
    path = EVALS / sample["file"]
    assert path.parent == EVALS / "samples"
    assert path.suffix == ".txt"
    assert path.stem == sample["id"]
    text = path.read_text(encoding="utf-8")
    assert text.startswith("Subject: ")
    assert text.strip()
    assert text != (ROOT / "sample.txt").read_text(encoding="utf-8")

    expected = sample["expected"]
    assert set(expected) == {"category", "extracted"}
    triage.Category(expected["category"])
    assert set(expected["extracted"]) == set(triage.Extracted.model_fields)
    extracted = triage.Extracted.model_validate(expected["extracted"])
    assert extracted.model_dump(mode="json") == expected["extracted"]
    assert set(extracted.deadlines) <= set(extracted.dates)
    assert len(extracted.dates) == len(set(extracted.dates))
    assert len(extracted.deadlines) == len(set(extracted.deadlines))
    assert len(extracted.names) == len(set(extracted.names))
    for name in extracted.names:
        assert name in text
    for amount in extracted.amounts:
        assert amount.currency in {"JPY", "USD", "EUR"}
