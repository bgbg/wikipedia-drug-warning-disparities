"""Harness-liveness smoke test for the Data Accuracy Audit.

This is NOT a regression test — it only proves that pytest collects and runs
under the `pharmaco` env. Real regression tests are added by their own audit
issues (e.g. the Hebrew reference-count guard in #36).
"""

from pathlib import Path


def test_pytest_runs():
    assert True


def test_audit_scaffolding_present():
    audit = Path(__file__).resolve().parent.parent / "audit"
    for name in ("README.md", "report.md", "flags.md"):
        assert (audit / name).is_file(), f"missing audit/{name}"
