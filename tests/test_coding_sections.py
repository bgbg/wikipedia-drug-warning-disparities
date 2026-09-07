"""Regression test for `coding.split_sections` heading parsing (audit flag F-004).

No network. Locks the invariant that only *level-2* headings (`== H ==`) split a
section: a deeper heading like `=== Sub ===` must stay inside its parent level-2
body and must not be mis-parsed as a level-2 section named `= sub =` (the bug the
old `^== *(.+?) *==$` regex produced, which truncated the parent body at the
first subsection).

`split_sections`/`detect_location` are used only by the deprecated Milestone-1
keyword pipeline; this test guards the fix if that path is ever revived.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from g6pd.coding import split_sections  # noqa: E402

WIKITEXT = """lead paragraph before any heading
== Pharmacology ==
parent body before the subsection
=== Mechanism of action ===
nested level-3 body
== Adverse effects ==
G6PD deficiency contraindication text
"""


def test_level3_heading_stays_in_parent_body():
    sections = split_sections(WIKITEXT)

    # exactly the two level-2 headings (plus the lead) become keys
    assert set(sections) == {"lead", "pharmacology", "adverse effects"}

    # the old regex produced a spurious "= sub =" / "mechanism..." key
    assert "= mechanism of action =" not in sections
    assert "mechanism of action" not in sections

    # the level-3 heading and its body remain inside the parent level-2 section
    assert "=== Mechanism of action ===" in sections["pharmacology"]
    assert "nested level-3 body" in sections["pharmacology"]


def test_level2_headings_still_split_with_surrounding_whitespace():
    wikitext = "lead\n==  Spaced Heading  ==  \nbody text\n"
    sections = split_sections(wikitext)
    assert "spaced heading" in sections
    assert "body text" in sections["spaced heading"]


def test_level4_heading_not_split():
    wikitext = "lead\n== Top ==\nbody\n==== Deep ====\ndeep body\n"
    sections = split_sections(wikitext)
    assert set(sections) == {"lead", "top"}
    assert "==== Deep ====" in sections["top"]
