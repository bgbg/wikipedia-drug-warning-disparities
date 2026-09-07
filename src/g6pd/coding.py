"""
Keyword-based G6PD warning detection and coding.

For each Wikipedia article record, applies multilingual keyword matching to
detect G6PD safety warnings, codes their explicitness (0/1/2), and records
their location in the article.
"""

import json
import re

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

DISEASE_TERMS = {
    "en": ["g6pd", "glucose-6-phosphate dehydrogenase", "g6pd deficiency"],
    "he": ["g6pd", "גלוקוז-6-פוספט דהידרוגנאז", "חסר g6pd", "מחסור ב-g6pd"],
    "ar": ["g6pd", "الجلوكوز-6-فوسفات", "نقص g6pd", "نقص إنزيم",
           "الغلوكوز-6-فوسفات", "عوز نازعة هيدروجين"],
    "ru": ["g6pd", "г6фд", "глюкозо-6-фосфатдегидрогеназа", "дефицит g6pd", "дефицит г6фд"],
}

ACTION_TERMS = {
    "en": ["contraindicated", "avoid", "do not use", "not recommended"],
    "he": ["התוויה נגד", "להימנע", "אסור"],
    "ar": ["موانع الاستخدام", "يُمنع", "تجنب", "محظور"],
    "ru": ["противопоказан", "противопоказания", "не применять", "избегать"],
}

HEMOLYSIS_TERMS = {
    "en": ["hemolysis", "haemolysis", "hemolytic anemia", "haemolytic anaemia"],
    "he": ["המוליזה", "אנמיה המוליטית"],
    "ar": ["انحلال الدم", "فقر الدم الانحلالي"],
    "ru": ["гемолиз", "гемолитическая анемия"],
}

# Heading synonyms for location detection
_CONTRAINDICATION_HEADINGS = {
    "en": ["contraindication", "contraindications"],
    "he": ["התוויות נגד", "התווית נגד"],
    "ar": ["موانع", "موانع الاستخدام", "مضادات الاستطباب", "موانع الاستعمال"],
    "ru": ["противопоказания", "противопоказан"],
}

_ADVERSE_HEADINGS = {
    "en": ["adverse", "side effect", "side effects", "undesirable"],
    "he": ["תופעות לוואי", "תגובות שליליות"],
    "ar": ["آثار جانبية", "تأثيرات جانبية", "أضرار", "الآثار الجانبية",
           "الأعراض الجانبية", "تفاعلات دوائية ضارة"],
    "ru": ["побочные", "нежелательные", "побочное действие"],
}


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def split_sections(wikitext: str) -> dict[str, str]:
    """Split wikitext by '== Heading ==' markers.

    Returns a dict where the key 'lead' holds content before the first level-2
    heading, and subsequent keys are the normalised (lowercase, stripped)
    heading names mapped to their section body text.

    Only level-2 headings (exactly two '=' signs on each side) are used as
    top-level section delimiters, matching the standard Wikipedia structure.
    Deeper headings (`=== Sub ===`, `==== … ====`) are left inside their
    parent level-2 section body, not split out.
    """
    sections: dict[str, str] = {}
    # Split on level-2 headings only. The lookarounds require *exactly* two '='
    # on each side, so a level-3 heading like `=== Sub ===` is not mistaken for
    # a level-2 section named `= sub =` (which truncated the parent body).
    pattern = re.compile(r'^==(?!=) *(.+?) *(?<!=)==(?!=)[ \t]*$', re.MULTILINE)
    parts = pattern.split(wikitext)
    # parts[0] is lead, then alternating heading / body
    sections["lead"] = parts[0]
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip().lower()
        body = parts[i + 1]
        # If the same heading appears twice (rare), append content
        if heading in sections:
            sections[heading] = sections[heading] + "\n" + body
        else:
            sections[heading] = body
    return sections


# ---------------------------------------------------------------------------
# Location detection
# ---------------------------------------------------------------------------

def _contains_any(text: str, terms: list[str]) -> bool:
    """Return True if text contains any of the terms (case-insensitive)."""
    text_lower = text.lower()
    return any(t.lower() in text_lower for t in terms)


def _strip_refs(text: str) -> str:
    """Remove <ref>...</ref> content (including multiline)."""
    return re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)


def _only_in_refs(text: str, terms: list[str]) -> bool:
    """Return True if terms appear in text only inside <ref> tags."""
    if not _contains_any(text, terms):
        return False
    return not _contains_any(_strip_refs(text), terms)


def detect_location(sections: dict, lang: str, matched_terms: list) -> str:
    """Determine where in the article the G6PD-related content was found.

    Parameters
    ----------
    sections : dict
        Output of split_sections().
    lang : str
        Language code ("en", "he", "ar", "ru").
    matched_terms : list
        The disease terms that were found in the article.

    Returns
    -------
    One of: "lead", "contraindications_section", "adverse_effects_section",
    "body", "footnote_only", "absent".
    """
    if not matched_terms:
        return "absent"

    contra_synonyms = _CONTRAINDICATION_HEADINGS.get(lang, [])
    adverse_synonyms = _ADVERSE_HEADINGS.get(lang, [])

    # Check lead section
    lead_text = sections.get("lead", "")
    if _contains_any(lead_text, matched_terms):
        if not _only_in_refs(lead_text, matched_terms):
            return "lead"

    # Check named sections
    found_in_contra = False
    found_in_adverse = False
    found_in_body = False
    found_in_refs_only = False

    for heading, body in sections.items():
        if heading == "lead":
            continue
        if not _contains_any(body, matched_terms):
            continue
        # Determine section type
        heading_lower = heading.lower()
        is_contra = any(syn.lower() in heading_lower for syn in contra_synonyms)
        is_adverse = any(syn.lower() in heading_lower for syn in adverse_synonyms)

        in_refs_only = _only_in_refs(body, matched_terms)
        if in_refs_only:
            found_in_refs_only = True
            continue

        if is_contra:
            found_in_contra = True
        elif is_adverse:
            found_in_adverse = True
        else:
            found_in_body = True

    if found_in_contra:
        return "contraindications_section"
    if found_in_adverse:
        return "adverse_effects_section"
    if found_in_body:
        return "body"
    if found_in_refs_only:
        return "footnote_only"

    # Check lead refs-only as last resort
    if _contains_any(lead_text, matched_terms):
        return "footnote_only"

    return "absent"


# ---------------------------------------------------------------------------
# Article coding
# ---------------------------------------------------------------------------

def code_article(record: dict) -> dict:
    """Apply keyword detection to a single article record.

    Parameters
    ----------
    record : dict
        A record from articles.jsonl with keys: drug, lang, cpic_risk,
        category, exists, wikitext (optional).

    Returns
    -------
    dict with keys:
        warning_code      : int  (0, 1, or 2)
        warning_location  : str
        matched_terms     : str  (JSON-encoded list)
    """
    lang = record.get("lang", "en")

    # Non-existent articles
    if not record.get("exists", False):
        return {
            "warning_code": 0,
            "warning_location": "absent",
            "matched_terms": json.dumps([]),
        }

    wikitext = record.get("wikitext", "")
    sections = split_sections(wikitext)

    disease_terms = DISEASE_TERMS.get(lang, DISEASE_TERMS["en"])
    action_terms = ACTION_TERMS.get(lang, ACTION_TERMS["en"])

    # ---- Step 1: find disease terms in the full text ----
    found_disease: list[str] = []
    full_text_lower = wikitext.lower()
    for term in disease_terms:
        if term.lower() in full_text_lower:
            found_disease.append(term)

    if not found_disease:
        return {
            "warning_code": 0,
            "warning_location": "absent",
            "matched_terms": json.dumps([]),
        }

    # ---- Step 2: check for action terms in same section as disease term ----
    found_action: list[str] = []
    action_in_same_section = False

    for section_text in sections.values():
        section_lower = section_text.lower()
        disease_in_section = any(t.lower() in section_lower for t in disease_terms)
        if not disease_in_section:
            continue
        for term in action_terms:
            if term.lower() in section_lower:
                if term not in found_action:
                    found_action.append(term)
                action_in_same_section = True

    # ---- Step 3: collect all matched terms ----
    matched: list[str] = list(found_disease)
    for term in found_action:
        if term not in matched:
            matched.append(term)

    # ---- Step 4: assign warning code ----
    if action_in_same_section:
        warning_code = 2
    else:
        warning_code = 1

    # ---- Step 5: detect location ----
    location = detect_location(sections, lang, found_disease)

    return {
        "warning_code": warning_code,
        "warning_location": location,
        "matched_terms": json.dumps(matched, ensure_ascii=False),
    }
