"""
build_new_topic_prompts.py — add QT and PGx-SCAR few-shot prompts (M3, #28).

Idempotently injects two topic entries into
data/prompts/classification_prompts.json, mirroring the structure of the
existing g6pd/teratogenicity prompts (system_parts + few-shot examples). The
0/1/2 scale matches the project coding scheme: 0 = no mention, 1 = mentioned
without a clinical-action directive, 2 = explicit contraindication/avoidance.

Usage:
    python scripts/build_new_topic_prompts.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import CLASSIFICATION_PROMPTS  # noqa: E402

COMMON_RULES = [
    "",
    "IMPORTANT RULES:",
    "- The article may be written in ANY language, including English, Hebrew, Arabic, Russian, or others. You must classify the content accurately regardless of the language.",
    "- Base your classification ONLY on the text provided. Do NOT use your own medical knowledge about the drug. If the article does not mention the risk, classify as 0 even if you know the drug carries it.",
    "- Only classify what is explicitly written in the article text.",
]

LOCATION_BLOCK = [
    "",
    "Also identify WHERE in the article the warning appears:",
    "- 'lead': in the introductory section before any headings",
    "- 'contraindications_section': in a section about contraindications",
    "- 'adverse_effects_section': in a section about adverse effects or side effects",
    "- 'body': in another named section",
    "- 'footnote_only': only in references/footnotes",
    "- 'absent': no mention found",
    "",
    'Respond with ONLY a JSON object in this exact format:',
    '{"code": <0|1|2>, "location": "<location>", "justification": "<one sentence explaining your classification>"}',
]

QT = {
    "system_parts": [
        "You are a medical information classifier.",
        "You will be given the full text of a Wikipedia drug article.",
        "Your task is to determine whether the article contains a warning that the drug can prolong the QT interval or cause Torsades de Pointes (a dangerous ventricular heart-rhythm disturbance).",
        *COMMON_RULES,
        "",
        "Classify using this scale:",
        "- 0: No mention of QT prolongation, the QT interval, Torsades de Pointes, or a drug-induced dangerous ventricular arrhythmia / heart-rhythm disturbance anywhere in the article.",
        "- 1: QT prolongation, Torsades, or arrhythmia risk is mentioned, but there is no explicit clinical action instruction (e.g., it appears in a pharmacology/mechanism section, or in a side-effects list without any directive).",
        "- 2: There is an explicit directive linked to QT/Torsades risk. This includes ANY of the following:",
        "    - The word 'contraindicated' used in connection with QT prolongation, long-QT syndrome, or co-administration with other QT-prolonging drugs.",
        "    - QT prolongation or 'risk of Torsades' listed under a 'Contraindications' or 'Warnings' heading with an avoidance directive (in any language).",
        "    - Instructions such as 'avoid', 'do not use', 'should not be used', 'use only with ECG monitoring', or 'avoid in congenital long-QT syndrome / with low potassium or magnesium' in connection with the cardiac risk.",
        "  The term 'contraindicated' is itself a clinical action instruction.",
        *LOCATION_BLOCK,
    ],
    "examples": [
        {"role": "user", "content": "Classify this article for QT-prolongation / Torsades de Pointes warnings:\n\nAmoxicillin is a penicillin-type antibiotic used to treat bacterial infections. Common side effects include nausea, diarrhea, and rash."},
        {"role": "assistant", "content": '{"code": 0, "location": "absent", "justification": "The article does not mention QT prolongation, Torsades, or any cardiac rhythm risk."}'},
        {"role": "user", "content": "Classify this article for QT-prolongation / Torsades de Pointes warnings:\n\nAzithromycin is a macrolide antibiotic. == Side effects == Reported adverse effects include diarrhea, nausea, and prolongation of the QT interval."},
        {"role": "assistant", "content": '{"code": 1, "location": "adverse_effects_section", "justification": "QT prolongation is listed as a side effect but the article gives no directive to avoid or monitor."}'},
        {"role": "user", "content": "Classify this article for QT-prolongation / Torsades de Pointes warnings:\n\nCitalopram is an antidepressant. == Contraindications == Citalopram is contraindicated in patients with congenital long QT syndrome and should not be combined with other QT-prolonging medications because of the risk of Torsades de Pointes."},
        {"role": "assistant", "content": '{"code": 2, "location": "contraindications_section", "justification": "The contraindications section explicitly bars use in long-QT patients and with other QT-prolonging drugs due to Torsades risk."}'},
    ],
}

PGX = {
    "system_parts": [
        "You are a medical information classifier.",
        "You will be given the full text of a Wikipedia drug article.",
        "Your task is to determine whether the article contains a warning about a SEVERE adverse drug reaction tied to a pharmacogenetic (inherited) factor. Two kinds count:",
        "  (a) a severe cutaneous/hypersensitivity reaction — Stevens-Johnson syndrome (SJS), toxic epidermal necrolysis (TEN), DRESS, or HLA-linked hypersensitivity — associated with a specific HLA allele (e.g., HLA-B*15:02, HLA-B*58:01, HLA-B*57:01); or",
        "  (b) life-threatening respiratory depression / opioid toxicity in CYP2D6 ultrarapid metabolizers (e.g., codeine or tramadol), including the related contraindication in children or breastfeeding mothers.",
        *COMMON_RULES,
        "- A generic mention of 'rash' or mild skin reaction with NO link to a severe reaction (SJS/TEN/DRESS) or a genetic factor does NOT count for this topic (code 0). A generic mention of metabolism with no ultrarapid-metabolizer toxicity does NOT count (code 0).",
        "",
        "Classify using this scale:",
        "- 0: No mention of a pharmacogenetic severe-ADR risk — neither an HLA-associated severe skin/hypersensitivity reaction nor CYP2D6 ultrarapid-metabolizer opioid toxicity.",
        "- 1: Such a risk is mentioned (e.g., the article notes an HLA-B allele association with SJS/TEN, or that ultrarapid CYP2D6 metabolizers reach higher morphine levels), but there is no explicit clinical action instruction.",
        "- 2: There is an explicit directive tied to the pharmacogenetic risk. This includes ANY of the following:",
        "    - The word 'contraindicated' in connection with the genetic factor (HLA-allele carriers; CYP2D6 ultrarapid metabolizers; codeine in children or breastfeeding).",
        "    - A recommendation or requirement to perform genetic testing / genotyping (e.g., for HLA-B*15:02 or HLA-B*57:01) BEFORE prescribing.",
        "    - Instructions such as 'avoid', 'do not use', or 'should not be used' in carriers of the allele or in ultrarapid metabolizers, or such a risk listed under a 'Contraindications' / boxed-warning heading with an avoidance directive (in any language).",
        "  The term 'contraindicated' is itself a clinical action instruction.",
        *LOCATION_BLOCK,
    ],
    "examples": [
        {"role": "user", "content": "Classify this article for pharmacogenetic severe-adverse-reaction warnings:\n\nOmeprazole is a proton-pump inhibitor used to reduce stomach acid. Common side effects include headache and abdominal pain."},
        {"role": "assistant", "content": '{"code": 0, "location": "absent", "justification": "The article mentions no HLA-linked severe skin reaction and no CYP2D6 ultrarapid-metabolizer toxicity."}'},
        {"role": "user", "content": "Classify this article for pharmacogenetic severe-adverse-reaction warnings:\n\nCarbamazepine is an anticonvulsant. == Mechanism and pharmacogenetics == The HLA-B*15:02 allele, common in some Asian populations, is associated with an increased risk of Stevens-Johnson syndrome."},
        {"role": "assistant", "content": '{"code": 1, "location": "body", "justification": "The HLA-B*15:02 association with Stevens-Johnson syndrome is described but no testing or avoidance directive is given."}'},
        {"role": "user", "content": "Classify this article for pharmacogenetic severe-adverse-reaction warnings:\n\nCodeine is an opioid. == Contraindications == Codeine is contraindicated in children under 12 and in CYP2D6 ultrarapid metabolizers, who are at risk of fatal respiratory depression; it should not be used by breastfeeding mothers."},
        {"role": "assistant", "content": '{"code": 2, "location": "contraindications_section", "justification": "Use is explicitly contraindicated in ultrarapid metabolizers, children, and breastfeeding mothers due to fatal respiratory depression."}'},
    ],
}


def main() -> None:
    with open(CLASSIFICATION_PROMPTS, encoding="utf-8") as fh:
        prompts = json.load(fh)
    prompts["qt"] = QT
    prompts["pgx_scar"] = PGX
    with open(CLASSIFICATION_PROMPTS, "w", encoding="utf-8") as fh:
        json.dump(prompts, fh, ensure_ascii=False, indent=2)
    print(f"Wrote topics {list(prompts)} to {CLASSIFICATION_PROMPTS}")


if __name__ == "__main__":
    main()
