"""One-off fix: replace metformin with omeprazole in all data files."""
import sys, json, time
sys.path.insert(0, '/Users/boris/devel/jce/pharmaco/g6pd/src')
from g6pd.fetch import fetch_article

ARTICLES = "/Users/boris/devel/jce/pharmaco/g6pd/data/raw/articles.jsonl"
LANGS = {"en": "Omeprazole", "he": "אומפרזול", "ar": "أوميبرازول", "ru": "Омепразол"}

records = []
with open(ARTICLES) as f:
    for line in f:
        rec = json.loads(line)
        if rec["drug"] != "metformin":
            records.append(rec)

print(f"After removing metformin: {len(records)} records")

for lang, title in LANGS.items():
    print(f"Fetching omeprazole/{lang}...")
    rec = fetch_article(lang, title, "omeprazole", "", "", "control")
    records.append(rec)
    print(f"  exists={rec['exists']}, words={rec.get('word_count')}, refs={rec.get('n_references')}")
    time.sleep(1)

with open(ARTICLES, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Done. Total records: {len(records)}")
