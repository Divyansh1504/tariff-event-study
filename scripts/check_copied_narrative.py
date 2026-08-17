"""One-off check: does any prose in README.md, CORRECTIONS.md, or the notebook markdown/text
cells share an exact N-word phrase with the excluded Word docs (Capstone Report 1, Capstone
Report 2 Final)? Flags verbatim or near-verbatim carryover, not just file-level copying."""

import json
import re
import sys


def normalize_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def shingles(words, n):
    return set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def load_source_shingles(path, n):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return shingles(normalize_words(text), n)


def notebook_prose(path):
    nb = json.load(open(path, encoding="utf-8"))
    parts = []
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            parts.append("".join(c["source"]))
        elif c["cell_type"] == "code":
            # include comments and string literals used as print()/narration, not full code
            for line in "".join(c["source"]).splitlines():
                s = line.strip()
                if s.startswith("#"):
                    parts.append(s.lstrip("#").strip())
    return "\n".join(parts)


N = 7  # 7-word exact-phrase match threshold
sources = {
    "Capstone Report 1": load_source_shingles("D:/tmp_Capstone_Report_1.txt", N),
    "Capstone Report 2 Final": load_source_shingles("D:/tmp_Capstone_Report_2_Final.txt", N),
}

targets = {
    "README.md": open("README.md", encoding="utf-8").read(),
    "CORRECTIONS.md": open("CORRECTIONS.md", encoding="utf-8").read(),
    "notebooks/report1_industry_tariff_impact.ipynb": notebook_prose("notebooks/report1_industry_tariff_impact.ipynb"),
    "notebooks/report2_portfolio_event_study.ipynb": notebook_prose("notebooks/report2_portfolio_event_study.ipynb"),
}

any_hit = False
for tname, ttext in targets.items():
    twords = normalize_words(ttext)
    tshingles = shingles(twords, N)
    for sname, sset in sources.items():
        overlap = tshingles & sset
        if overlap:
            any_hit = True
            print(f"OVERLAP: {tname} <-> {sname}")
            for phrase in sorted(overlap):
                print("   ", " ".join(phrase))

if not any_hit:
    print(f"No {N}-word exact-phrase overlaps found between repo prose and the excluded Word docs.")
