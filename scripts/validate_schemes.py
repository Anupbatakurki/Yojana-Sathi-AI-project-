"""Validate data/schemes.json before every commit / deploy (also runs in CI)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saathi.eligibility import DERIVED_LABELS, FLAG_LABELS
from saathi.schemes import load_schemes

KNOWN = set(FLAG_LABELS) | set(DERIVED_LABELS)
ALLOWED_RULES = {
    "states",
    "min_age",
    "max_age",
    "genders",
    "residence",
    "max_annual_income",
    "requires",
    "any_of",
    "excludes",
}

schemes = load_schemes(Path(__file__).resolve().parent.parent / "data" / "schemes.json")
errors = []
for s in schemes:
    unknown_keys = set(s.rules) - ALLOWED_RULES
    if unknown_keys:
        errors.append(f"{s.id}: unknown rule keys {unknown_keys}")
    for key in ("requires", "any_of", "excludes"):
        for flag in s.rules.get(key, []):
            if flag not in KNOWN:
                errors.append(f"{s.id}: unknown flag '{flag}' in {key}")
    if not s.official_url.startswith("https://"):
        errors.append(f"{s.id}: official_url must be https")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print(f"OK: {len(schemes)} schemes validated")
