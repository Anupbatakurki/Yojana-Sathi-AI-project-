"""Deterministic, explainable eligibility engine.

Design choice: the LLM never decides who is eligible. Rules in schemes.json do,
so the result is auditable and cannot be hallucinated. The LLM only explains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .schemes import Scheme

# Human-readable labels for the yes/no flags a citizen can tick in the UI.
FLAG_LABELS = {
    "farmer": "Owns / cultivates agricultural land",
    "bpl": "BPL / Antyodaya ration card household",
    "income_tax_payer": "Someone in the family pays income tax",
    "govt_employee": "Government employee (self or spouse)",
    "student": "Currently a student",
    "street_vendor": "Street vendor",
    "artisan": "Traditional artisan / craftsperson",
    "pregnant_or_lactating": "Pregnant or lactating mother",
    "has_girl_child_under_10": "Has a daughter below 10 years",
    "unemployed_graduate": "Unemployed graduate / diploma holder",
    "houseless": "No pucca (permanent) house",
    "has_lpg_connection": "Household already has an LPG connection",
    "domestic_electricity_consumer": "Has a household electricity connection",
    "entrepreneur": "Wants to start a new business",
}

DERIVED_LABELS = {
    "female": "woman",
    "male": "man",
    "transgender": "transgender person",
    "sc": "Scheduled Caste",
    "st": "Scheduled Tribe",
    "obc": "OBC",
    "senior_70_plus": "aged 70 or above",
}


@dataclass
class CitizenProfile:
    age: int | None = None
    gender: str | None = None  # male / female / transgender
    state: str | None = None  # e.g. "Karnataka"
    residence: str | None = None  # rural / urban
    social_category: str | None = None  # general / obc / sc / st
    annual_income: int | None = None  # household, in rupees
    flags: set[str] = field(default_factory=set)

    def all_flags(self) -> set[str]:
        f = set(self.flags)
        if self.gender:
            f.add(self.gender.lower())
        if self.social_category:
            f.add(self.social_category.lower())
        if self.age is not None and self.age >= 70:
            f.add("senior_70_plus")
        return f


class Verdict(str, Enum):
    ELIGIBLE = "eligible"
    MAYBE = "maybe"  # nothing failed, but some info is missing
    NOT_ELIGIBLE = "not_eligible"


@dataclass
class EligibilityResult:
    scheme: Scheme
    verdict: Verdict
    reasons: list[str]  # why it passed / failed
    missing: list[str]  # what we'd need to know to be sure


def _label(flag: str) -> str:
    return FLAG_LABELS.get(flag, DERIVED_LABELS.get(flag, flag))


def check(scheme: Scheme, p: CitizenProfile) -> EligibilityResult:
    r = scheme.rules
    flags = p.all_flags()
    fails, oks, missing = [], [], []

    states = r.get("states", ["ALL"])
    if "ALL" not in states:
        if p.state is None:
            missing.append("state of residence")
        elif p.state.lower() not in [s.lower() for s in states]:
            fails.append(f"only for residents of {', '.join(states)}")
        else:
            oks.append(f"resident of {p.state}")

    if "min_age" in r or "max_age" in r:
        lo, hi = r.get("min_age", 0), r.get("max_age", 200)
        if p.age is None:
            missing.append("age")
        elif not lo <= p.age <= hi:
            fails.append(f"age must be between {lo} and {hi}" if "max_age" in r else f"minimum age is {lo}")
        else:
            oks.append(f"age {p.age} is within limits")

    if "genders" in r:
        if p.gender is None:
            missing.append("gender")
        elif p.gender.lower() not in r["genders"]:
            fails.append(f"only for {' / '.join(r['genders'])} applicants")
        else:
            oks.append(f"applicant is {_label(p.gender.lower())}")

    if "residence" in r:
        if p.residence is None:
            missing.append("rural or urban residence")
        elif p.residence.lower() not in r["residence"]:
            fails.append(f"only for {' / '.join(r['residence'])} areas")
        else:
            oks.append(f"lives in a {p.residence} area")

    if "max_annual_income" in r:
        if p.annual_income is None:
            missing.append("annual family income")
        elif p.annual_income > r["max_annual_income"]:
            fails.append(f"family income must be at most Rs {r['max_annual_income']:,}")
        else:
            oks.append("income is within the limit")

    for req in r.get("requires", []):
        if req in flags:
            oks.append(_label(req))
        else:
            fails.append(f"requires: {_label(req)}")

    any_of = r.get("any_of", [])
    if any_of:
        hit = [a for a in any_of if a in flags]
        if hit:
            oks.append(_label(hit[0]))
        else:
            fails.append("requires one of: " + " / ".join(_label(a) for a in any_of))

    for exc in r.get("excludes", []):
        if exc in flags:
            fails.append(f"not available if: {_label(exc)}")

    if fails:
        return EligibilityResult(scheme, Verdict.NOT_ELIGIBLE, fails, missing)
    if missing:
        return EligibilityResult(scheme, Verdict.MAYBE, oks, missing)
    return EligibilityResult(scheme, Verdict.ELIGIBLE, oks, [])


def screen(schemes: list[Scheme], profile: CitizenProfile) -> list[EligibilityResult]:
    order = {Verdict.ELIGIBLE: 0, Verdict.MAYBE: 1, Verdict.NOT_ELIGIBLE: 2}
    results = [check(s, profile) for s in schemes]
    return sorted(results, key=lambda x: (order[x.verdict], x.scheme.name))
