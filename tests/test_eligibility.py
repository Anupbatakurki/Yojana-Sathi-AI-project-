import pytest

from saathi.eligibility import CitizenProfile, Verdict, check
from saathi.schemes import chunk_schemes


def test_knowledge_base_is_valid(schemes):
    assert len(schemes) >= 15
    assert all(s.official_url.startswith("https://") for s in schemes.values())


def test_chunks_are_self_contained(schemes):
    chunks = chunk_schemes(list(schemes.values()))
    assert len(chunks) == 5 * len(schemes)
    assert all(c.scheme_name in c.text for c in chunks)


def test_farmer_gets_pm_kisan(schemes):
    assert check(schemes["pm-kisan"], CitizenProfile(age=45, flags={"farmer"})).verdict == Verdict.ELIGIBLE


def test_tax_payer_excluded_from_pm_kisan(schemes):
    r = check(schemes["pm-kisan"], CitizenProfile(age=45, flags={"farmer", "income_tax_payer"}))
    assert r.verdict == Verdict.NOT_ELIGIBLE
    assert any("income tax" in x for x in r.reasons)


@pytest.mark.parametrize(
    "state,expected",
    [
        (None, Verdict.MAYBE),
        ("Karnataka", Verdict.ELIGIBLE),
        ("karnataka", Verdict.ELIGIBLE),
        ("Kerala", Verdict.NOT_ELIGIBLE),
    ],
)
def test_state_specific_scheme(schemes, state, expected):
    assert check(schemes["ka-shakti"], CitizenProfile(gender="female", state=state)).verdict == expected


@pytest.mark.parametrize(
    "age,expected",
    [(17, Verdict.NOT_ELIGIBLE), (18, Verdict.ELIGIBLE), (40, Verdict.ELIGIBLE), (41, Verdict.NOT_ELIGIBLE)],
)
def test_age_band_boundaries(schemes, age, expected):
    assert check(schemes["apy"], CitizenProfile(age=age)).verdict == expected


def test_missing_age_is_maybe(schemes):
    r = check(schemes["apy"], CitizenProfile())
    assert r.verdict == Verdict.MAYBE and "age" in r.missing


def test_senior_gets_pmjay_without_bpl(schemes):
    assert check(schemes["pmjay"], CitizenProfile(age=72)).verdict == Verdict.ELIGIBLE
    assert check(schemes["pmjay"], CitizenProfile(age=30)).verdict == Verdict.NOT_ELIGIBLE


def test_income_ceiling(schemes):
    p = CitizenProfile(social_category="sc", annual_income=400000, flags={"student"})
    assert check(schemes["nsp-postmatric-sc"], p).verdict == Verdict.NOT_ELIGIBLE
    p.annual_income = 200000
    assert check(schemes["nsp-postmatric-sc"], p).verdict == Verdict.ELIGIBLE


def test_screen_orders_eligible_first(offline_pipe):
    results = offline_pipe.screen(CitizenProfile(age=30, flags={"farmer"}))
    order = [r.verdict for r in results]
    assert order == sorted(order, key=[Verdict.ELIGIBLE, Verdict.MAYBE, Verdict.NOT_ELIGIBLE].index)
