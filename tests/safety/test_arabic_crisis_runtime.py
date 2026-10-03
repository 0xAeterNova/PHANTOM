from phantom.config import SafetyConfig
from phantom.safety.crisis_router import CrisisRouter
from phantom.schemas import SafetyRoute


def test_arabic_first_person_self_harm_language_routes_to_crisis() -> None:
    result = CrisisRouter(SafetyConfig()).assess("أنا أريد أن أقتل نفسي")
    assert result.route is SafetyRoute.CRISIS


def test_arabic_research_context_is_not_treated_as_first_person_crisis() -> None:
    result = CrisisRouter(SafetyConfig()).assess("هذا بحث عن الانتحار عند شخص آخر")
    assert result.route is SafetyRoute.SUPPORT


def test_arabic_ordinary_distress_after_accident_does_not_route_to_emergency() -> None:
    result = CrisisRouter(SafetyConfig()).assess(
        "لقد تعرضت لحادث اليوم ونفسيتي تعبانة جدًا وحزين ولا أعلم ما العمل"
    )
    assert result.route is SafetyRoute.NORMAL


def test_arabic_sadness_without_immediate_danger_is_not_a_crisis() -> None:
    result = CrisisRouter(SafetyConfig()).assess("أنا حزين ومتعب نفسيًا وأحتاج إلى من يسمعني")
    assert result.route is SafetyRoute.NORMAL


def test_arabic_passive_death_wish_routes_to_support_not_emergency() -> None:
    result = CrisisRouter(SafetyConfig()).assess("أنا لا أريد أن أعيش")
    assert result.route is SafetyRoute.SUPPORT


def test_arabic_indirect_death_wish_routes_to_support() -> None:
    result = CrisisRouter(SafetyConfig()).assess("أتمنى أن أنام ولا أستيقظ غدًا.")
    assert result.route is SafetyRoute.SUPPORT


def test_arabic_imminent_self_harm_with_means_routes_to_crisis() -> None:
    result = CrisisRouter(SafetyConfig()).assess("سأقتل نفسي الآن ولدي سكين بجانبي.")
    assert result.route is SafetyRoute.CRISIS


def test_arabic_imminent_violence_with_weapon_routes_to_crisis() -> None:
    result = CrisisRouter(SafetyConfig()).assess("سأقتل الشخص الذي أغضبني الآن والسلاح معي.")
    assert result.route is SafetyRoute.CRISIS


def test_arabic_fictional_violence_context_is_not_routed_to_crisis() -> None:
    result = CrisisRouter(SafetyConfig()).assess(
        "في رواية يقول البطل سأقتل الشخص الذي أغضبني الآن والسلاح معي"
    )
    assert result.route is SafetyRoute.SUPPORT
