from caep.evaluation.culture_mapping import (
    classify_culture_proposal,
    detect_culture_components,
    load_culture_taxonomy,
)


def test_simple_paracas_is_strictly_eligible() -> None:
    taxonomy = load_culture_taxonomy()

    components = detect_culture_components(
        "Peru, South Coast, Paracas",
        taxonomy,
    )

    result = classify_culture_proposal(
        "Peru, South Coast, Paracas",
        components,
        taxonomy,
    )

    assert components == ["Paracas"]
    assert result["strict_eligible"] is True
    assert (
        result["proposal_decision"]
        == "propuesta_aceptable"
    )


def test_nasca_wari_is_composite() -> None:
    taxonomy = load_culture_taxonomy()

    components = detect_culture_components(
        "Nasca-Wari",
        taxonomy,
    )

    result = classify_culture_proposal(
        "Nasca-Wari",
        components,
        taxonomy,
    )

    assert components == ["Nasca", "Wari"]
    assert result["is_composite"] is True
    assert result["strict_eligible"] is False


def test_uncertain_paracas_or_nasca_is_not_eligible() -> None:
    taxonomy = load_culture_taxonomy()

    label = "Peru, South Coast, Paracas or Nasca?"
    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Paracas", "Nasca"]
    assert result["is_uncertain"] is True
    assert result["strict_eligible"] is False


def test_peruvian_is_unattributed() -> None:
    taxonomy = load_culture_taxonomy()

    components = detect_culture_components(
        "Peruvian",
        taxonomy,
    )

    result = classify_culture_proposal(
        "Peruvian",
        components,
        taxonomy,
    )

    assert components == []
    assert result["is_unattributed"] is True
    assert (
        result["proposal_decision"]
        == "excluir_no_atribuida"
    )


def test_chimu_inka_is_composite() -> None:
    taxonomy = load_culture_taxonomy()

    components = detect_culture_components(
        "Peru, Chimú or Chimú-Inka",
        taxonomy,
    )

    assert components == ["Inca", "Chimu"]

def test_or_signal_does_not_match_middle_horizon() -> None:
    taxonomy = load_culture_taxonomy()

    label = (
        "Peru, South Coast, Wari Culture, "
        "Middle Horizon, 8th-12th Century"
    )

    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Wari"]
    assert result["is_uncertain"] is False
    assert result["strict_eligible"] is True
    assert (
        result["proposal_decision"]
        == "propuesta_aceptable"
    )


def test_or_signal_does_not_match_north_coast() -> None:
    taxonomy = load_culture_taxonomy()

    label = "Peru, Moche, north coast"

    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Moche"]
    assert result["is_uncertain"] is False
    assert result["strict_eligible"] is True


def test_standalone_or_remains_uncertain() -> None:
    taxonomy = load_culture_taxonomy()

    label = "Peru, South Coast, Paracas or Nasca?"

    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Paracas", "Nasca"]
    assert result["is_uncertain"] is True
    assert result["strict_eligible"] is False


def test_question_mark_remains_uncertain() -> None:
    taxonomy = load_culture_taxonomy()

    label = "Peru, South Coast, Paracas, Carhua?"

    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Paracas"]
    assert result["is_uncertain"] is True
    assert result["strict_eligible"] is False

def test_lambayeque_sican_is_strictly_eligible() -> None:
    taxonomy = load_culture_taxonomy()

    label = (
        "Peru, North Coast, "
        "Lambayeque (Sicán) people"
    )

    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Lambayeque_Sican"]
    assert result["is_uncertain"] is False
    assert result["is_style_attribution"] is False
    assert result["strict_eligible"] is True
    assert (
        result["proposal_decision"]
        == "propuesta_aceptable"
    )


def test_chavin_style_requires_style_review() -> None:
    taxonomy = load_culture_taxonomy()

    label = (
        "Peru, South Coast, Ica Valley, "
        "Chavín style"
    )

    components = detect_culture_components(
        label,
        taxonomy,
    )

    result = classify_culture_proposal(
        label,
        components,
        taxonomy,
    )

    assert components == ["Chavin"]
    assert result["is_uncertain"] is False
    assert result["is_style_attribution"] is True
    assert result["strict_eligible"] is False
    assert (
        result["proposal_decision"]
        == "revisar_estilo"
    )