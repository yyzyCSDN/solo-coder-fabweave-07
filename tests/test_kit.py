from __future__ import annotations

import pytest

from fabweave.core.timebox import TimeBox
from fabweave.recipe.qualification import Qualification
from fabweave.workflows.kit import ChemicalLine, KitBook, KitRequest, PartLine


def _kit(
    request_id: str,
    job_id: str,
    window: TimeBox,
    chemicals: tuple[ChemicalLine, ...] = (),
    parts: tuple[PartLine, ...] = (),
    recipe_revision: int = 1,
    tool_id: str = "T1",
    chamber_id: str = "C1",
) -> KitRequest:
    return KitRequest(request_id, job_id, tool_id, chamber_id, recipe_revision, window, chemicals, parts)


def _qualifications(recipe_revision: int = 1, expires_at: int = 100) -> list[Qualification]:
    return [Qualification("T1", "C1", recipe_revision, 0, expires_at)]


def _chemical(lot_id: str, opened_at: int = 0, shelf_hours: int = 50, frozen: tuple[TimeBox, ...] = ()) -> ChemicalLine:
    return ChemicalLine("CHEM", lot_id, opened_at, shelf_hours, frozen, 1)


def _part(part_id: str, window: TimeBox, recipe_revision: int = 1) -> PartLine:
    return PartLine(part_id, window, recipe_revision)


def test_complete_set_reserves_chemicals_and_parts_together():
    book = KitBook()
    request = _kit(
        "REQ-1",
        "JOB-1",
        TimeBox(10, 20),
        chemicals=(_chemical("LOT-A"),),
        parts=(_part("RET-1", TimeBox(10, 20)),),
    )
    decision = book.reserve(request, _qualifications())
    assert decision.allowed
    assert decision.evidence["version"] == 1
    assert decision.evidence["chemical_count"] == 1
    assert decision.evidence["part_count"] == 1
    assert book.version == 1
    assert len(book.chemical_locks("LOT-A")) == 1
    assert len(book.part_reservations("RET-1")) == 1


def test_expired_chemical_denies_and_rolls_back_whole_set():
    book = KitBook()
    request = _kit(
        "REQ-1",
        "JOB-1",
        TimeBox(60, 70),
        chemicals=(_chemical("LOT-A", opened_at=0, shelf_hours=10),),
        parts=(_part("RET-1", TimeBox(60, 70)),),
    )
    decision = book.reserve(request, _qualifications())
    assert not decision.allowed
    assert "chemical_shelf_life_expired:LOT-A" in decision.reasons
    # Nothing from the failed request is locked.
    assert book.chemical_locks("LOT-A") == ()
    assert book.part_reservations("RET-1") == ()
    assert book.version == 0


def test_frozen_periods_pause_the_chemical_age_clock():
    book = KitBook()
    # 60 hours old, but 40 of those hours were frozen -> 20 hours effective age,
    # still within a 30-hour shelf life.
    request = _kit(
        "REQ-1",
        "JOB-1",
        TimeBox(60, 70),
        chemicals=(_chemical("LOT-A", opened_at=0, shelf_hours=30, frozen=(TimeBox(10, 50),)),),
    )
    decision = book.reserve(request, _qualifications())
    assert decision.allowed
    assert book.chemical_locks("LOT-A")[0].window == TimeBox(60, 70)


def test_chemical_lot_is_exclusive_between_jobs_in_overlapping_windows():
    book = KitBook()
    first = _kit("REQ-1", "JOB-1", TimeBox(10, 20), chemicals=(_chemical("LOT-A"),))
    assert book.reserve(first, _qualifications()).allowed
    second = _kit("REQ-2", "JOB-2", TimeBox(15, 25), chemicals=(_chemical("LOT-A"),))
    decision = book.reserve(second, _qualifications())
    assert not decision.allowed
    assert "chemical_conflict:LOT-A" in decision.reasons
    assert book.version == 1


def test_reticle_part_conflict_is_denied_without_partial_locks():
    book = KitBook()
    first = _kit("REQ-1", "JOB-1", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(10, 20)),))
    assert book.reserve(first, _qualifications()).allowed
    second = _kit(
        "REQ-2",
        "JOB-2",
        TimeBox(12, 18),
        chemicals=(_chemical("LOT-B"),),
        parts=(_part("RET-1", TimeBox(12, 18)),),
    )
    decision = book.reserve(second, _qualifications())
    assert not decision.allowed
    assert "part_conflict:RET-1" in decision.reasons
    # The chemical in the failed second request is also rolled back.
    assert book.chemical_locks("LOT-B") == ()
    assert book.version == 1


def test_touching_windows_do_not_conflict_half_open():
    book = KitBook()
    first = _kit("REQ-1", "JOB-1", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(0, 10)),))
    assert book.reserve(first, _qualifications()).allowed
    second = _kit("REQ-2", "JOB-2", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(10, 20)),))
    assert book.reserve(second, _qualifications()).allowed


def test_unqualified_recipe_denies_the_kit():
    book = KitBook()
    request = _kit("REQ-1", "JOB-1", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(10, 20)),), recipe_revision=3)
    decision = book.reserve(request, _qualifications(recipe_revision=1))
    assert not decision.allowed
    assert "qualification_missing" in decision.reasons
    assert book.part_reservations("RET-1") == ()


def test_replay_returns_the_original_decision():
    book = KitBook()
    request = _kit("REQ-1", "JOB-1", TimeBox(60, 70), chemicals=(_chemical("LOT-A", shelf_hours=10),))
    first = book.reserve(request, _qualifications())
    replay = book.reserve(request, _qualifications())
    assert first is replay
    assert book.version == 0  # rejected once, not re-evaluated


def test_replay_of_committed_kit_returns_original_and_does_not_double_lock():
    book = KitBook()
    request = _kit("REQ-1", "JOB-1", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(10, 20)),))
    first = book.reserve(request, _qualifications())
    replay = book.reserve(request, _qualifications())
    assert first is replay
    assert book.version == 1
    assert len(book.part_reservations("RET-1")) == 1


def test_request_id_reused_with_different_content_raises():
    book = KitBook()
    first = _kit("REQ-1", "JOB-1", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(10, 20)),))
    assert book.reserve(first, _qualifications()).allowed
    altered = _kit("REQ-1", "JOB-2", TimeBox(10, 20), parts=(_part("RET-2", TimeBox(10, 20)),))
    with pytest.raises(ValueError):
        book.reserve(altered, _qualifications())


def test_stale_version_is_a_transient_conflict_then_retry_succeeds():
    book = KitBook()
    first = _kit("REQ-1", "JOB-1", TimeBox(10, 20), parts=(_part("RET-1", TimeBox(10, 20)),))
    assert book.reserve(first, _qualifications(), expected_version=0).allowed

    # A concurrent caller still holding version 0 is rejected, not settled.
    concurrent = _kit("REQ-2", "JOB-2", TimeBox(10, 20), parts=(_part("RET-2", TimeBox(10, 20)),))
    stale = book.reserve(concurrent, _qualifications(), expected_version=0)
    assert not stale.allowed
    assert "stale_kit_version" in stale.reasons
    assert book.version == 1

    # Retrying against the current version commits.
    retry = book.reserve(concurrent, _qualifications(), expected_version=1)
    assert retry.allowed
    assert book.version == 2


def test_empty_kit_is_denied():
    book = KitBook()
    decision = book.reserve(_kit("REQ-1", "JOB-1", TimeBox(10, 20)), _qualifications())
    assert not decision.allowed
    assert "empty_kit" in decision.reasons


def test_structurally_invalid_lines_raise():
    with pytest.raises(ValueError):
        _chemical("LOT-A", shelf_hours=-1)
    with pytest.raises(ValueError):
        ChemicalLine("CHEM", "LOT-A", 0, 10, (), 0)
    with pytest.raises(ValueError):
        _part("RET-1", TimeBox(10, 20), recipe_revision=0)
