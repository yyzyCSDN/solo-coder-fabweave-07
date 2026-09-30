from __future__ import annotations

from dataclasses import dataclass

from fabweave.chemicals.shelf_life import shelf_life_remaining
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.events import EventLog
from fabweave.core.timebox import TimeBox
from fabweave.recipe.qualification import Qualification, qualification_decision


@dataclass(frozen=True, slots=True)
class ChemicalLine:
    """One opened chemical lot consumed by a job."""

    chemical_id: str
    lot_id: str
    opened_at: int
    shelf_hours: int
    frozen_periods: tuple[TimeBox, ...]
    quantity: int

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("chemical quantity must be positive")
        if self.shelf_hours < 0:
            raise ValueError("shelf hours must be non-negative")
        if self.opened_at < 0:
            raise ValueError("opening time must be non-negative")


@dataclass(frozen=True, slots=True)
class PartLine:
    """One key part (reticle) reserved for a job's time slot."""

    part_id: str
    window: TimeBox
    recipe_revision: int

    def __post_init__(self) -> None:
        if self.recipe_revision <= 0:
            raise ValueError("recipe revision must be positive")


@dataclass(frozen=True, slots=True)
class KitRequest:
    """A complete-set demand: chemicals and key parts reserved together."""

    request_id: str
    job_id: str
    tool_id: str
    chamber_id: str
    recipe_revision: int
    window: TimeBox
    chemicals: tuple[ChemicalLine, ...]
    parts: tuple[PartLine, ...]


@dataclass(frozen=True, slots=True)
class _ChemicalLock:
    lot_id: str
    job_id: str
    request_id: str
    window: TimeBox


@dataclass(frozen=True, slots=True)
class _PartReservation:
    part_id: str
    job_id: str
    request_id: str
    window: TimeBox
    recipe_revision: int


class _Rollback(Exception):
    """Raised when staging a commit would violate a settled constraint."""


class KitBook:
    """Atomic, idempotent complete-set reservation for chemicals and key parts.

    A request is validated in full against committed state before anything is
    locked.  Only when every line is usable does the book swap in the new state;
    a failed request leaves committed state untouched (whole-set rollback).
    Replaying a settled request returns the original decision instead of
    re-evaluating it.  Optimistic concurrency is exposed through ``version``.
    """

    def __init__(self) -> None:
        self._version = 0
        self._chemical_locks: list[_ChemicalLock] = []
        self._part_reservations: list[_PartReservation] = []
        self._requests: dict[str, KitRequest] = {}
        self._decisions: dict[str, Decision] = {}
        self._log = EventLog()

    @property
    def version(self) -> int:
        return self._version

    def reserve(
        self,
        request: KitRequest,
        qualifications: list[Qualification],
        expected_version: int | None = None,
    ) -> Decision:
        # Replay: a settled request returns its original decision.
        prior = self._requests.get(request.request_id)
        if prior is not None:
            if prior != request:
                raise ValueError("request id reused with different content")
            return self._decisions[request.request_id]

        # Optimistic concurrency: a stale version is a transient conflict, not a
        # settled outcome, so it is not cached and a retry can still succeed.
        if expected_version is not None and expected_version != self._version:
            return deny(
                "stale_kit_version",
                current_version=self._version,
                expected_version=expected_version,
            )

        decision = self._evaluate(request, qualifications)
        if not decision.allowed:
            self._settle(request, decision)
            return decision

        try:
            chemical_locks = [*self._chemical_locks]
            part_reservations = [*self._part_reservations]
            for line in request.chemicals:
                lock = _ChemicalLock(line.lot_id, request.job_id, request.request_id, request.window)
                if self._chemical_conflict(lock, chemical_locks):
                    raise _Rollback
                chemical_locks.append(lock)
            for part in request.parts:
                reservation = _PartReservation(
                    part.part_id, request.job_id, request.request_id, part.window, part.recipe_revision
                )
                if self._part_conflict(reservation, part_reservations):
                    raise _Rollback
                part_reservations.append(reservation)
        except _Rollback:
            # Nothing was swapped in; committed state is untouched.
            decision = deny("kit_commit_aborted", request_id=request.request_id)
            self._settle(request, decision)
            return decision

        self._chemical_locks = chemical_locks
        self._part_reservations = part_reservations
        self._version += 1
        committed = allow(
            request_id=request.request_id,
            job_id=request.job_id,
            version=self._version,
            chemical_count=len(request.chemicals),
            part_count=len(request.parts),
        )
        self._settle(request, committed, kind="kit_reserved")
        return committed

    def chemical_locks(self, lot_id: str) -> tuple[_ChemicalLock, ...]:
        return tuple(lock for lock in self._chemical_locks if lock.lot_id == lot_id)

    def part_reservations(self, part_id: str) -> tuple[_PartReservation, ...]:
        return tuple(item for item in self._part_reservations if item.part_id == part_id)

    def _evaluate(self, request: KitRequest, qualifications: list[Qualification]) -> Decision:
        reasons: list[str] = []
        evidence: dict[str, object] = {"request_id": request.request_id, "job_id": request.job_id}

        if not request.chemicals and not request.parts:
            reasons.append("empty_kit")

        for line in request.chemicals:
            if line.opened_at > request.window.start:
                reasons.append(f"chemical_not_yet_opened:{line.lot_id}")
                continue
            remaining = shelf_life_remaining(
                line.opened_at, line.shelf_hours, request.window.start, list(line.frozen_periods)
            )
            if remaining < 0:
                reasons.append(f"chemical_shelf_life_expired:{line.lot_id}")
                evidence[f"remaining_shelf_hours:{line.lot_id}"] = remaining
                continue
            lock = _ChemicalLock(line.lot_id, request.job_id, request.request_id, request.window)
            if self._chemical_conflict(lock, self._chemical_locks):
                reasons.append(f"chemical_conflict:{line.lot_id}")

        for part in request.parts:
            reservation = _PartReservation(
                part.part_id, request.job_id, request.request_id, part.window, part.recipe_revision
            )
            if self._part_conflict(reservation, self._part_reservations):
                reasons.append(f"part_conflict:{part.part_id}")

        qualification = qualification_decision(
            request.tool_id,
            request.chamber_id,
            request.recipe_revision,
            request.window.start,
            qualifications,
        )
        reasons.extend(qualification.reasons)
        evidence.update(qualification.evidence)

        return deny(*reasons, **evidence) if reasons else allow(**evidence)

    @staticmethod
    def _chemical_conflict(lock: _ChemicalLock, existing: list[_ChemicalLock]) -> bool:
        return any(
            prior.lot_id == lock.lot_id
            and prior.job_id != lock.job_id
            and lock.window.overlaps(prior.window)
            for prior in existing
        )

    @staticmethod
    def _part_conflict(reservation: _PartReservation, existing: list[_PartReservation]) -> bool:
        return any(
            prior.part_id == reservation.part_id
            and prior.job_id != reservation.job_id
            and reservation.window.overlaps(prior.window)
            for prior in existing
        )

    def _settle(self, request: KitRequest, decision: Decision, kind: str = "kit_rejected") -> None:
        self._requests[request.request_id] = request
        self._decisions[request.request_id] = decision
        self._log.append(
            f"kit:{request.request_id}",
            0,
            request.request_id,
            request.window.start,
            kind,
            {"allowed": decision.allowed, "reasons": list(decision.reasons)},
        )
