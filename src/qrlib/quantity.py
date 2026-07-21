"""Quantity spaces and qualitative values.

The foundational representations of QSIM-style qualitative reasoning:

- :class:`QuantitySpace` — a totally ordered tuple of named landmark values,
  optionally unbounded below/above (conceptual ``-inf`` / ``+inf`` endpoints).
- :class:`Qdir` — direction of change (decreasing / steady / increasing).
- :class:`QVal` — a qualitative value: a magnitude (at a landmark, or in the
  open interval between adjacent landmarks) paired with a direction.

Magnitudes use the *rank encoding*: for a space whose effective landmark list
(including any conceptual infinities) is ``l_0 < l_1 < ... < l_{n-1}``, rank
``2*i`` means "at landmark l_i" and rank ``2*i + 1`` means "in the open
interval (l_i, l_{i+1})". Ordering and adjacency of magnitudes are then plain
integer arithmetic, and the same encoding serializes directly into the
integer tensors used by ``qrlib.tensor``.

Note (QSIM landmark discovery): quantity spaces are immutable values here;
inserting a new landmark produces a *new* space. Engines that discover
landmarks mid-simulation therefore version their spaces per behavior branch
(see docs/qsim.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

__all__ = ["Qdir", "QuantitySpace", "QVal"]

NEG_INF = "-inf"
POS_INF = "+inf"


class Qdir(IntEnum):
    """Direction of change of a variable. ``sign`` maps to {-1, 0, +1}."""

    DEC = 0
    STD = 1
    INC = 2

    @property
    def sign(self) -> int:
        return int(self) - 1


@dataclass(frozen=True)
class QuantitySpace:
    """A totally ordered set of named landmarks.

    ``landmarks`` are the *finite*, named landmarks in increasing order
    (e.g. ``("0", "FULL")``). ``lower_unbounded`` / ``upper_unbounded`` add
    conceptual ``-inf`` / ``+inf`` endpoints, which behave as ordinary
    (unreachable-in-finite-time) landmarks for encoding purposes.
    """

    landmarks: tuple[str, ...]
    lower_unbounded: bool = False
    upper_unbounded: bool = False

    def __post_init__(self) -> None:
        if not self.landmarks:
            raise ValueError("a QuantitySpace needs at least one named landmark")
        if len(set(self.landmarks)) != len(self.landmarks):
            raise ValueError(f"duplicate landmark names: {self.landmarks!r}")
        for reserved in (NEG_INF, POS_INF):
            if reserved in self.landmarks:
                raise ValueError(
                    f"{reserved!r} is implicit; use lower_unbounded/upper_unbounded"
                )

    @property
    def effective_landmarks(self) -> tuple[str, ...]:
        """Landmark names including any conceptual infinities."""
        pre = (NEG_INF,) if self.lower_unbounded else ()
        post = (POS_INF,) if self.upper_unbounded else ()
        return pre + self.landmarks + post

    @property
    def num_ranks(self) -> int:
        """Number of distinct qualitative magnitudes (landmarks + gaps)."""
        return 2 * len(self.effective_landmarks) - 1

    def rank_of(self, landmark: str) -> int:
        """Rank of the magnitude 'at ``landmark``'."""
        return 2 * self.effective_landmarks.index(landmark)

    def rank_between(self, lower: str, upper: str) -> int:
        """Rank of the open interval between two *adjacent* landmarks."""
        eff = self.effective_landmarks
        i, j = eff.index(lower), eff.index(upper)
        if j != i + 1:
            raise ValueError(f"{lower!r} and {upper!r} are not adjacent in {eff}")
        return 2 * i + 1

    def is_landmark_rank(self, rank: int) -> bool:
        self._check_rank(rank)
        return rank % 2 == 0

    def describe(self, rank: int) -> str:
        """Human-readable magnitude, e.g. ``"FULL"`` or ``"(0, FULL)"``."""
        self._check_rank(rank)
        eff = self.effective_landmarks
        if rank % 2 == 0:
            return eff[rank // 2]
        return f"({eff[rank // 2]}, {eff[rank // 2 + 1]})"

    def sign_of(self, rank: int, *, zero: str = "0") -> int:
        """Sign of a magnitude relative to the ``zero`` landmark (-1/0/+1).

        Raises ``ValueError`` if the space has no such landmark; constraints
        that need sign algebra require one.
        """
        self._check_rank(rank)
        zero_rank = self.rank_of(zero)
        return (rank > zero_rank) - (rank < zero_rank)

    def insert_landmark(self, name: str, *, after: str) -> "QuantitySpace":
        """A new space with ``name`` inserted just above landmark ``after``.

        This is the primitive behind QSIM new-landmark discovery.
        """
        if name in self.effective_landmarks:
            raise ValueError(f"landmark {name!r} already exists")
        i = self.landmarks.index(after) + 1 if after in self.landmarks else None
        if i is None:
            if after == NEG_INF and self.lower_unbounded:
                i = 0
            else:
                raise ValueError(f"unknown landmark {after!r}")
        return QuantitySpace(
            self.landmarks[:i] + (name,) + self.landmarks[i:],
            lower_unbounded=self.lower_unbounded,
            upper_unbounded=self.upper_unbounded,
        )

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.num_ranks:
            raise ValueError(f"rank {rank} out of range for {self}")


@dataclass(frozen=True)
class QVal:
    """A qualitative value: magnitude rank + direction, within some space."""

    mag: int
    dir: Qdir

    def describe(self, space: QuantitySpace) -> str:
        arrow = {Qdir.DEC: "↓", Qdir.STD: "·", Qdir.INC: "↑"}[self.dir]
        return f"{space.describe(self.mag)}{arrow}"
