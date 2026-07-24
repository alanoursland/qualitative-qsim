"""Quantity spaces and qualitative values.

The foundational representations of QSIM-style qualitative reasoning:

- :class:`Landmark` — a named landmark value; the *name* is its canonical
  identity, with optional numeric knowledge attached (an exact ``value``
  and/or interval bounds ``(lo, hi)``). Purely qualitative machinery uses
  only names and order; abstraction, coverage checking, and
  semi-quantitative propagation use the numbers when present (see
  docs/host-integration.md, Surface 1).
- :class:`QuantitySpace` — a totally ordered tuple of landmarks, optionally
  unbounded below/above (conceptual ``-inf`` / ``+inf`` endpoints).
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

__all__ = ["Landmark", "Qdir", "QuantitySpace", "QVal"]

NEG_INF = "-inf"
POS_INF = "+inf"


class Qdir(IntEnum):
    """Direction of change of a variable. ``sign`` maps to {-1, 0, +1}.

    ``IGN`` marks a direction deliberately not tracked (chatter abstraction,
    ``SimConfig.ignore_qdir``): engines store it in states but never feed it
    to constraint predicates — candidates are generated and filtered with
    concrete directions and only then projected to ``IGN``.
    """

    DEC = 0
    STD = 1
    INC = 2
    IGN = 3

    @property
    def sign(self) -> int:
        return 0 if self is Qdir.IGN else int(self) - 1


@dataclass(frozen=True)
class Landmark:
    """A named landmark, optionally carrying numeric knowledge.

    ``value`` is the exact numeric location when known (e.g. a computed
    equilibrium level); ``lo``/``hi`` are interval bounds for
    semi-quantitative reasoning or uncertain estimates. Hosts that know a
    landmark's location only symbolically keep the expression on their side,
    keyed by ``name``, and pass ``value=None`` (or numeric bounds) here.
    """

    name: str
    value: float | None = None
    lo: float | None = None
    hi: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a Landmark needs a non-empty name")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"landmark {self.name!r}: lo {self.lo} > hi {self.hi}")
        if self.value is not None:
            if self.lo is not None and self.value < self.lo:
                raise ValueError(f"landmark {self.name!r}: value below lo bound")
            if self.hi is not None and self.value > self.hi:
                raise ValueError(f"landmark {self.name!r}: value above hi bound")


@dataclass(frozen=True)
class QuantitySpace:
    """A totally ordered set of landmarks.

    ``landmarks`` are the *finite* landmarks in increasing order, given as
    :class:`Landmark` objects or bare name strings (normalized to
    ``Landmark``). ``lower_unbounded`` / ``upper_unbounded`` add conceptual
    ``-inf`` / ``+inf`` endpoints, which behave as ordinary
    (unreachable-in-finite-time) landmarks for encoding purposes. Known
    numeric values must respect the declared order.
    """

    landmarks: tuple[Landmark | str, ...]
    lower_unbounded: bool = False
    upper_unbounded: bool = False

    def __post_init__(self) -> None:
        norm = tuple(
            lm if isinstance(lm, Landmark) else Landmark(str(lm))
            for lm in self.landmarks
        )
        object.__setattr__(self, "landmarks", norm)
        names = [lm.name for lm in norm]
        if not names:
            raise ValueError("a QuantitySpace needs at least one named landmark")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate landmark names: {names!r}")
        for reserved in (NEG_INF, POS_INF):
            if reserved in names:
                raise ValueError(
                    f"{reserved!r} is implicit; use lower_unbounded/upper_unbounded"
                )
        prev: Landmark | None = None
        for lm in norm:
            if lm.value is not None:
                if prev is not None and lm.value <= prev.value:
                    raise ValueError(
                        f"landmark values out of order: {prev.name!r}={prev.value} "
                        f"is not below {lm.name!r}={lm.value}"
                    )
                prev = lm

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(lm.name for lm in self.landmarks)

    @property
    def effective_landmarks(self) -> tuple[str, ...]:
        """Landmark names including any conceptual infinities."""
        pre = (NEG_INF,) if self.lower_unbounded else ()
        post = (POS_INF,) if self.upper_unbounded else ()
        return pre + self.names + post

    @property
    def num_ranks(self) -> int:
        """Number of distinct qualitative magnitudes (landmarks + gaps)."""
        return 2 * len(self.effective_landmarks) - 1

    def landmark(self, name: str) -> Landmark:
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        raise KeyError(name)

    def rank_of(self, landmark: str) -> int:
        """Rank of the magnitude 'at ``landmark``' (by name)."""
        return 2 * self.effective_landmarks.index(landmark)

    def rank_between(self, lower: str, upper: str) -> int:
        """Rank of the open interval between two *adjacent* landmarks."""
        eff = self.effective_landmarks
        i, j = eff.index(lower), eff.index(upper)
        if j != i + 1:
            raise ValueError(f"{lower!r} and {upper!r} are not adjacent in {eff}")
        return 2 * i + 1

    def rank_of_value(self, x: float, *, atol: float = 0.0) -> int:
        """Rank of the magnitude containing numeric value ``x``.

        Requires every named landmark to carry a numeric ``value``; a
        landmark is hit when ``|x - value| <= atol``, otherwise ``x`` falls
        in an interval. This is the scalar seed of the batched bucketization
        in ``qrlib.bridge.abstraction``.
        """
        values = [lm.value for lm in self.landmarks]
        if any(v is None for v in values):
            missing = [lm.name for lm in self.landmarks if lm.value is None]
            raise ValueError(f"landmarks without numeric values: {missing}")
        off = 1 if self.lower_unbounded else 0
        for i, v in enumerate(values):
            if abs(x - v) <= atol:
                return 2 * (i + off)
        below = sum(1 for v in values if v < x)
        if below == 0 and not self.lower_unbounded:
            raise ValueError(f"{x} lies below this space")
        if below == len(values) and not self.upper_unbounded:
            raise ValueError(f"{x} lies above this space")
        return 2 * (below - 1 + off) + 1

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

    def insert_landmark(self, landmark: Landmark | str, *, after: str) -> "QuantitySpace":
        """A new space with ``landmark`` inserted just above landmark ``after``.

        This is the primitive behind QSIM new-landmark discovery, and behind
        landmark intake from hosts (``qrlib.bridge.harvest``).
        """
        lm = landmark if isinstance(landmark, Landmark) else Landmark(str(landmark))
        if lm.name in self.effective_landmarks:
            raise ValueError(f"landmark {lm.name!r} already exists")
        if after in self.names:
            i = self.names.index(after) + 1
        elif after == NEG_INF and self.lower_unbounded:
            i = 0
        else:
            raise ValueError(f"unknown landmark {after!r}")
        return QuantitySpace(
            self.landmarks[:i] + (lm,) + self.landmarks[i:],
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

    def describe(self, space: QuantitySpace, *, ascii: bool = False) -> str:
        """Human-readable value, optionally restricted to ASCII glyphs."""
        arrows = (
            {Qdir.DEC: "v", Qdir.STD: "-", Qdir.INC: "^", Qdir.IGN: "?"}
            if ascii
            else {Qdir.DEC: "↓", Qdir.STD: "·", Qdir.INC: "↑", Qdir.IGN: "?"}
        )
        arrow = arrows[self.dir]
        return f"{space.describe(self.mag)}{arrow}"
