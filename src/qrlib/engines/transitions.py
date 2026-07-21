"""QSIM P- and I-transition tables.

Per-variable successor rules, derived from the intermediate/mean value
theorems applied to QSIM's "reasonable functions" (C1, finitely many
critical points). Transcribed from Kuipers (1986, Table 1; 1994, ch. 5);
row labels kept in comments for audit. Magnitudes use the rank encoding
(even = at landmark, odd = open interval above that landmark).

Two structural facts engines rely on:

- **P-transitions (leaving a time point) branch only from steadiness.**
  A variable at a landmark with nonzero direction *must* move into the
  adjacent interval; steadiness may persist or break either way.
- **I-transitions (leaving a time interval) branch.** A moving variable may
  reach the next landmark (arriving still moving, or becoming steady
  exactly there) or remain in its interval (still moving, or becoming
  steady at an unnamed value — the hook for landmark discovery, phase 2).

A variable *at a landmark* with nonzero direction is impossible over an
open time interval (it would leave immediately); the I-table returns no
successors for that invalid input.

A variable steady over an open interval is constant there; continuity of
the derivative forces it to arrive steady at the closing time point.
"""

from __future__ import annotations

from ..quantity import Qdir, QuantitySpace, QVal

__all__ = ["point_successors", "interval_successors"]


def point_successors(qv: QVal, space: QuantitySpace) -> list[QVal]:
    """Possible values on the open time interval following a time point.

    An empty result means the variable sits at a boundary landmark of a
    bounded space with an outward direction — the region-exit condition.
    """
    r, d = qv.mag, qv.dir
    top = space.num_ranks - 1
    if r % 2 == 0:  # at a landmark
        if d is Qdir.STD:
            out = [QVal(r, Qdir.STD)]  # P1: remain steady at the landmark
            if r + 1 <= top:
                out.append(QVal(r + 1, Qdir.INC))  # P2: start rising into the interval above
            if r - 1 >= 0:
                out.append(QVal(r - 1, Qdir.DEC))  # P3: start falling into the interval below
            return out
        if d is Qdir.INC:
            # P4: must move into the interval above immediately
            return [QVal(r + 1, Qdir.INC)] if r + 1 <= top else []
        # P6: must move into the interval below immediately
        return [QVal(r - 1, Qdir.DEC)] if r - 1 >= 0 else []
    # in an open interval
    if d is Qdir.INC:
        return [QVal(r, Qdir.INC)]  # P5: keeps rising within the interval
    if d is Qdir.DEC:
        return [QVal(r, Qdir.DEC)]  # P7: keeps falling within the interval
    # momentarily steady inside an interval: any resumption is possible
    return [
        QVal(r, Qdir.INC),  # P8
        QVal(r, Qdir.STD),  # P9
        QVal(r, Qdir.DEC),  # P10
    ]


def interval_successors(qv: QVal, space: QuantitySpace) -> list[QVal]:
    """Possible values at the time point closing an open time interval."""
    r, d = qv.mag, qv.dir
    if r % 2 == 0:  # at a landmark throughout the interval
        # I1: constant at the landmark, arrives steady. Nonzero direction at
        # a landmark is not a valid interval state -> no successors.
        return [QVal(r, Qdir.STD)] if d is Qdir.STD else []
    if d is Qdir.INC:
        return [
            QVal(r + 1, Qdir.STD),  # I2: reaches the landmark above, becoming steady there
            QVal(r + 1, Qdir.INC),  # I3: reaches the landmark above, still rising
            QVal(r, Qdir.INC),      # I4: still rising within the interval
            QVal(r, Qdir.STD),      # I5: becomes steady at an unnamed value inside the interval
        ]
    if d is Qdir.DEC:
        return [
            QVal(r - 1, Qdir.STD),  # I6: reaches the landmark below, becoming steady there
            QVal(r - 1, Qdir.DEC),  # I7: reaches the landmark below, still falling
            QVal(r, Qdir.DEC),      # I8: still falling within the interval
            QVal(r, Qdir.STD),      # I9: becomes steady at an unnamed value inside the interval
        ]
    # steady (constant) inside an interval: arrives steady at the same value
    return [QVal(r, Qdir.STD)]
