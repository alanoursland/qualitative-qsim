from qrlib.engines.transitions import interval_successors, point_successors
from qrlib.quantity import Qdir, QuantitySpace, QVal

SPACE = QuantitySpace(("0", "FULL"))  # ranks: 0=at 0, 1=(0,FULL), 2=at FULL
UNBOUNDED = QuantitySpace(
    ("0",), lower_unbounded=True, upper_unbounded=True
)  # ranks: 0=-inf, 1=(-inf,0), 2=at 0, 3=(0,inf), 4=+inf


def vals(qvs):
    return {(q.mag, q.dir) for q in qvs}


def test_point_from_landmark_std_branches_three_ways_interior():
    space = UNBOUNDED
    out = point_successors(QVal(2, Qdir.STD), space)  # at 0
    assert vals(out) == {(2, Qdir.STD), (3, Qdir.INC), (1, Qdir.DEC)}


def test_point_from_landmark_std_at_boundaries():
    out = point_successors(QVal(0, Qdir.STD), SPACE)  # at bottom landmark
    assert vals(out) == {(0, Qdir.STD), (1, Qdir.INC)}
    out = point_successors(QVal(2, Qdir.STD), SPACE)  # at top landmark
    assert vals(out) == {(2, Qdir.STD), (1, Qdir.DEC)}


def test_point_from_moving_landmark_is_forced():
    assert vals(point_successors(QVal(0, Qdir.INC), SPACE)) == {(1, Qdir.INC)}
    assert vals(point_successors(QVal(2, Qdir.DEC), SPACE)) == {(1, Qdir.DEC)}


def test_point_boundary_exit_has_no_successors():
    assert point_successors(QVal(2, Qdir.INC), SPACE) == []  # off the top
    assert point_successors(QVal(0, Qdir.DEC), SPACE) == []  # off the bottom


def test_point_from_interval_preserves_motion():
    assert vals(point_successors(QVal(1, Qdir.INC), SPACE)) == {(1, Qdir.INC)}
    assert vals(point_successors(QVal(1, Qdir.DEC), SPACE)) == {(1, Qdir.DEC)}


def test_point_from_interval_std_resumes_any_direction():
    out = point_successors(QVal(1, Qdir.STD), SPACE)
    assert vals(out) == {(1, Qdir.INC), (1, Qdir.STD), (1, Qdir.DEC)}


def test_interval_from_moving_interval_branches_four_ways():
    out = interval_successors(QVal(1, Qdir.INC), SPACE)
    assert vals(out) == {
        (2, Qdir.STD),  # reach landmark, become steady
        (2, Qdir.INC),  # reach landmark, still moving
        (1, Qdir.INC),  # still inside
        (1, Qdir.STD),  # steady at unnamed value
    }
    out = interval_successors(QVal(1, Qdir.DEC), SPACE)
    assert vals(out) == {
        (0, Qdir.STD),
        (0, Qdir.DEC),
        (1, Qdir.DEC),
        (1, Qdir.STD),
    }


def test_interval_from_steady_is_identity():
    assert vals(interval_successors(QVal(0, Qdir.STD), SPACE)) == {(0, Qdir.STD)}
    assert vals(interval_successors(QVal(1, Qdir.STD), SPACE)) == {(1, Qdir.STD)}


def test_interval_moving_at_landmark_is_invalid():
    assert interval_successors(QVal(0, Qdir.INC), SPACE) == []
