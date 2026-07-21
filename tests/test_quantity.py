import pytest

from qrlib.quantity import QuantitySpace, QVal, Qdir


def test_rank_encoding_round_trip():
    space = QuantitySpace(("0", "FULL"), upper_unbounded=True)
    assert space.effective_landmarks == ("0", "FULL", "+inf")
    assert space.num_ranks == 5
    assert space.rank_of("0") == 0
    assert space.rank_between("0", "FULL") == 1
    assert space.rank_of("FULL") == 2
    assert space.rank_between("FULL", "+inf") == 3
    assert space.describe(1) == "(0, FULL)"
    assert space.describe(2) == "FULL"


def test_rank_between_requires_adjacency():
    space = QuantitySpace(("0", "FULL"), upper_unbounded=True)
    with pytest.raises(ValueError):
        space.rank_between("0", "+inf")


def test_sign_of_relative_to_zero():
    space = QuantitySpace(("0",), lower_unbounded=True, upper_unbounded=True)
    below = space.rank_between("-inf", "0")
    at = space.rank_of("0")
    above = space.rank_between("0", "+inf")
    assert space.sign_of(below) == -1
    assert space.sign_of(at) == 0
    assert space.sign_of(above) == 1


def test_insert_landmark_preserves_order():
    space = QuantitySpace(("0", "FULL"), upper_unbounded=True)
    grown = space.insert_landmark("EQ", after="0")
    assert grown.landmarks == ("0", "EQ", "FULL")
    assert grown.rank_of("EQ") == 2
    # original space untouched (spaces are values)
    assert space.landmarks == ("0", "FULL")


def test_space_validation():
    with pytest.raises(ValueError):
        QuantitySpace(())
    with pytest.raises(ValueError):
        QuantitySpace(("0", "0"))
    with pytest.raises(ValueError):
        QuantitySpace(("0", "+inf"))


def test_qdir_sign():
    assert Qdir.DEC.sign == -1
    assert Qdir.STD.sign == 0
    assert Qdir.INC.sign == 1


def test_qval_describe():
    space = QuantitySpace(("0",), upper_unbounded=True)
    assert QVal(space.rank_between("0", "+inf"), Qdir.INC).describe(space) == "(0, +inf)↑"
