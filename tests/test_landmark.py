import pytest

from qrlib.quantity import Landmark, QuantitySpace


def test_strings_normalize_to_landmarks():
    space = QuantitySpace(("0", "FULL"), upper_unbounded=True)
    assert all(isinstance(lm, Landmark) for lm in space.landmarks)
    assert space.names == ("0", "FULL")
    assert space.landmark("FULL").value is None


def test_landmarks_carry_numeric_knowledge():
    space = QuantitySpace(
        (Landmark("0", value=0.0), Landmark("FULL", value=1.5, lo=1.4, hi=1.6)),
        upper_unbounded=True,
    )
    assert space.landmark("FULL").value == 1.5
    assert space.landmark("FULL").lo == 1.4
    # rank API is unchanged by numeric annotations
    assert space.rank_of("FULL") == 2


def test_landmark_validation():
    with pytest.raises(ValueError):
        Landmark("")
    with pytest.raises(ValueError):
        Landmark("x", lo=2.0, hi=1.0)
    with pytest.raises(ValueError):
        Landmark("x", value=5.0, hi=1.0)


def test_known_values_must_respect_order():
    with pytest.raises(ValueError):
        QuantitySpace((Landmark("a", value=2.0), Landmark("b", value=1.0)))
    # partial knowledge is fine as long as known values are ordered
    QuantitySpace((Landmark("a", value=1.0), "middle", Landmark("c", value=3.0)))


@pytest.mark.parametrize("field", ("value", "lo", "hi"))
@pytest.mark.parametrize("number", (float("nan"), float("inf"), float("-inf")))
def test_quantity_spaces_reject_non_finite_landmark_numbers(field, number):
    bad = Landmark("bad", **{field: number})
    with pytest.raises(ValueError, match="landmark 'bad'.*must be finite"):
        QuantitySpace((Landmark("0", value=0.0), bad))


def test_rank_of_value():
    space = QuantitySpace(
        (Landmark("0", value=0.0), Landmark("FULL", value=2.0)),
        upper_unbounded=True,
    )
    assert space.rank_of_value(0.0) == space.rank_of("0")
    assert space.rank_of_value(1.0) == space.rank_between("0", "FULL")
    assert space.rank_of_value(2.0000001, atol=1e-3) == space.rank_of("FULL")
    assert space.rank_of_value(5.0) == space.rank_between("FULL", "+inf")
    with pytest.raises(ValueError):  # below a bounded-below space
        space.rank_of_value(-1.0)
    with pytest.raises(ValueError):  # values not all known
        QuantitySpace(("0", "FULL")).rank_of_value(1.0)


def test_insert_landmark_accepts_landmark_objects():
    space = QuantitySpace((Landmark("0", value=0.0), Landmark("FULL", value=2.0)))
    grown = space.insert_landmark(Landmark("EQ", value=1.2), after="0")
    assert grown.names == ("0", "EQ", "FULL")
    assert grown.landmark("EQ").value == 1.2
    with pytest.raises(ValueError):  # inserted value must respect order
        space.insert_landmark(Landmark("BAD", value=9.0), after="0")
