"""Deterministic adversarial model generation for property tests.

The generator deliberately produces small, odd-but-valid models rather than
physically meaningful examples. Backend equivalence, serialization, resource
bounds, and determinism are public contracts for every valid model, not only
for textbook systems.

Seeds are part of the test contract. Keeping them deterministic makes failures
replayable and lets the suite assert that all intended constraint and regional
branches are actually represented.
"""

from __future__ import annotations

import random

import qrlib as qr


def random_model(seed: int) -> qr.Model:
    """Build a bounded-size valid QDE from ``seed``."""
    rng = random.Random(seed)
    count = 3 if seed % 4 == 0 else 2
    names = [f"v{index}" for index in range(count)]
    model = qr.Model(f"adversarial-{seed}")

    for name in names:
        style = rng.choice(
            ("zero-unbounded", "zero-up", "two-landmarks", "three-landmarks")
        )
        if style == "zero-unbounded":
            model.variable(
                name,
                landmarks=(qr.Landmark("0", value=0.0),),
                unbounded=True,
            )
        elif style == "zero-up":
            model.variable(
                name,
                landmarks=(qr.Landmark("0", value=0.0),),
                upper_unbounded=True,
            )
        elif style == "two-landmarks":
            model.variable(
                name,
                landmarks=(
                    qr.Landmark("0", value=0.0),
                    qr.Landmark("top", value=1.0),
                ),
            )
        else:
            model.variable(
                name,
                landmarks=(
                    qr.Landmark("bot", value=-1.0),
                    qr.Landmark("0", value=0.0),
                    qr.Landmark("top", value=1.0),
                ),
            )

    left, right = rng.sample(names, 2)
    model.constrain(qr.Deriv(left, right))

    for _ in range(rng.randint(1, 3)):
        kind = rng.choice(
            ("mplus", "mminus", "deriv", "constant", "minus", "add", "at")
        )
        if kind == "constant":
            model.constrain(qr.Constant(rng.choice(names)))
        elif kind == "at":
            variable = rng.choice(names)
            landmark = rng.choice(model.variables[variable].space.names)
            model.constrain(qr.At(variable, landmark))
        elif kind == "add" and count >= 3:
            model.constrain(qr.Add(*rng.sample(names, 3)))
        else:
            first, second = rng.sample(names, 2)
            constructors = {
                "mplus": lambda a, b: qr.MPlus(
                    a,
                    b,
                    cvals=(("0", "0"),),
                ),
                "mminus": lambda a, b: qr.MMinus(
                    a,
                    b,
                    cvals=(("0", "0"),),
                ),
                "deriv": qr.Deriv,
                "minus": qr.Minus,
                # A two-variable seed cannot host Add, so retain a
                # corresponding-value constraint instead.
                "add": lambda a, b: qr.MPlus(
                    a,
                    b,
                    cvals=(("0", "0"),),
                ),
            }
            model.constrain(constructors[kind](first, second))

    if seed % 5 == 0:
        model.region("active", constraints=tuple(model.constraints))
        model.region("relaxed", constraints=(model.constraints[0],))
        model.initial_region = "active"

    return model


def consistent_point_states(
    model: qr.Model,
    *,
    limit: int = 2,
) -> tuple[qr.QState, ...]:
    """Return non-vacuous initial states from the model's total portrait."""
    portrait = qr.envision(
        model,
        region=model.initial_region,
        max_states=20_000,
    )
    states = tuple(
        node.state
        for node in portrait.nodes
        if node.state.time is qr.TimeTag.POINT
    )[:limit]
    if not states:
        raise AssertionError(f"generated model {model.name!r} has no point states")
    return states


def constraint_kinds(model: qr.Model) -> frozenset[str]:
    return frozenset(type(constraint).__name__ for constraint in model.constraints)

