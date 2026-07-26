"""Resource limits must cover result enumeration, not only graph search.

``SimConfig.max_states`` is a strict graph-node bound and
``SimResult.behaviors()`` is the primary result accessor. Envisionment merges
states into a smaller cyclic graph, but an unsafe path enumerator can still
materialize exponentially many paths.

The accessor cannot be interrupted in-process. Every probe therefore runs in
its own subprocess under both a wall-clock limit and a 2 GB RSS watchdog.
Do not replace the watchdog with an in-process timeout: an unguarded version
of this adversarial case previously consumed 15.9 GB before termination.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


BUDGET_SECONDS = 25.0
MEMORY_CAP_GB = 2.0
ROOT = Path(__file__).resolve().parents[1]

PROBE = textwrap.dedent(
    """
    import json
    import sys
    import time

    sys.path[:0] = [{src!r}, {tests!r}]

    import qrlib as qr
    from test_qsim_phase2 import damped_spring

    model, initial = damped_spring()
    result = qr.qsim(
        model,
        initial,
        config=qr.SimConfig(
            max_states={cap},
            discover_landmarks=False,
            envisionment={envisionment},
        ),
    )
    started = time.perf_counter()
    count = len(result.behaviors())
    print(json.dumps({{
        "nodes": len(result.graph.nodes),
        "behaviors": count,
        "seconds": time.perf_counter() - started,
    }}))
    """
)


def _probe(cap: int, envisionment: bool, budget: float = BUDGET_SECONDS):
    """Return metrics, or ``None`` when the child exceeds either guard."""
    import time

    import psutil

    source = PROBE.format(
        src=str(ROOT / "src"),
        tests=str(ROOT / "tests"),
        cap=cap,
        envisionment=envisionment,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    watched = psutil.Process(process.pid)
    deadline = time.monotonic() + budget

    while process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            process.wait()
            return None
        try:
            if watched.memory_info().rss > MEMORY_CAP_GB * 1024**3:
                process.kill()
                process.wait()
                return None
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)

    stdout, stderr = process.communicate()
    assert process.returncode == 0, f"probe failed:\n{stderr}"
    return json.loads(stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("cap", (50, 100, 150, 200, 400))
def test_bounded_simulation_yields_readable_result(cap):
    """A quickly completed simulation must also have a readable result."""
    metrics = _probe(cap, True)
    assert metrics is not None, (
        f"max_states={cap}: behaviors() exceeded {BUDGET_SECONDS}s or "
        f"{MEMORY_CAP_GB} GB"
    )


@pytest.mark.parametrize("cap", (50, 100, 150))
def test_behavior_count_is_not_wildly_superlinear_in_nodes(cap):
    """The bounded result must stay outside the explosive path regime."""
    metrics = _probe(cap, True)
    assert metrics is not None
    assert metrics["behaviors"] <= metrics["nodes"] ** 3


def test_envisionment_does_not_multiply_the_reported_result():
    """State merging must not multiply the public behavior result."""
    tree = _probe(150, False)
    envisionment = _probe(150, True)

    assert tree is not None
    assert envisionment is not None
    assert envisionment["nodes"] <= tree["nodes"] + 1
    assert envisionment["behaviors"] <= max(tree["behaviors"] * 10, 100)

