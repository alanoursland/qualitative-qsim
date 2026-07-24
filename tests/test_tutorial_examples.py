"""Execute the Python fences in the two tutorial tracks.

Lessons are intentionally cumulative: later snippets in a lesson reuse names
introduced earlier, and core lessons occasionally reuse the running model
from the preceding lesson. Output and pseudocode must therefore use a
non-Python fence rather than relying on an unexecuted Python snippet.
"""

from __future__ import annotations

import re
from pathlib import Path


TUTORIAL = Path(__file__).parents[1] / "docs" / "tutorial"
PYTHON_FENCE = re.compile(
    r"^```python\s*\n(.*?)^```\s*$",
    flags=re.MULTILINE | re.DOTALL,
)


def test_all_tutorial_python_fences_execute_in_lesson_order():
    namespace = {"__name__": "__qrlib_tutorial__"}
    executed = 0

    for lesson in sorted(TUTORIAL.glob("[0-9][0-9]-*.md")):
        source = lesson.read_text(encoding="utf-8")
        for block_number, block in enumerate(PYTHON_FENCE.findall(source), 1):
            filename = f"{lesson}::python-block-{block_number}"
            exec(compile(block, filename, "exec"), namespace)
            executed += 1

    assert executed >= 50
