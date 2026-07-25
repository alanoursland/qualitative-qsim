"""Smoke-test an installed wheel without importing from the source tree.

Run this file with the clean environment's Python while the current working
directory is outside the repository.
"""

from __future__ import annotations

import ast
from importlib import metadata
import inspect
import os
from pathlib import Path
import pydoc
import warnings

import qrlib as qr


def _assert_installed_copy() -> None:
    source_root = os.environ.get("QRLIB_SOURCE_ROOT")
    if source_root:
        installed = Path(qr.__file__).resolve()
        source = Path(source_root).resolve()
        assert not installed.is_relative_to(source), (
            f"imported source checkout instead of installed wheel: {installed}"
        )


def _assert_metadata_links_are_portable() -> None:
    package = metadata.distribution("qualitative-qsim")
    description = package.metadata.get_payload()
    assert "](docs/" not in description
    assert "](paper." not in description
    assert (
        "https://github.com/alanoursland/qualitative_reasoning_lib/blob/HEAD/"
        in description
    )


def _assert_installed_docstrings_are_self_contained() -> None:
    package = metadata.distribution("qualitative-qsim")
    offenders = []
    for entry in package.files or ():
        if entry.suffix != ".py" or not entry.parts or entry.parts[0] != "qrlib":
            continue
        path = Path(package.locate_file(entry))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(entry))
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            doc = ast.get_docstring(node, clean=False) or ""
            if ".md" in doc or "docs/" in doc or "docs\\" in doc:
                offenders.append(str(entry))
                break
    assert not offenders, (
        "installed docstrings reference repository-only Markdown: "
        f"{sorted(offenders)}"
    )


def _exercise_public_entry_point() -> None:
    config = qr.SimConfig(max_landmarks_per_variable=5)
    assert config.max_states == 512
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert config.max_landmarks == 5
    assert any(
        issubclass(warning.category, DeprecationWarning)
        for warning in caught
    )

    model = qr.Model("wheel-smoke")
    model.variable("x", landmarks=("0",), upper_unbounded=True)
    model.constrain(qr.Constant("x"))
    initial = model.state(
        time=qr.TimeTag.POINT,
        x=("0", qr.Qdir.STD),
    )
    result = qr.qsim(model, initial)
    assert result.status is qr.SimStatus.COMPLETE
    assert result.to_dict()["config"]["profile"] == "practical"
    assert "qrlib" in pydoc.render_doc(qr)
    assert inspect.getdoc(qr)


if __name__ == "__main__":
    _assert_installed_copy()
    _assert_metadata_links_are_portable()
    _assert_installed_docstrings_are_self_contained()
    _exercise_public_entry_point()
    print(f"wheel smoke passed: qrlib {qr.__version__} from {qr.__file__}")
