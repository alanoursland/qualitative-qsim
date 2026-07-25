"""Installed-docstring contracts."""

import importlib
import inspect
import pkgutil

import qrlib


def _public_documented_objects(module):
    yield module.__name__, module
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if getattr(value, "__module__", None) != module.__name__:
            continue
        if not (inspect.isclass(value) or inspect.isfunction(value)):
            continue
        yield f"{module.__name__}.{name}", value
        if inspect.isclass(value):
            for member_name, member in vars(value).items():
                if member_name.startswith("_"):
                    continue
                yield f"{module.__name__}.{name}.{member_name}", member


def test_public_docstrings_do_not_reference_repository_markdown():
    """Installed help must not depend on files that the wheel omits."""
    modules = [qrlib]
    modules.extend(
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(
            qrlib.__path__, prefix=f"{qrlib.__name__}."
        )
    )
    offenders = []
    for module in modules:
        for qualified_name, value in _public_documented_objects(module):
            doc = inspect.getdoc(value) or ""
            if ".md" in doc or "docs/" in doc or "docs\\" in doc:
                offenders.append(qualified_name)
    assert not offenders, (
        "public docstrings reference repository Markdown absent from the "
        f"installed wheel: {offenders}"
    )
