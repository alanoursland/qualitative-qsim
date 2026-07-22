"""Alternative model-authoring front-ends.

The engine consumes :class:`~qrlib.model.Model`; these modules are ways of
*writing* one. ``qpt`` authors models as processes that influence
quantities (after Forbus's Qualitative Process Theory); ``devices``
composes models by wiring reusable component types together (after
de Kleer & Brown's ENVISION). Both compile down to ordinary models —
nothing in the engine knows they exist.
"""

from . import devices, qpt

__all__ = ["devices", "qpt"]
